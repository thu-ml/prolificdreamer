import os
import glob
import tqdm
import imageio
import random
import tensorboardX
import numpy as np

import time

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
from typing import List, Optional, Tuple, Union

from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils import deprecate

import sys
sys.path.append("..")


class DDIMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler, v_pred = False, x_pred = False):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.v_pred = v_pred
        self.x_pred = x_pred

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        pose = None,
        shading = None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                if `True` or `False`, see documentation for `DDIMScheduler.step`. If `None`, nothing is passed
                downstream to the scheduler. So use `None` for schedulers which don't support this argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """

        if (
            generator is not None
            and isinstance(generator, torch.Generator)
            and generator.device.type != self.device.type
            and self.device.type != "mps"
        ):
            message = (
                f"The `generator` device is `{generator.device}` and does not match the pipeline "
                f"device `{self.device}`, so the `generator` will be ignored. "
                f'Please use `generator=torch.Generator(device="{self.device}")` instead.'
            )
            deprecate(
                "generator.device == 'cpu'",
                "0.12.0",
                message,
            )
            generator = None


        # Sample gaussian noise to begin loop
        if isinstance(self.unet.sample_size, int):
            image_shape = (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size)
        else:
            image_shape = (batch_size, self.unet.in_channels, *self.unet.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        rand_device = "cpu" if self.device.type == "mps" else self.device
        if isinstance(generator, list):
            shape = (1,) + image_shape[1:]
            image = [
                torch.randn(shape, generator=generator[i], device=rand_device, dtype=self.unet.dtype)
                for i in range(batch_size)
            ]
            image = torch.cat(image, dim=0).to(self.device)
        else:
            image = torch.randn(image_shape, generator=generator, device=rand_device, dtype=self.unet.dtype)
            image = image.to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            if pose is None:
                if shading is None:
                    model_output = self.unet(image, t).sample
                else:
                    model_output = self.unet(image, t, shading = shading).sample
            else:
                if shading is None:
                    model_output = self.unet(image, t, c=pose).sample
                else:
                    model_output = self.unet(image, t, c=pose, shading = shading).sample  
                                  
            if self.v_pred or self.x_pred:
                sqrt_alpha_prod = self.scheduler.alphas_cumprod.to(image.device)[t] ** 0.5
                sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod.to(image.device)[t]) ** 0.5
                if self.v_pred:
                    model_output = sqrt_alpha_prod * model_output + sqrt_one_minus_alpha_prod * image
                elif self.x_pred:
                    model_output = (image - sqrt_alpha_prod * model_output) / sqrt_one_minus_alpha_prod
            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

        return image


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        if error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = - torch.ones_like(i)
    xs = - (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    # directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()

@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


class Trainer(object):
    def __init__(self, 
		         argv, # command line args
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 guidance, # guidance network
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        
        self.argv = argv
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
    
        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        # guide model
        self.guidance = guidance

        # text prompt
        if self.guidance is not None:
            
            for p in self.guidance.parameters():
                p.requires_grad = False

            self.prepare_text_embeddings()
        
        else:
            self.text_z = None
        
    
        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)


        if not opt.lora:
            if opt.q_cond:
                from conditional_unet import CondUNet2DModel
                unetname = CondUNet2DModel
            else:
                from diffusers import UNet2DModel
                unetname = UNet2DModel
            self.unet = unetname(
                sample_size=64, # render height for NeRF in training, assert opt.h==opt.w==64
                in_channels=4,
                out_channels=4,
                layers_per_block=2,
                block_out_channels=(128, 256, 384, 512),
                down_block_types=(
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "AttnDownBlock2D",
                    "AttnDownBlock2D",
                ),
                up_block_types=(
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                ),
            )    
        else:
            # use lora
            from lora_unet import UNet2DConditionModel     
            from diffusers.loaders import AttnProcsLayers
            from diffusers.models.attention_processor import LoRAAttnProcessor
            import einops
            if not opt.v_pred:
                _unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="unet", low_cpu_mem_usage=False, device_map=None).to(device)
            else:
                _unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="unet", low_cpu_mem_usage=False, device_map=None).to(device)
            _unet.requires_grad_(False)
            lora_attn_procs = {}
            for name in _unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else _unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = _unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(_unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = _unet.config.block_out_channels[block_id]
                lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            _unet.set_attn_processor(lora_attn_procs)
            lora_layers = AttnProcsLayers(_unet.attn_processors)

            text_input = self.guidance.tokenizer(opt.text, padding='max_length', max_length=self.guidance.tokenizer.model_max_length, truncation=True, return_tensors='pt')
            with torch.no_grad():
                text_embeddings = self.guidance.text_encoder(text_input.input_ids.to(self.guidance.device))[0]
            
            class LoraUnet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.unet = _unet
                    self.sample_size = 64
                    self.in_channels = 4
                    self.device = device
                    self.dtype = torch.float32
                    self.text_embeddings = text_embeddings
                def forward(self,x,t,c=None,shading="albedo"):
                    textemb = einops.repeat(self.text_embeddings, '1 L D -> B L D', B=x.shape[0]).to(device)
                    return self.unet(x,t,encoder_hidden_states=textemb,c=c,shading=shading)
            self._unet = _unet
            self.lora_layers = lora_layers
            self.unet = LoraUnet().to(device)                     

        self.unet = self.unet.to(self.device)
        if not opt.lora:
            self.unet_optimizer = optim.Adam(self.unet.parameters(), lr=self.opt.unet_lr) # naive adam
        else:
            params = [
                {'params': self.lora_layers.parameters()},
                {'params': self._unet.camera_emb.parameters()},
                {'params': self._unet.lambertian_emb},
                {'params': self._unet.textureless_emb},
                {'params': self._unet.normal_emb},
            ] 
            self.unet_optimizer = optim.AdamW(params, lr=self.opt.unet_lr) # naive adam
        warm_up_lr_unet = lambda iter: iter / (self.opt.warm_iters*self.opt.K+1) if iter <= (self.opt.warm_iters*self.opt.K+1) else 1
        self.unet_scheduler = optim.lr_scheduler.LambdaLR(self.unet_optimizer, warm_up_lr_unet)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        # self.test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)


        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(f'[INFO] Opt: {opt}')
        self.log(f'[INFO] Cmdline: {self.argv}')
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        
        self.buffer_imgs = None
        self.buffer_poses = None


    def init_evalpose(self, loader):
        poses = None
        for data in loader:
            pose = data['pose']
            pose = pose.view(pose.shape[0],16).contiguous()
            if poses is None:
                poses = pose
            else:
                poses = torch.cat([poses, pose], dim = 0)
        return poses

    def add_buffer(self, latent, pose):
        pose = pose.view(pose.shape[0],16).contiguous()
        if self.buffer_imgs is None:
            self.buffer_imgs = latent
            self.buffer_poses = pose
        elif self.buffer_imgs.shape[0] < self.opt.buffer_size:
            self.buffer_imgs = torch.cat([self.buffer_imgs, latent], dim = 0)
            self.buffer_poses = torch.cat([self.buffer_poses, pose], dim = 0)
        else:
            self.buffer_imgs = torch.cat([self.buffer_imgs[1:], latent], dim = 0)
            self.buffer_poses = torch.cat([self.buffer_poses[1:], pose], dim = 0)            

    def sample_buffer(self, bs):
        idxs = torch.tensor(random.sample(range(self.opt.buffer_size), bs), device = self.device).long()
        s_imgs = torch.index_select(self.buffer_imgs, 0, idxs) 
        s_poses = torch.index_select(self.buffer_poses, 0, idxs)
        return s_imgs, s_poses
    
    # calculate the text embs.
    def prepare_text_embeddings(self):

        if self.opt.text is None:
            self.log(f"[WARN] text prompt is not provided.")
            self.text_z = None
            return

        if not self.opt.dir_text:
            self.text_z = self.guidance.get_text_embeds([self.opt.text], [self.opt.negative])
        else:
            self.text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                # construct dir-encoded text
                text = f"{self.opt.text}, {d} view"

                negative_text = f"{self.opt.negative}"

                # explicit negative dir-encoded text
                if self.opt.suppress_face:
                    if negative_text != '': negative_text += ', '

                    if d == 'back': negative_text += "face"
                    # elif d == 'front': negative_text += ""
                    elif d == 'side': negative_text += "face"
                    elif d == 'overhead': negative_text += "face"
                    elif d == 'bottom': negative_text += "face"
                
                text_z = self.guidance.get_text_embeds([text], [negative_text])
                self.text_z.append(text_z)

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp'] # [B, 4, 4]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if self.global_step < self.opt.albedo_iters+1:
            shading = 'albedo'
            ambient_ratio = 1.0
        else: 
            rand = random.random()
            if rand > 0.8: 
                shading = 'albedo'
                ambient_ratio = 1.0
            elif rand > 0.4 and (not self.opt.no_textureless): 
                shading = 'textureless'
                ambient_ratio = 0.1
            else: 
                if not self.opt.no_lambertian:
                    shading = 'lambertian'
                    ambient_ratio = 0.1
                else:
                    shading = 'albedo'
                    ambient_ratio = 1.0                    

        # if random.random() < self.opt.p_normal:
        #     shading = 'normal'
        #     ambient_ratio = 1.0
        # 
        light_d = None
        if self.opt.normal:
            shading = 'normal'
            ambient_ratio = 1.0     
            if self.opt.p_textureless > random.random():
                shading = 'textureless'
                ambient_ratio = 0.1             
                light_d = data['rays_o'].contiguous().view(-1, 3)[0] + 0.3 * torch.randn(3, device=rays_o.device, dtype=torch.float)
                light_d = safe_normalize(light_d)             
        if self.global_step < self.opt.normal_iters+1:
            as_latent = True
            shading = 'normal'
            ambient_ratio = 1.0                   
        else:
            as_latent = False

        bg_color = None
        if self.global_step > 2000:
            if random.random() > 0.5:
                bg_color = None # use bg_net
            else:
                bg_color = torch.rand(3).to(self.device) # single color random bg
        
        if self.opt.backbone == "particle":
            self.model.mytraining = True
        binarize = False
        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=False, light_d= light_d,perturb=True, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, binarize=binarize)
        if self.opt.backbone == "particle":
            self.model.mytraining = False

        pred_depth = outputs['depth'].reshape(B, 1, H, W)
        
        if as_latent:
            pred_rgb = torch.cat([outputs['image'], outputs['weights_sum'].unsqueeze(-1)], dim=-1).reshape(B, H, W, 4).permute(0, 3, 1, 2).contiguous()
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 3 if not self.opt.latent else 4).permute(0, 3, 1, 2).contiguous() # [1, 3, H, W]
        
        # text embeddings
        if self.opt.dir_text:
            dirs = data['dir'] # [B,]
            text_z = self.text_z[dirs]
        else:
            text_z = self.text_z
        

        q_unet = self.unet
        if self.opt.q_cond:
            pose = data['pose'].view(B, 16)
        else:
            pose = None


        grad_clip = None
        if self.opt.dynamic_clip:
            grad_clip = 2 + 6 * min(1, self.epoch/(100.0*self.opt.n_particles))

        t5 = False
        if self.opt.t5_iters != -1 and self.global_step >= self.opt.t5_iters:
            if self.global_step == self.opt.t5_iters:
                print("Change into tmax = 500 setting")
            t5 = True

        # encode pred_rgb to latents
        loss, pseudo_loss, latents = self.guidance.train_step(text_z, pred_rgb, self.opt.scale, q_unet, pose, shading = shading, as_latent=as_latent, t5=t5)

        # regularizations
        if not self.opt.dmtet:
            if self.opt.lambda_opacity > 0:
                loss_opacity = (outputs['weights_sum'] ** 2).mean()
                loss = loss + self.opt.lambda_opacity * loss_opacity

            if self.opt.lambda_entropy > 0:

                alphas = outputs['weights'].clamp(1e-5, 1 - 1e-5)
                # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
                loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()

                # lambda_entropy = self.opt.lambda_entropy * min(1, 2 * self.global_step / self.opt.iters)

                loss = loss + self.opt.lambda_entropy * loss_entropy

            if self.opt.lambda_orient > 0 and 'loss_orient' in outputs:
                loss_orient = outputs['loss_orient']
                loss = loss + self.opt.lambda_orient * loss_orient
        else:
            if self.opt.lambda_normal > 0:
                loss = loss + self.opt.lambda_normal * outputs['normal_loss']

            if self.opt.lambda_lap > 0:
                loss = loss + self.opt.lambda_lap * outputs['lap_loss']

        return pred_rgb, pred_depth, loss, pseudo_loss, latents, shading
    
    def post_train_step(self):

        if self.opt.backbone == 'grid':

            lambda_tv = min(1.0, self.global_step / 1000) * self.opt.lambda_tv
            # unscale grad before modifying it!
            # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            self.scaler.unscale_(self.optimizer)
            self.model.encoder.grad_total_variation(lambda_tv, None, self.model.bound)
        elif self.opt.backbone == "particle" and self.opt.lambda_tv > 0:
            self.scaler.unscale_(self.optimizer)
            self.model.encoders[self.model.idx].grad_total_variation(self.opt.lambda_tv, None, self.model.bound)       

    def eval_step(self, data, shading):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        # shading = data['shading'] if 'shading' in data else 'albedo'
        # ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        # light_d = data['light_d'] if 'light_d' in data else None

        if shading == "albedo":
            ambient_ratio = 1.0
            light_d = None
        elif shading == "lambertian":
            ambient_ratio = 0.1
            light_d = data['rays_o'].contiguous().view(-1, 3)[0]
            light_d = safe_normalize(light_d)
        elif shading == "textureless":
            ambient_ratio = 0.1
            light_d = data['rays_o'].contiguous().view(-1, 3)[0] + 0.3 * torch.randn(3, device=rays_o.device, dtype=torch.float)
            light_d = safe_normalize(light_d)
            # light_d = None
        elif shading == "normal":
            ambient_ratio = 1.0
            light_d = None            
        else:
            raise NotImplementedError()

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=False, bg_color=None, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading)
        if not self.opt.latent:
            pred_rgb = outputs['image'].reshape(B, H, W, 3)
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 4)
            with torch.no_grad():
                pred_rgb = self.guidance.decode_latents(pred_rgb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        pred_depth = outputs['depth'].reshape(B, H, W)

        # dummy 
        loss = torch.zeros([1], device=pred_rgb.device, dtype=pred_rgb.dtype)

        return pred_rgb, pred_depth, loss

    def test_step(self, data, bg_color=None, perturb=False, shading=None):  
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        mvp = data['mvp']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        # if bg_color is not None:
        #     bg_color = bg_color.to(rays_o.device)
        # else:
        #     bg_color = torch.ones(3, device=rays_o.device) # [3]
        bg_color = torch.ones(3, device=rays_o.device)
        # shading = data['shading'] if 'shading' in data else 'albedo'
        # ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        # light_d = data['light_d'] if 'light_d' in data else None
        if shading == "albedo":
            ambient_ratio = 1.0
            light_d = None
        elif shading == "lambertian":
            ambient_ratio = 0.1
            light_d = data['rays_o'].contiguous().view(-1, 3)[0]
            light_d = safe_normalize(light_d)
        elif shading == "textureless":
            ambient_ratio = 0.1
            light_d = data['rays_o'].contiguous().view(-1, 3)[0]
            light_d = safe_normalize(light_d)
        elif shading == "normal":
            ambient_ratio = 1.0
            light_d = None            
        else:
            raise NotImplementedError()
    
        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=perturb, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, bg_color=bg_color)

        if not self.opt.latent:
            pred_rgb = outputs['image'].reshape(B, H, W, 3)
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 4)
            with torch.no_grad():
                pred_rgb = self.guidance.decode_latents(pred_rgb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_mask = outputs['weights_sum'].reshape(B, H, W) > 0.95

        return pred_rgb, pred_depth, pred_mask

    def generate_point_cloud(self, loader):

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        all_points = []
        all_normals = []

        with torch.no_grad():

            for i, data in enumerate(loader):

                data['shading'] = 'normal' # to get normal as color
                
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_mask = self.test_step(data)

                pred_mask = preds_mask[0].detach().cpu().numpy().reshape(-1) # [H, W], bool
                pred_depth = preds_depth[0].detach().cpu().numpy().reshape(-1, 1) # [N, 1]

                normals = preds[0].detach().cpu().numpy() * 2 - 1 # normals in [-1, 1]
                normals = normals.reshape(-1, 3) # shape [N, 3]

                rays_o = data['rays_o'][0].detach().cpu().numpy() # [N, 3]
                rays_d = data['rays_d'][0].detach().cpu().numpy() # [N, 3]
                points = rays_o + pred_depth * rays_d

                if pred_mask.any():
                    all_points.append(points[pred_mask])
                    all_normals.append(normals[pred_mask])

                pbar.update(loader.batch_size)
        
        points = np.concatenate(all_points, axis=0)
        normals = np.concatenate(all_normals, axis=0)
            
        return points, normals

    def save_mesh(self, loader=None, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        # if loader is None: # mcubes
        #     self.model.export_mesh(save_path, resolution=self.opt.mcubes_resolution, decimate_target=self.opt.decimate_target)
        # else: # poisson (TODO: not working currently...)
        #     points, normals = self.generate_point_cloud(loader)
        #     self.model.export_mesh(save_path, points=points, normals=normals, decimate_target=self.opt.decimate_target)
        self.model.export_mesh(save_path, resolution=self.opt.mcubes_resolution, decimate_target=self.opt.decimate_target)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):

        assert self.text_z is not None, 'Training must provide a text prompt!'

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            if epoch >= self.opt.iter512 and self.opt.iter512 > 0:
                if epoch == self.opt.iter512:
                    print("Change into 512 resolution!")
                train_loader = self.train_loader512

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                pass
                # self.save_checkpoint(full=True, best=False)

            if self.epoch > 1000:
                continue

            if self.epoch % self.eval_interval == 0 or self.epoch < 2:
                self.evaluate_one_epoch(valid_loader, shading = "albedo")
                self.evaluate_one_epoch(valid_loader, shading = "normal")
                self.evaluate_one_epoch(valid_loader, shading = "textureless")
                if not self.opt.albedo or self.opt.p_normal > 0:           
                    self.evaluate_one_epoch(valid_loader, shading = "lambertian")
                if self.epoch < 402:
                    self.save_checkpoint(full=False, best=False)
                    # self.save_checkpoint(full=False, best=True)

            unet_bs = 8 if not self.opt.lora else 2

            if (self.epoch % self.eval_interval == 0 or self.epoch == 1 or self.epoch < 2) and self.opt.K > 0:
                pipeline = DDIMPipeline(unet=self.unet, scheduler=self.guidance.scheduler, v_pred = self.opt.v_pred)
                with torch.no_grad():
                    images = pipeline(batch_size=unet_bs, output_type="numpy", shading = "albedo")
                    rgb = self.guidance.decode_latents(images)
                img = rgb.detach().permute(0,2,3,1).cpu().numpy()
                img = torch.tensor(img.transpose(0,3,1,2), dtype=torch.float32)
                torchvision.utils.save_image(img, os.path.join(self.workspace, 'validation', f'{self.name}_ep{self.epoch:04d}' + "-unet-albedo.png"), normalize=True, range=(0,1))
                
                if not self.opt.albedo:
                    with torch.no_grad():
                        images = pipeline(batch_size=unet_bs, output_type="numpy", shading = "textureless")
                        rgb = self.guidance.decode_latents(images)
                    img = rgb.detach().permute(0,2,3,1).cpu().numpy()
                    img = torch.tensor(img.transpose(0,3,1,2), dtype=torch.float32)
                    torchvision.utils.save_image(img, os.path.join(self.workspace, 'validation', f'{self.name}_ep{self.epoch:04d}' + "-unet-textureless.png"), normalize=True, range=(0,1))
                    
                    if not self.opt.no_lambertian:
                        with torch.no_grad():
                            images = pipeline(batch_size=unet_bs, output_type="numpy", shading = "lambertian")
                            rgb = self.guidance.decode_latents(images)
                        img = rgb.detach().permute(0,2,3,1).cpu().numpy()
                        img = torch.tensor(img.transpose(0,3,1,2), dtype=torch.float32)
                        torchvision.utils.save_image(img, os.path.join(self.workspace, 'validation', f'{self.name}_ep{self.epoch:04d}' + "-unet-lambertian.png"), normalize=True, range=(0,1))
                            
                # if self.opt.p_normal > 0:
                with torch.no_grad():
                    images = pipeline(batch_size=unet_bs, output_type="numpy", shading = "normal")
                    rgb = self.guidance.decode_latents(images)
                img = rgb.detach().permute(0,2,3,1).cpu().numpy()
                img = torch.tensor(img.transpose(0,3,1,2), dtype=torch.float32)
                torchvision.utils.save_image(img, os.path.join(self.workspace, 'validation', f'{self.name}_ep{self.epoch:04d}' + "-unet-normal.png"), normalize=True, range=(0,1))
                            

                # poses = self.init_evalpose(valid_loader)
                # with torch.no_grad():
                #     images = pipeline(batch_size=poses.shape[0], output_type="numpy", pose = poses, shading = "albedo")
                #     rgb = self.guidance.decode_latents(images)
                # img = rgb.detach().permute(0,2,3,1).cpu().numpy()
                # img = torch.tensor(img.transpose(0,3,1,2), dtype=torch.float32)
                # torchvision.utils.save_image(img, os.path.join(self.workspace, 'validation', f'{self.name}_ep{self.epoch:04d}' + "-unet-pose.png"), normalize=True, range=(0,1))
             
                       
                if self.buffer_imgs is not None and self.buffer_imgs.shape[0] == self.opt.buffer_size:
                    _, poses = self.sample_buffer(8)
                    with torch.no_grad():
                        images = pipeline(batch_size=8, output_type="numpy", pose = poses)
                        rgb = self.guidance.decode_latents(images)
                    img = rgb.detach().permute(0,2,3,1).cpu().numpy()
                    img = torch.tensor(img.transpose(0,3,1,2), dtype=torch.float32)
                    torchvision.utils.save_image(img, os.path.join(self.workspace, 'validation', f'{self.name}_ep{self.epoch:04d}' + "-unet-cond.png"), normalize=True, range=(0,1))

            if self.epoch % self.opt.test_interval == 0:
                self.save_checkpoint(full=False, best=True)
                if self.opt.backbone == 'particle':
                    for idx in range(self.opt.n_particles):
                        self.model.set_idx(idx)
                        for shading in ["textureless", "albedo", "normal"]:
                            self.test(self.test_loader, idx=idx, shading = shading)   
                        # break 
                else:
                    self.test(self.test_loader)

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        raise NotImplementedError()

    def test(self, loader, save_path=None, name=None, write_video=True, idx = 0, shading = None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, _ = self.test_step(data, shading=shading)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                pred_depth = (pred_depth * 255).astype(np.uint8)
                pred_depth = cv2.applyColorMap(pred_depth, cv2.COLORMAP_JET)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                # else:
                    # if i % 3 == 0:
                    #     cv2.imwrite(os.path.join(save_path, f'img_{name}_{idx:02d}_{i:06d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    #     cv2.imwrite(os.path.join(save_path, f'img_{name}_{idx:02d}_{i:06d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            
            imageio.mimwrite(os.path.join(save_path, f'{name}_{idx:02d}_{shading}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            if shading == "albedo":
                imageio.mimwrite(os.path.join(save_path, f'{name}_{idx:02d}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")
    
    def train_one_epoch(self, loader):
        self.log(f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            self.model.set_idx()
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, pred_depths, loss, pseudo_loss, latents, shading = self.train_step(data)

            self.scaler.scale(loss).backward()
            self.post_train_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.opt.buffer_size != -1:
                self.add_buffer(latents, data['pose'])


            assert self.opt.q_cond
            if self.global_step % self.opt.K2 == 0 and not self.opt.sds:
                for _ in range(self.opt.K):
                    self.unet_optimizer.zero_grad()
                    timesteps = torch.randint(0, 1000, (self.opt.unet_bs,), device=self.device).long() # temperarily hard-coded for simplicity
                    with torch.no_grad():
                        if self.buffer_imgs is None or self.buffer_imgs.shape[0]<self.opt.buffer_size:
                            latents_clean = latents.expand(self.opt.unet_bs, latents.shape[1], latents.shape[2], latents.shape[3]).contiguous()
                            if self.opt.q_cond:
                                pose = data['pose']
                                pose = pose.view(pose.shape[0], 16)
                                pose = pose.expand(self.opt.unet_bs, 16).contiguous()
                                if random.random() < self.opt.uncond_p:
                                    pose = torch.zeros_like(pose)
                        else:
                            latents_clean, pose = self.sample_buffer(self.opt.unet_bs)
                            if random.random() < self.opt.uncond_p:
                                pose = torch.zeros_like(pose)
                    noise = torch.randn(latents_clean.shape, device=self.device)
                    latents_noisy = self.guidance.scheduler.add_noise(latents_clean, noise, timesteps)
                    if self.opt.q_cond:
                        model_output = self.unet(latents_noisy, timesteps, c = pose, shading = shading).sample
                    else:
                        model_output = self.unet(latents_noisy, timesteps).sample
                    if self.opt.v_pred:
                        loss_unet = F.mse_loss(model_output, self.guidance.scheduler.get_velocity(latents_clean, noise, timesteps))
                    else:
                        loss_unet = F.mse_loss(model_output, noise)
                    loss_unet.backward()
                    self.unet_optimizer.step()
                    if self.scheduler_update_every_step:
                        self.unet_scheduler.step()                    

            if self.scheduler_update_every_step:
                pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
            else:
                pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
            pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None, shading = None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0
            pre_imgs = None
            pre_depths = None
            for idx in range(self.opt.val_nz):
                if self.opt.backbone == 'particle':
                    self.model.set_idx(idx)
                for data in loader:    
                    self.local_step += 1

                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        preds, preds_depth, loss = self.eval_step(data, shading)

                    if self.world_size > 1:
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        loss = loss / self.world_size
                        
                        preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                        dist.all_gather(preds_list, preds)
                        preds = torch.cat(preds_list, dim=0)

                        preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                        dist.all_gather(preds_depth_list, preds_depth)
                        preds_depth = torch.cat(preds_depth_list, dim=0)
                    
                    loss_val = loss.item()
                    total_loss += loss_val

                    if self.local_rank == 0:
                        save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)

                        if pre_imgs == None:
                            pre_imgs = preds
                        else:
                            pre_imgs = torch.cat([pre_imgs, preds], dim = 0)
                        if pre_depths == None:
                            pre_depths = preds_depth
                        else:
                            pre_depths = torch.cat([pre_depths, preds_depth], dim = 0)

                        pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                        pbar.update(loader.batch_size)
                if not (self.opt.backbone == 'particle'):
                    break
            if self.local_rank == 0:
                torchvision.utils.save_image(pre_imgs.permute(0,3,1,2), os.path.join(self.workspace, 'validation', f'{self.name}_ep{self.epoch:06d}' + "-rgb-"+shading+".png"), nrow=self.opt.val_size, normalize=True, range=(0,1))
                if shading == "albedo":
                    torchvision.utils.save_image(pre_depths.unsqueeze(1), os.path.join(self.workspace, 'validation', f'{self.name}_ep{self.epoch:06d}' + "-depth.png"), nrow=self.opt.val_size, normalize=True)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            # state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if self.opt.dmtet:
            state['tet_scale'] = self.model.tet_scale

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:
            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:    
            state['model'] = self.model.state_dict()
            file_path = f"best_{name}.pth"
            torch.save(state, os.path.join(self.ckpt_path, file_path))

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict({(k):v for k,v in checkpoint_dict['model'].items()}, strict=False)

        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            # if 'mean_count' in checkpoint_dict:
            #     self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if self.opt.dmtet:
            if 'tet_scale' in checkpoint_dict:
                self.model.verts *= checkpoint_dict['tet_scale'] / self.model.tet_scale
                self.model.tet_scale = checkpoint_dict['tet_scale']

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
