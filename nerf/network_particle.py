import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp, biased_softplus
from .renderer import NeRFRenderer

from encoding import get_encoder
import random
from .utils import safe_normalize


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x

class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 hidden_dim=32,
                 num_layers_bg=2,
                 hidden_dim_bg=16,
                 ):
        
        super().__init__(opt)

        self.opt = opt
        self.num_layers = opt.num_layers
        num_layers = self.num_layers
        self.hidden_dim = hidden_dim
        self.img_channels = 4 if opt.latent ==True else 3
        self.n_particles = opt.n_particles

        self.encoders = torch.nn.ModuleList([get_encoder('hashgrid', input_dim=3, log2_hashmap_size=19, desired_resolution=self.opt.desired_resolution * self.bound, interpolation='smoothstep')[0] for _ in range(self.n_particles)])
        self.in_dim = self.encoders[0].output_dim
        self.sigma_nets = torch.nn.ModuleList([MLP(self.in_dim, self.img_channels + 1, hidden_dim, num_layers, bias=True) for _ in range(self.n_particles)])
        self.normal_nets = torch.nn.ModuleList([MLP(self.in_dim, 3, hidden_dim, num_layers, bias=True) for _ in range(self.n_particles)])

        self.density_activation = trunc_exp if self.opt.density_activation == 'exp' else biased_softplus

        # background network
        if self.opt.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg

            if not self.opt.complex_bg:
                self.encoder_bgs = torch.nn.ModuleList([get_encoder('frequency', input_dim=3, multires=4)[0] for _ in range(self.n_particles)])
            else:
                self.encoder_bgs = torch.nn.ModuleList([get_encoder('frequency', input_dim=3, log2_hashmap_size=19, desired_resolution=self.opt.desired_resolution * self.bound, interpolation='smoothstep')[0] for _ in range(self.n_particles)])
            self.in_dim_bg = self.encoder_bgs[0].output_dim
            self.bg_nets = torch.nn.ModuleList([MLP(self.in_dim_bg, self.img_channels, hidden_dim_bg, num_layers_bg, bias=True) for _ in range(self.n_particles)])           
        else:
            self.bg_nets = None

        self.idx = None
        self.mytraining = False
    
    def set_idx(self, idx=None):
        if idx == None:
            self.idx = random.randint(0, self.n_particles-1)
        else:
            self.idx = idx

    # add a density blob to the scene center
    @torch.no_grad()
    def density_blob(self, x):
        # x: [B, N, 3]
        
        d = (x ** 2).sum(-1)
        g = self.opt.blob_density * (1 - torch.sqrt(d) / self.opt.blob_radius)

        if self.opt.upper_clip_m > -10:
            mask2 = torch.clamp(torch.sign(x[..., 1] - self.opt.upper_clip_m), min = 0)
            mask1 = torch.clamp(torch.sign(g), min = 0)
            mask = 1 - mask2 * mask1
        else:
            mask = 1.0

        return g * mask

    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        # print(x.shape)
        dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal

    def common_forward(self, x):
        enc = self.encoders[self.idx](x, bound=self.bound)

        h = self.sigma_nets[self.idx](enc)

        with torch.no_grad():
            if self.opt.sphere_mask:
                mask = torch.clamp(torch.sign(self.bound **2 - (x**2).sum(-1)), min = 0)
            else:
                mask = 1.0

        sigma_pre = h[..., 0]
        if self.mytraining == True and self.opt.pre_noise:
            sigma_pre = sigma_pre + torch.randn_like(sigma_pre)
        sigma = self.density_activation(sigma_pre + self.density_blob(x)) * mask
        if self.opt.latent:
            albedo = h[..., 1:]
        else:
            albedo = torch.sigmoid(h[..., 1:])

        return sigma, albedo

    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        sigma, albedo = self.common_forward(x)

        if shading == 'albedo':
            normal = None
            color = albedo
        
        else: # lambertian shading
            normal = self.finite_difference_normal(x)
            normal = safe_normalize(normal)
            normal = torch.nan_to_num(normal)

            lambertian = ratio + (1 - ratio) * (normal @ l).clamp(min=0) # [N,]
            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, self.img_channels)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else: # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)
            
        return sigma, color, normal

      
    def density(self, x):
        sigma, albedo = self.common_forward(x)
        
        return {
            'sigma': sigma,
            'albedo': albedo,
        }


    def background(self, d):
        h = self.encoder_bgs[self.idx](d)
        h = self.bg_nets[self.idx](h)

        if self.opt.latent:
            rgbs = h
        else:
            rgbs = torch.sigmoid(h)

        return rgbs

    def get_params(self, lr, finetune = False):
        params = [
            {'params': self.encoders.parameters(), 'lr': lr * 10},
            {'params': self.sigma_nets.parameters(), 'lr': lr},
        ]        

        if self.opt.bg_radius > 0:
            params.append({'params': self.encoder_bgs.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_nets.parameters(), 'lr': lr})

        if self.opt.dmtet and not finetune:
            params.append({'params': self.sdf, 'lr': lr * 5})
            params.append({'params': self.deform, 'lr': lr * 5})

        return params
