import ipdb
import torch
import kaolin
import torchvision
import numpy as np
from dmtet_network import Decoder

# ------------ get normal mesh -----------------

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)

class Mesh:
    def __init__(self, v_pos=None, t_pos_idx=None, v_nrm=None, t_nrm_idx=None, v_tex=None, t_tex_idx=None, v_tng=None, t_tng_idx=None, 
    v_weights=None, bone_mtx=None, material=None, base=None):
        self.v_pos = v_pos
        self.v_weights = v_weights
        self.v_nrm = v_nrm
        self.v_tex = v_tex
        self.v_tng = v_tng
        self.t_pos_idx = t_pos_idx
        self.t_nrm_idx = t_nrm_idx
        self.t_tex_idx = t_tex_idx
        self.t_tng_idx = t_tng_idx
        self.material = material
        self.bone_mtx = bone_mtx

        if base is not None:
            self.copy_none(base)

    def copy_none(self, other):
        if self.v_pos is None:
            self.v_pos = other.v_pos
        if self.v_weights is None:
            self.v_weights = other.v_weights
        if self.t_pos_idx is None:
            self.t_pos_idx = other.t_pos_idx
        if self.v_nrm is None:
            self.v_nrm = other.v_nrm
        if self.t_nrm_idx is None:
            self.t_nrm_idx = other.t_nrm_idx
        if self.v_tex is None:
            self.v_tex = other.v_tex
        if self.t_tex_idx is None:
            self.t_tex_idx = other.t_tex_idx
        if self.v_tng is None:
            self.v_tng = other.v_tng
        if self.t_tng_idx is None:
            self.t_tng_idx = other.t_tng_idx
        if self.material is None:
            self.material = other.material
        if self.bone_mtx is None:
            self.bone_mtx = other.bone_mtx

    def get_frames(self):
        return self.bone_mtx.shape[0] if self.bone_mtx is not None else 1

    def clone(self):
        out = Mesh(base=self)
        if out.v_pos is not None:
            out.v_pos = out.v_pos.clone()
        if out.v_weights is not None:
            out.v_weights = out.v_weights.clone()
        if out.t_pos_idx is not None:
            out.t_pos_idx = out.t_pos_idx.clone()
        if out.v_nrm is not None:
            out.v_nrm = out.v_nrm.clone()
        if out.t_nrm_idx is not None:
            out.t_nrm_idx = out.t_nrm_idx.clone()
        if out.v_tex is not None:
            out.v_tex = out.v_tex.clone()
        if out.t_tex_idx is not None:
            out.t_tex_idx = out.t_tex_idx.clone()
        if out.v_tng is not None:
            out.v_tng = out.v_tng.clone()
        if out.t_tng_idx is not None:
            out.t_tng_idx = out.t_tng_idx.clone()
        if out.bone_mtx is not None:
            out.bone_mtx = out.bone_mtx.clone()
        return out

    def eval(self, params={}):
        return self

def auto_normals(mesh):
    class mesh_op_auto_normals:
        def __init__(self, input):
            self.input = input

        def eval(self, params={}):
            imesh = self.input.eval(params)

            i0 = imesh.t_pos_idx[:, 0]
            i1 = imesh.t_pos_idx[:, 1]
            i2 = imesh.t_pos_idx[:, 2]

            v0 = imesh.v_pos[i0, :]
            v1 = imesh.v_pos[i1, :]
            v2 = imesh.v_pos[i2, :]

            face_normals = torch.cross(v1 - v0, v2 - v0)

            # Splat face normals to vertices
            v_nrm = torch.zeros_like(imesh.v_pos)
            v_nrm.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
            v_nrm.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
            v_nrm.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

            # Normalize, replace zero (degenerated) normals with some default value
            v_nrm = torch.where(dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))

            self.v_nrm = safe_normalize(v_nrm)

            if torch.is_anomaly_enabled():
                assert torch.all(torch.isfinite(self.v_nrm))

            return Mesh(v_nrm = self.v_nrm, t_nrm_idx=imesh.t_pos_idx, base = imesh)

    return mesh_op_auto_normals(mesh)

def simple_normals(verts, faces):


    i0 = faces[:, 0]
    i1 = faces[:, 1]
    i2 = faces[:, 2]

    v0 = verts[i0, :]
    v1 = verts[i1, :]
    v2 = verts[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(verts)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))

    v_nrm = safe_normalize(v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_nrm))

    return v_nrm

# -------------- dmtet ------------------

# path to the point cloud to be reconstructed
pcd_path = "./dmtet/data/bear_pointcloud.usd"
# path to the output logs (readable with the training visualizer in the omniverse app)
logs_path = './dmtet/logs/'

# We initialize the timelapse that will store USD for the visualization apps
timelapse = kaolin.visualize.Timelapse(logs_path)


device = 'cuda'
lr = 1e-3
laplacian_weight = 0.1
iterations = 200
save_every = 100
multires = 2
grid_res = 128



points = kaolin.io.usd.import_pointclouds(pcd_path)[0].points.to(device)
if points.shape[0] > 100000:
    idx = list(range(points.shape[0]))
    np.random.shuffle(idx)
    idx = torch.tensor(idx[:100000], device=points.device, dtype=torch.long)    
    points = points[idx]

# The reconstructed object needs to be slightly smaller than the grid to get watertight surface after MT.
center = (points.max(0)[0] + points.min(0)[0]) / 2
max_l = (points.max(0)[0] - points.min(0)[0]).max()
points = ((points - center) / max_l)* 0.9
timelapse.add_pointcloud_batch(category='input',
                               pointcloud_list=[points.cpu()], points_type = "usd_geom_points")


# # Loading the Tetrahedral Grid
# 
# DMTet starts from a uniform tetrahedral grid of predefined resolution, and uses a network to predict the SDF value as well as deviation vector at each grid vertex. 
# 
# Here we load the pre-generated tetrahedral grid using [Quartet](https://github.com/crawforddoran/quartet) at resolution 128, which has roughly the same number of vertices as a voxel grid of resolution 65. We use a simple MLP + positional encoding to predict the SDF and deviation vectors in DMTet, and initialize the encoded SDF to represent a sphere. 

tet_verts = torch.tensor(np.load('./dmtet/data/{}_verts.npz'.format(grid_res))['data'], dtype=torch.float, device=device)
tets = torch.tensor(([np.load('./dmtet/data/{}_tets_{}.npz'.format(grid_res, i))['data'] for i in range(4)]), dtype=torch.long, device=device).permute(1,0)
print (tet_verts.shape, tets.shape)

# Initialize model and create optimizer
model = Decoder(multires=multires).to(device)
model.pre_train_sphere(1000)


# # Preparing the Losses and Regularizer
# 
# During training we will use two losses defined on the surface mesh:
# - We use Chamfer Distance as the reconstruction loss. At each step, we randomly sample points from the surface mesh and compute the point-to-point distance to the GT point cloud.
# - DMTet can employ direct regularization on the surface mesh to impose useful geometric constraints. We demonstrate this with a Laplacian loss which encourages the surface to be smooth.
# 

# Laplacian regularization using umbrella operator (Fujiwara / Desbrun).
# https://mgarland.org/class/geom04/material/smoothing.pdf
def laplace_regularizer_const(mesh_verts, mesh_faces):
    term = torch.zeros_like(mesh_verts)
    norm = torch.zeros_like(mesh_verts[..., 0:1])

    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)

def loss_f(mesh_verts, mesh_faces, points, it):
    pred_points = kaolin.ops.mesh.sample_points(mesh_verts.unsqueeze(0), mesh_faces, 50000)[0][0]
    chamfer = kaolin.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), points.unsqueeze(0)).mean()
    if it > iterations//2:
        lap = laplace_regularizer_const(mesh_verts, mesh_faces)
        return chamfer + lap * laplacian_weight
    return chamfer



# # Setting up Optimizer

vars = [p for _, p in model.named_parameters()]
optimizer = torch.optim.Adam(vars, lr=lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10**(-x*0.0002))) # LR decay over time


# # Training
# 
# At every iteration, we first predict SDF and deviation vector at each vertex with the network. Next, we extract the triangular mesh by running Marching Tetrahedra on the grid. We then compute loss functions on the extracted mesh and backpropagate gradient to the network weights. Notice that the topology of the mesh is changing during training, as shown in the output message. The training takes ~5 minutes on a TITAN RTX GPU.


for it in range(iterations):
    pred = model(tet_verts) # predict SDF and per-vertex deformation
    sdf, deform = pred[:,0], pred[:,1:]
    verts_deformed = tet_verts + torch.tanh(deform) / grid_res # constraint deformation to avoid flipping tets
    mesh_verts, mesh_faces = kaolin.ops.conversions.marching_tetrahedra(verts_deformed.unsqueeze(0), tets, sdf.unsqueeze(0)) # running MT (batched) to extract surface mesh
    mesh_verts, mesh_faces = mesh_verts[0], mesh_faces[0]

    loss = loss_f(mesh_verts, mesh_faces, points, it)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if (it) % save_every == 0 or it == (iterations - 1): 
        print ('Iteration {} - loss: {}, # of mesh vertices: {}, # of mesh faces: {}'.format(it, loss, mesh_verts.shape[0], mesh_faces.shape[0]))
        # save reconstructed mesh
        timelapse.add_mesh_batch(
            iteration=it+1,
            category='extracted_mesh',
            vertices_list=[mesh_verts.cpu()],
            faces_list=[mesh_faces.cpu()]
        )

# # Visualize Training
# 
# You can now use [the Omniverse app](https://docs.omniverse.nvidia.com/app_kaolin/app_kaolin) to visualize the mesh optimization over training by using the training visualizer on "./logs/", where we stored the checkpoints.
# 
# Alternatively, you can use [kaolin-dash3d](https://kaolin.readthedocs.io/en/latest/notes/checkpoints.html?highlight=usd#visualizing-with-kaolin-dash3d) to visualize the checkpoint by running <code>kaolin-dash3d --logdir=$logs_path --port=8080</code>. This command will launch a web server that will stream geometry to web clients. You can view the input point cloud and the reconstructed mesh at [localhost:8080](localhost:8080) as shown below. You can change the *global iteration* on the left to see how the mesh evolves during training. 
# 

# -------------------
mesh_normals = simple_normals(mesh_verts, mesh_faces)

# opt_base_mesh = Mesh(v_pos_opt, normalized_base_mesh.t_pos_idx, material=opt_material, base=normalized_base_mesh)

# # Scale from [-1, 1] local coordinate space to match extents of the reference mesh
# opt_base_mesh = mesh.align_with_reference(opt_base_mesh, ref_mesh)

# # Compute smooth vertex normals
# opt_base_mesh = mesh.auto_normals(opt_base_mesh)

# color_opt = render.render_mesh(glctx, _opt_detail, mvp, campos, lightpos, FLAGS.light_power, iter_res, 
#     spp=iter_spp, num_layers=FLAGS.layers, msaa=True , background=randomBgColor, 
#     min_roughness=FLAGS.min_roughness)
# img_loss = image_loss_fn(color_opt, color_ref)

# -----------dr-------------

import nvdiffrast.torch as dr
import util
# Transform vertex positions to clip space
def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def render(glctx, mtx, pos, pos_idx, vtx_col, col_idx, resolution: int):
    pos_clip    = transform_pos(mtx, pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    color, _    = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
    color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
    return color

def make_grid(arr, ncols=2):
    n, height, width, nc = arr.shape
    nrows = n//ncols
    assert n == nrows*ncols
    return arr.reshape(nrows, ncols, height, width, nc).swapaxes(1,2).reshape(height*nrows, width*ncols, nc)

log_file = None
writer = None

# Create position/triangle index tensors
# pos_idx = torch.from_numpy(mesh_faces.astype(np.int32)).cuda()
# col_idx = torch.from_numpy(mesh_faces.astype(np.int32)).cuda()
# vtx_pos = torch.from_numpy(mesh_verts.astype(np.float32)).cuda()
# vtx_col = torch.from_numpy(mesh_normals.astype(np.float32)).cuda()
pos_idx = mesh_faces.int()
col_idx = mesh_faces.int()
vtx_pos = mesh_verts
vtx_col = mesh_normals


# Rasterizer context
glctx = dr.RasterizeGLContext() if False else dr.RasterizeCudaContext()

# Repeats.

ang = 0.0
gl_avg = []

# vtx_pos_rand = np.random.uniform(-0.5, 0.5, size=mesh_verts.shape) + mesh_verts
# vtx_col_rand = np.random.uniform(0.0, 1.0, size=mesh_normals.shape)
# vtx_pos_opt  = torch.tensor(vtx_pos_rand, dtype=torch.float32, device='cuda', requires_grad=True)
# vtx_col_opt  = torch.tensor(vtx_col_rand, dtype=torch.float32, device='cuda', requires_grad=True)

# Adam optimizer for vertex position and color with a learning rate ramp.
# optimizer    = torch.optim.Adam([vtx_pos_opt, vtx_col_opt], lr=1e-2)
# scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.01, 10**(-x*0.0005)))


# Random rotation/translation matrix for optimization.
r_rot = util.random_rotation_translation(0.25)

# Smooth rotation for display.
a_rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(ang))

# Modelview and modelview + projection matrices.
proj  = util.projection(x=0.4)
r_mv  = np.matmul(util.translate(0, 0, -3.5), r_rot)
r_mvp = np.matmul(proj, r_mv).astype(np.float32)
a_mv  = np.matmul(util.translate(0, 0, -3.5), a_rot)
a_mvp = np.matmul(proj, a_mv).astype(np.float32)

# Compute geometric error for logging.
# with torch.no_grad():
#     geom_loss = torch.mean(torch.sum((torch.abs(vtx_pos_opt) - .5)**2, dim=1)**0.5)
#     gl_avg.append(float(geom_loss))

# Print/save log.

# gl_val = np.mean(np.asarray(gl_avg))
# gl_avg = []
# s = ("rep=%d," % rep) if repeats > 1 else ""
# s += "iter=%d,err=%f" % (it, gl_val)
# print(s)
# if log_file:
#     log_file.write(s + "\n")
ipdb.set_trace()
color     = render(glctx, r_mvp, vtx_pos, pos_idx, vtx_col, col_idx, 256)
torchvision.utils.save_image(color.swapaxes(1,3).swapaxes(2,3),"./dmtet/norm.png")
