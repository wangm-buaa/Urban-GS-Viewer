import os
import json
import numpy as np
import torch
import math
import torchvision
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type
from torch.nn import Parameter
import torch.nn.functional as F
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.viewer.server.viewer_elements import *
from nerfstudio.fields.urban_gs_field import GaussianSplattingField
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from nerfstudio.utils.gaussian_splatting_sh_utils import eval_sh
from nerfstudio.cameras.gaussian_splatting_camera import Camera as GaussianSplattingCamera
from nerfstudio.utils.gaussian_splatting_graphics_utils import getWorld2View2, focal2fov, fov2focal
import yaml
from types import SimpleNamespace
from gsplat import rasterization, rasterization_2dgs
from gsplat.cuda._wrapper import fully_fused_projection, fully_fused_projection_2dgs
import matplotlib.cm as cm

def parse_cfg(cfg):
    lp = SimpleNamespace(**cfg.get('model_params', {}))
    op = SimpleNamespace(**cfg.get('optim_params', {}))
    pp = SimpleNamespace(**cfg.get('pipeline_params', {}))
    return lp, op, pp

@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    _target: Type = field(
        default_factory=lambda: GaussianSplatting
    )

    background_color: str = "white"

    sh_degree: int = 3


class PipelineParams():
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def prefilter_voxel(viewpoint_camera, pc, pre_visible_anchor=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    if pre_visible_anchor is None:
        means = pc.get_anchor[pc._anchor_mask]
        scales = pc.get_scaling[pc._anchor_mask][:, :3]
        quats = pc.get_rotation[pc._anchor_mask]
    else:
        # pre_fileter_mask = torch.bitwise_and(pc._anchor_mask, pre_visible_anchor)
        # idx = torch.nonzero(pre_fileter_mask, as_tuple=True)[0]
        # means = pc.get_anchor[idx]
        # scales = pc.get_scaling[idx][:, :3]
        # quats = pc.get_rotation[idx]
        if pre_visible_anchor.shape[0] != pc.get_anchor.shape[0]:
            anchor_mask = pc._anchor_mask[pre_visible_anchor]
            means = pc.get_anchor[pre_visible_anchor][anchor_mask]
            scales = pc.get_scaling_w_mask(pre_visible_anchor)[anchor_mask][:, :3]
            quats = pc.get_rotation_w_mask(pre_visible_anchor)[anchor_mask]
        else:
            pre_fileter_mask = torch.bitwise_and(pc._anchor_mask, pre_visible_anchor)
            means = pc.get_anchor[pre_fileter_mask]
            scales = pc.get_scaling_w_mask(pre_fileter_mask)[:, :3]
            quats = pc.get_rotation_w_mask(pre_fileter_mask)

    # Set up rasterization configuration
    Ks = torch.tensor([
        [viewpoint_camera.fx, 0, viewpoint_camera.cx],
        [0, viewpoint_camera.fy, viewpoint_camera.cy],
        [0, 0, 1],
    ], dtype=torch.float32, device="cuda")[None]
    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None]

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    if pc.gs_attr == "3D":
        proj_results = fully_fused_projection(
            means,
            None,  # covars,
            quats,
            scales,
            viewmats,
            Ks,
            int(viewpoint_camera.image_width),
            int(viewpoint_camera.image_height),
            eps2d=0.3,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            radius_clip=0.0,
            sparse_grad=False,
            calc_compensations=False,
        )
    elif pc.gs_attr == "2D":
        densifications = (
            torch.zeros((C, N, 2), dtype=means.dtype, device="cuda")
        )
        # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
        proj_results = fully_fused_projection_2dgs(
            means,
            quats,
            scales,
            viewmats,
            densifications,
            Ks,
            int(viewpoint_camera.image_width),
            int(viewpoint_camera.image_height),
            eps2d=0.3,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            radius_clip=0.0,
            sparse_grad=False,
        )
    else:
        raise ValueError(f"Unknown gs_attr: {pc.gs_attr}")

    # print("mask / rasterization: {:.3f}".format(mask_time/rasterization_time), " mask: {:.3f}, rasterization: {:.3f}".format(mask_time * 1000, rasterization_time * 1000))
    # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
    radii, means2d, depths, conics, compensations = proj_results
    camera_ids, gaussian_ids = None, None

    if pre_visible_anchor is None:
        visible_mask = pc._anchor_mask.clone()
        visible_mask[pc._anchor_mask] = radii.squeeze(0) > 0
        return visible_mask
    else:
        if pre_visible_anchor.shape[0] != pc.get_anchor.shape[0]:
            return pre_visible_anchor[pc._anchor_mask[pre_visible_anchor]][radii.squeeze(0) > 0]
        else:
            visible_mask = pre_fileter_mask.clone()
            visible_mask[pre_fileter_mask] = radii.squeeze(0) > 0
            return visible_mask
        # return radii.squeeze(0) > 0

def depthImgToPosCam_Batched(d, screenCoords, focal, princpt):
    p = screenCoords - princpt[:, :, None, None]
    x = (d * p[:, 0:1, :, :]) / focal[:, 0:1, 0, None, None]
    y = (d * p[:, 1:2, :, :]) / focal[:, 1:2, 1, None, None]
    return torch.cat([x, y, d], dim=1)

# p: b x 3 x H x W
# out: b x 3 x H x W
def computeNormalsFromPosCam_Batched(p):
    p = F.pad(p, (1, 1, 1, 1), "replicate")
    d0 = p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1]
    d1 = p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2]
    n = torch.cross(d0, d1, dim=1)
    norm = torch.norm(n, dim=1, keepdim=True)
    norm = norm + 1e-5
    norm[norm < 1e-5] = 1  # Can not backprop through this
    return -n / norm

def visualize_depth(depth, near=0.2, far=5, linear=False):
    depth = depth[0].clone().detach().cpu().numpy()
    colormap = cm.get_cmap('turbo')
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    if linear:
        curve_fn = lambda x: -x
    eps = np.finfo(np.float32).eps
    # near = near if near else depth.min()
    # far = far if far else depth.max()
    near = depth.min()
    far = depth.max()
    near -= eps
    far += eps
    # near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]
    out_depth = np.clip(np.nan_to_num(vis), 0., 1.) * 255
    out_depth = torch.from_numpy(out_depth).float().cuda() / 255
    return out_depth


def visualize_normal_from_depth(inputs, depth_p):
    # Normals
    uv = torch.stack(
        torch.meshgrid(
            torch.arange(depth_p.shape[2]), torch.arange(depth_p.shape[1]), indexing="xy"
        ),
        dim=0,
    )[None].float().cuda()
    position = depthImgToPosCam_Batched(
        depth_p[None, ...], uv, inputs["focal"], inputs["princpt"]
    )
    normal = 0.5 * (computeNormalsFromPosCam_Batched(position) + 1.0)
    normal = normal[0, [2, 1, 0], :, :]  # legacy code assumes BGR format

    return normal


class GaussianSplatting(Model):
    config: GaussianSplattingModelConfig
    model_path: str
    load_iteration: int
    ref_orientation: str
    orientation_transform: torch.Tensor
    gaussian_model: GaussianSplattingField

    def __init__(
            self,
            config: ModelConfig,
            scene_box: SceneBox,
            num_train_data: int,
            model_path: str = None,
            load_iteration: int = -1,
            orientation_transform: torch.Tensor = None,
    ) -> None:
        self.config = config
        self.model_path = model_path
        self.load_iteration = load_iteration
        self.orientation_transform = orientation_transform
        with open(os.path.join(self.model_path, "config.yaml")) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            lp, op, pp = parse_cfg(cfg)

        # load gaussian model
        self.model_config = lp.model_config
        self.pipeline_params = pp
        if self.config.background_color == "black":
            self.bg_color = [0, 0, 0]
        else:
            self.bg_color = [1, 1, 1]

        super().__init__(config, scene_box, num_train_data)

        self.scaling_modifier_slider = ViewerSlider(name="Scaling Modifier", default_value=1.0, min_value=0.0, max_value=1.0)

    def populate_modules(self):
        super().populate_modules()

        # get iteration
        if self.load_iteration == -1:
            self.load_iteration = self.search_for_max_iteration(os.path.join(self.model_path, "point_cloud"))
        print("Loading trained model at iteration {}".format(self.load_iteration))

        self.gaussian_model = GaussianSplattingField(**self.model_config['kwargs'])

        self.gaussian_model.load_ply(os.path.join(self.model_path,
                                                  "point_cloud",
                                                  "iteration_" + str(self.load_iteration),
                                                  "point_cloud.ply"))
        self.gaussian_model.load_mlp_checkpoints(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.load_iteration)))

    @staticmethod
    def search_for_max_iteration(folder):
        saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
        return max(saved_iters)

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        viewpoint_camera = self.ns2gs_camera(camera_ray_bundle.camera)
        # K = torch.tensor([
        #     [viewpoint_camera.fx, 0, viewpoint_camera.cx],
        #     [0, viewpoint_camera.fy, viewpoint_camera.cy],
        #     [0, 0, 1],
        # ], dtype=torch.float32, device="cuda")

        background = torch.tensor(self.bg_color, dtype=torch.float32, device=camera_ray_bundle.origins.device)

        render_results = self.render(
            viewpoint_camera=viewpoint_camera,
            pc=self.gaussian_model,
            pipe=self.pipeline_params,
            bg_color=background,
            scaling_modifier=self.scaling_modifier_slider.value,
        )

        render = render_results["render"]
        depth = render_results["depth"]
        invDepth = torch.where(depth > 0.0, 1.0 / depth, torch.zeros_like(depth))

        vis_depth = visualize_depth(invDepth)

        # vis_normal = visualize_normal_from_depth()
        rgb = torch.permute(torch.clamp(render, max=1.), (1, 2, 0))
        return {
            "rgb": rgb,
            "depth1": vis_depth,
            # "normal": vis_normal,
        }


    def ns2gs_camera(self, ns_camera):
        c2w = torch.clone(ns_camera.camera_to_worlds)
        c2w = torch.concatenate([c2w, torch.tensor([[0, 0, 0, 1]], device=ns_camera.camera_to_worlds.device)], dim=0)

        # reorient
        if self.orientation_transform is not None:
            c2w = torch.matmul(self.orientation_transform.to(c2w.device), c2w)

        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w.cpu().numpy())
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        FovY = focal2fov(ns_camera.fy, ns_camera.height)
        FovX = focal2fov(ns_camera.fx, ns_camera.width)

        return GaussianSplattingCamera(
            R=R,
            T=T,
            width=ns_camera.width,
            height=ns_camera.height,
            FoVx=FovX,
            FoVy=FovY,
            fx=ns_camera.fx,
            fy=ns_camera.fy,
            cx=ns_camera.cx,
            cy=ns_camera.cy,
            data_device=ns_camera.camera_to_worlds.device,
        )

    @staticmethod
    def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        pc.set_anchor_mask(viewpoint_camera.camera_center)
        visible_mask = prefilter_voxel(viewpoint_camera, pc).squeeze() if pipe.add_prefilter else pc._anchor_mask
        xyz, offset, color, opacity, scaling, rot, sh_degree, selection_mask = pc.generate_neural_gaussians(
            viewpoint_camera, visible_mask)

        K = torch.tensor([
            [viewpoint_camera.fx, 0, viewpoint_camera.cx],
            [0, viewpoint_camera.fy, viewpoint_camera.cy],
            [0, 0, 1],
        ], dtype=torch.float32, device="cuda")
        viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)  # [4, 4]
        gs_masks = torch.ones_like(opacity)
        render_colors, render_alphas, info = rasterization(
            means=xyz,  # [N, 3]
            quats=rot,  # [N, 4]
            scales=scaling,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=color,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            backgrounds=bg_color[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            gs_masks=gs_masks,
            packed=False,
            sh_degree=sh_degree,
            absgrad=True,
            radius_clip=0,
            rasterize_mode="classic",  # "classic", "antialiased"
            render_mode=pc.render_mode,
        )
        if render_colors.shape[-1] == 4:
            colors, depths = render_colors[..., 0:3], render_colors[..., 3:4]
            depth = depths[0].permute(2, 0, 1)
        else:
            colors = render_colors
            depth = None
        rendered_image = colors[0].permute(2, 0, 1)
        radii = info["radii"].squeeze(0)  # [N,]
        try:
            info["means2d"].retain_grad()  # [1, N, 2]
        except:
            pass
        return {"render": rendered_image,
                "depth": depth,
                "visibility_filter": radii > 0,
                "radii": radii}
