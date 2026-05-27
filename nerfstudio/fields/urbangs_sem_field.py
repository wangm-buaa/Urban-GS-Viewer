from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import math
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions

from nerfstudio.utils.gaussian_splatting_general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, \
    strip_symmetric, build_scaling_rotation

from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
import os
from einops import repeat


class GaussianSplattingField(Field):
    def __init__(self, **model_kwargs):
        super().__init__()

        for key, value in model_kwargs.items():
            setattr(self, key, value)

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._level = torch.empty(0)
        self._extra_level = torch.empty(0)

        self.offset_opacity_accum = torch.empty(0)
        self.anchor_opacity_accum = torch.empty(0)
        self.anchor_demon = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.optimizer = None
        self.spatial_lr_scale = 0
        self.padding = 0.0
        self.ape_code = -1
        self.dist2level = 'floor'
        self.setup_functions()

        # self.n_offsets = 10
        self.active_sh_degree = None
        self.max_sh_degree = None
        self.color_dim = 3

        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.feat_dim + self.view_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(self.feat_dim + self.view_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7 * self.n_offsets),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(self.feat_dim + self.view_dim + self.appearance_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.color_dim * self.n_offsets),
        ).cuda()

        self.setup_functions()

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def load_mlp_checkpoints(self, path):
        self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
        self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
        self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
        if self.appearance_dim > 0:
            self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        else:
            self.embedding_appearance = None

    def load_ply(self, path):
        plydata = PlyData.read(path)
        infos = plydata.obj_info
        for info in infos:
            var_name = info.split(' ')[0]
            self.__dict__[var_name] = float(info.split(' ')[1])

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                           np.asarray(plydata.elements[0]["y"]),
                           np.asarray(plydata.elements[0]["z"])), axis=1).astype(np.float32)

        levels = np.asarray(plydata.elements[0]["level"])[..., np.newaxis].astype(np.int16)
        extra_levels = np.asarray(plydata.elements[0]["extra_level"])[..., np.newaxis].astype(np.float32)
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key=lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key=lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        self._anchor_feat = nn.Parameter(
            torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._level = torch.tensor(levels, dtype=torch.int, device="cuda")
        self._extra_level = torch.tensor(extra_levels, dtype=torch.float, device="cuda").squeeze(dim=1)
        self._offset = nn.Parameter(
            torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        self.aerial_levels = round(self.aerial_levels)
        self.street_levels = round(self.street_levels)
        self.active_sh_degree = self.max_sh_degree

    def load_sem_weights(self, model_path):
        ins_feat = torch.load(os.path.join(model_path, "semantic_info.pth"))
        if ins_feat['semantic_feat_q'].shape[0] > 0:
            self._ins_feat = torch.from_numpy(ins_feat['semantic_feat_q']).cuda()
        else:
            self._ins_feat = torch.from_numpy(ins_feat['semantic_feat']).cuda()

    def get_ins_feat(self):
        ins_feat = self._ins_feat
        ins_feat = torch.nn.functional.normalize(ins_feat, dim=1)
        return ins_feat

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_anchor_feat(self):
        return self._anchor_feat

    @property
    def get_offset(self):
        return self._offset

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_anchor(self):
        return self._anchor

    def get_scaling_w_mask(self, mask):
        return self.scaling_activation(self._scaling[mask])

    def get_rotation_w_mask(self, mask):
        return self.rotation_activation(self._rotation[mask])
    # @property
    # def get_features(self):
    #     features_dc = self._features_dc
    #     features_rest = self._features_rest
    #     return torch.cat((features_dc, features_rest), dim=1)

    # @property
    # def get_opacity(self):
    #     return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    # @property
    # def get_view_mask_mlp(self):
    #     return self.mlp_mask

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    def map_to_int_level(self, pred_level, cur_level):
        if self.dist2level == 'floor':
            int_level = torch.floor(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level == 'round':
            int_level = torch.round(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level == 'ceil':
            int_level = torch.ceil(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level == 'progressive':
            pred_level = torch.clamp(pred_level + 1.0, min=0.9999, max=cur_level + 0.9999)
            int_level = torch.floor(pred_level).int()
            self._prog_ratio = torch.frac(pred_level).unsqueeze(dim=1)
            self.transition_mask = (self._level.squeeze(dim=1) == int_level)
        else:
            raise ValueError(f"Unknown dist2level: {self.dist2level}")

        return int_level

    def set_anchor_mask(self, cam_center, mask=None):
        dist = torch.sqrt(torch.sum((self.get_anchor - cam_center)**2, dim=1))
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        int_level = self.map_to_int_level(pred_level, self.street_levels - 1)
        if mask is None:
            self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)
        else:
            self._anchor_mask = (self._level.squeeze(dim=1) <= int_level) & mask

    def smooth_complement(self, visible_mask):
        return torch.ones((visible_mask.sum(), 1), dtype=torch.float, device="cuda")

    def generate_neural_gaussians(self, viewpoint_camera, visible_mask=None, cur_visible_anchor=None, origin=False):
        ## view frustum filtering for acceleration
        if visible_mask is None:
            visible_mask = torch.ones(self.get_anchor.shape[0], dtype=torch.bool, device=self.get_anchor.device)

        if cur_visible_anchor is not None:
            anchor = self.get_anchor[cur_visible_anchor][visible_mask]
            feat = self.get_anchor_feat[cur_visible_anchor][visible_mask]
            grid_offsets = self.get_offset[cur_visible_anchor][visible_mask]
            grid_scaling = self.get_scaling[cur_visible_anchor][visible_mask]
            ins_feat = (self.get_ins_feat()[cur_visible_anchor][visible_mask] + 1) / 2
        else:
            anchor = self.get_anchor[visible_mask]
            feat = self.get_anchor_feat[visible_mask]
            grid_offsets = self.get_offset[visible_mask]
            grid_scaling = self.get_scaling_w_mask(visible_mask)
            ins_feat = (self.get_ins_feat()[visible_mask] + 1) / 2

        ## get view properties for anchor
        ob_view = anchor - viewpoint_camera.camera_center
        # dist
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        # view
        ob_view = ob_view / ob_dist

        if self.view_dim > 0:
            cat_local_view = torch.cat([feat, ob_view], dim=1)  # [N, c+3]
        else:
            cat_local_view = feat  # [N, c]

        if self.appearance_dim > 0:
            if self.ape_code < 0:
                camera_indicies = torch.ones_like(cat_local_view[:, 0], dtype=torch.long,
                                                  device=ob_dist.device) * viewpoint_camera.uid
                appearance = self.get_appearance(camera_indicies)
            else:
                camera_indicies = torch.ones_like(cat_local_view[:, 0], dtype=torch.long,
                                                  device=ob_dist.device) * self.ape_code
                appearance = self.get_appearance(camera_indicies)

        # get offset's opacity
        neural_opacity = self.get_opacity_mlp(cat_local_view) * self.smooth_complement(visible_mask)

        # opacity mask generation
        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity > 0.0)
        mask = mask.view(-1)

        # select opacity
        opacity = neural_opacity[mask]

        # get offset's color
        if self.appearance_dim > 0:
            color = self.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = self.get_color_mlp(cat_local_view)

        color = color.reshape([anchor.shape[0] * self.n_offsets, self.color_dim])  # [mask]

        # get offset's cov
        scale_rot = self.get_cov_mlp(cat_local_view)
        scale_rot = scale_rot.reshape([anchor.shape[0] * self.n_offsets, 7])  # [mask]

        # offsets
        offsets = grid_offsets.view([-1, 3])  # [mask]

        n_anchors = anchor.shape[0]
        anchor_indices = torch.arange(n_anchors, device=anchor.device)
        anchor_indices_expanded = anchor_indices[:, None].expand(-1, self.n_offsets).reshape(-1)
        masked_anchor_indices = anchor_indices_expanded[mask]
        scaling_repeat = grid_scaling[masked_anchor_indices]
        repeat_anchor = anchor[masked_anchor_indices]
        color = color[mask]
        scale_rot = scale_rot[mask]
        offsets = offsets[mask]  # offsets已经是对应每个offset的
        masked_language_feature = ins_feat[masked_anchor_indices]

        # post-process cov
        scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])  # * (1+torch.sigmoid(repeat_dist))
        rot = self.rotation_activation(scale_rot[:, 3:7])

        # post-process offsets to get centers for gaussians
        # offsets = torch.tanh(offsets) * scaling_repeat[:, :3].detach()
        # offsets = torch.tanh(offsets) * scaling_repeat[:,:3]
        offsets = offsets * scaling_repeat[:, :3]
        xyz = repeat_anchor + offsets

        # if self.color_attr != "RGB":
        #     color = color.reshape([color.shape[0], self.color_dim // 3, 3])

        return xyz, offsets, color, opacity, scaling, rot, self.active_sh_degree, mask, masked_language_feature, masked_anchor_indices
