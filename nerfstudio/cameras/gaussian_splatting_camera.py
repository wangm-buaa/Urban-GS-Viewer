#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from nerfstudio.utils.gaussian_splatting_graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(
            self,
            R,
            T,
            width,
            height,
            FoVx,
            FoVy,
            fx,
            fy,
            cx,
            cy,
            trans=np.array([0.0, 0.0, 0.0]),
            scale=1.0,
            data_device="cuda",
    ):
        super(Camera, self).__init__()

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        self.data_device = data_device

        self.image_width = width
        self.image_height = height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        # self.fx = self.image_width / (2 * np.tan(self.FoVx * 0.5))
        # self.fy = self.image_height / (2 * np.tan(self.FoVy * 0.5))

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
