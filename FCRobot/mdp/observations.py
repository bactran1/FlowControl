# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

import cv2
from math import sqrt
import numpy as np
from numpy import inf
from isaaclab.utils import convert_dict_to_backend

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize joint position deviation from a target value."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     # wrap the joint positions to (-pi, pi)
#     joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
#     # compute the reward
#     return torch.sum(torch.square(joint_pos - target), dim=1)


def percentageArea_occupied(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    
    asset: Articulation = env.scene[asset_cfg.name]

    # env.render(recompute=True)
    
    # single_cam_data = convert_dict_to_backend(
    #             {k: v[0] for k, v in asset.data.output.items()}, backend="numpy")
    
    multi_cam_data = convert_dict_to_backend(
                {k: v for k, v in asset.data.output.items()}, backend="numpy")
    #print(multi_cam_data["distance_to_image_plane"], multi_cam_data["distance_to_image_plane"].size)
    
    depthImgData = multi_cam_data["distance_to_image_plane"]
    depthImgData = np.squeeze(depthImgData)
    depthImgData[depthImgData == inf]= 0
    # depthImgData = np.array_split(depthImgData, len(depthImgData) // 10000)
    # print(depthImgData.shape)
    
    # normalized_coverage = 1 - sqrt((coverage - target)**2)/100
    # print(img_gray)
    # print(depthImgData)
    # print(white_pix)
    # print(white_pix, type(white_pix), normalized_coverage, type(normalized_coverage), type(single_cam_data))
    # print(depthImgData[0], type(depthImgData))
    
    # print(np.max(depthImgData))
    
    # print(np.mean(depthImgData > 1.40)*100)
    heightThreshold = 1.5 # 1.5m from the camera down to the conveyor
    
    # print(depthImgData.shape[0])
    # print(env.num_envs, type(env.num_envs))
    areaCovered = torch.empty((depthImgData.shape[0],1),dtype=torch.float32,device=env.device)
    
    for i in range(depthImgData.shape[0]):
        miniAreaCovered = torch.tensor([np.mean(depthImgData[i,:,:] < heightThreshold)], dtype=torch.float32,device=env.device)
        areaCovered[i] = miniAreaCovered
    
    # areaCovered = torch.tensor(miniAreaCovered,dtype=torch.float32,device=env.device)
    #print(asset_cfg.name, areaCovered, areaCovered.size())
    
    
    #if asset_cfg.name == 'tiled_camera1':
    #    print(asset_cfg.name, areaCovered, areaCovered.size())
        
    return areaCovered