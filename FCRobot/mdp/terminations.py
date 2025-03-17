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


def coverMoreThan90(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize camera position deviation from a target value."""

    asset: Articulation = env.scene[asset_cfg.name]

    # env.render(recompute=True)
    
    multi_cam_data = convert_dict_to_backend(
                {k: v for k, v in asset.data.output.items()}, backend="numpy")
    #print(multi_cam_data["distance_to_image_plane"], multi_cam_data["distance_to_image_plane"].size)
    
    depthImgData = multi_cam_data["distance_to_image_plane"]
    depthImgData = np.squeeze(depthImgData)
    depthImgData[depthImgData == inf]= 0
    
    # print(np.mean(depthImgData > 1.40)*100)
    
    # def count_less_than_x(arr, x):

    #     return np.sum(arr < x)
    
    heightThreshold = 1.5 # 1.5m from the camera down to the conveyor
    
    # print(count_less_than_x(depthImgData, heightThreshold), np.mean(depthImgData < heightThreshold)*100)
    
    areaCovered = torch.empty((depthImgData.shape[0],1),dtype=torch.float32,device=env.device)
    
    for i in range(depthImgData.shape[0]):
        miniAreaCovered = torch.tensor([np.mean(depthImgData[i,:,:] < heightThreshold)], dtype=torch.float32,device=env.device)
        areaCovered[i] = miniAreaCovered
    
        
    return areaCovered > 0.3

# def joint_vel_positive(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
#     """The joint velocities of the asset w.r.t. the default joint velocities.

#     Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     # print(asset.data.joint_vel[:,:], asset.data.joint_vel[:,:].size())
#     # print(asset_cfg.joint_names, asset_cfg.joint_ids)
#     jointVel = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
#     # for k in jointVel: print(k)
#     # print(asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids])
#     return (jointVel < 1).any().item() | (jointVel > 10).any().item()