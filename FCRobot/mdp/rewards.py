# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from math import sqrt
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

import cv2
import numpy as np
from numpy import inf
from math import sqrt
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


def targetedCoverage(env: ManagerBasedRLEnv, heightThreshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    

    asset: Articulation = env.scene[asset_cfg.name]

    # env.render(recompute=True)
    
    single_cam_data = convert_dict_to_backend(
                {k: v[0] for k, v in asset.data.output.items()}, backend="numpy")

    # img_gray = cv2.cvtColor(single_cam_data['rgb'], cv2.COLOR_BGR2GRAY)
    # thr, img_th = cv2.threshold(img_gray, 160, 255, cv2.THRESH_BINARY)
    # white_pix = cv2.countNonZero(img_th)
    # coverage = ((100*100 - white_pix)/(100*100))*100
    
    depthImgData = single_cam_data["distance_to_image_plane"]
    depthImgData[depthImgData == inf]= 0
    
    
    # normalized_coverage = 1 - sqrt((coverage - target)**2)/100
    # print(img_gray)
    # print(depthImgData)
    # print(white_pix)
    # print(white_pix, type(white_pix), normalized_coverage, type(normalized_coverage), type(single_cam_data))
    # print(depthImgData[0], type(depthImgData))
    
    # print(np.max(depthImgData))
    
    # print(np.mean(depthImgData > 1.40)*100)
    if heightThreshold is None:
        heightThreshold = 1.5 # 1.5m from the camera down to the conveyor
    areaCovered = torch.tensor([np.mean(depthImgData < heightThreshold)],dtype=torch.float32,device=env.device)
    
    target = 0.5
    diff = abs(areaCovered - target)
    if diff == 0: return areaCovered * 10.0
    elif diff != 0: return diff * -5.0

def joint_vel_positive(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # print(asset.data.joint_vel[:,:], asset.data.joint_vel[:,:].size())
    # print(asset_cfg.joint_names, asset_cfg.joint_ids)
    jointVel = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    # for k in jointVel: print(k)
    # print(asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids])
    #print(((jointVel < 1).sum().item())/(int(env.num_envs)*128.0))
    JointVelNegCount = (jointVel < 7).sum().item()
    JointVelPosCount = (jointVel >= 7).sum().item()
    
    if JointVelPosCount == None: JointVelPosCount = 0
    if JointVelNegCount == None: JointVelNegCount = 0
    
    print(JointVelPosCount, JointVelNegCount)

    if JointVelNegCount < JointVelPosCount:
        return torch.tensor((5.0 + ((JointVelPosCount - JointVelNegCount)/128.0))*2, dtype=torch.float32, device=env.device)
    elif JointVelNegCount == 0 and JointVelPosCount == 0:
        return torch.tensor((-5.0*10), dtype=torch.float32, device=env.device)
    elif JointVelNegCount >= JointVelPosCount or JointVelNegCount > 0:
        return torch.tensor(-1.0*(-5.0 + ((JointVelPosCount - JointVelNegCount)/128.0))**2, dtype=torch.float32, device=env.device)
    # generalJointVel = ((jointVel < 5).sum().item())/(int(env.num_envs)*128.0)
    # # print(generalJointVel)
    # return torch.tensor(generalJointVel, dtype=torch.float32, device=env.device)
    
    
    
    
    