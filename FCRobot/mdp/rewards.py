# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from math import sqrt
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import TiledCamera
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

import cv2
import numpy as np
from numpy import inf
from math import sqrt
from isaaclab.utils import convert_dict_to_backend

from . import observations as obs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    
    asset : RigidObject = env.scene[asset_cfg.name]
    
    heading_proj = obs.base_heading_proj(env, target_pos, asset).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)

 
    
    
def targetedCoverage(env: ManagerBasedRLEnv, heightThreshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    

    asset: Articulation = env.scene[asset_cfg.name]

    # env.render(recompute=True)
    
    single_cam_data = convert_dict_to_backend(
                {k: v[0] for k, v in asset.data.output.items()}, backend="numpy")
    
    depthImgData = single_cam_data["distance_to_image_plane"]
    depthImgData[depthImgData == inf]= 0
    
    # print(depthImgData[0], type(depthImgData))
    
    # print(np.max(depthImgData))
    
    # print(np.mean(depthImgData > 1.40)*100)
    if heightThreshold is None:
        heightThreshold = 1.5 # 1.5m from the camera down to the conveyor
    areaCovered = torch.tensor([np.mean(depthImgData < heightThreshold)],dtype=torch.float32,device=env.device)
    
    target = 0.3
    diff = abs(areaCovered - target)
    if diff == 0: return areaCovered * 30.0
    elif diff != 0: return diff**2 * -30.0
    

def AreaCovJointVelRel(env: ManagerBasedRLEnv, sensor1_cfg: SceneEntityCfg, sensor2_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    

    asset: Articulation = env.scene[asset_cfg.name]
    
    # Two adjacent Areas should have the same area of coverage
    sensor1: TiledCamera = env.scene.sensors[sensor1_cfg.name]
    sensor2: TiledCamera = env.scene.sensors[sensor2_cfg.name]

    
    single_cam_data_1 = convert_dict_to_backend(
                {k: v for k, v in sensor1.data.output.items()}, backend="numpy")
    
    single_cam_data_2 = convert_dict_to_backend(
                {k: v for k, v in sensor2.data.output.items()}, backend="numpy")
    
    depthImgData1 = single_cam_data_1["distance_to_image_plane"]
    depthImgData2 = single_cam_data_2["distance_to_image_plane"]
    
    depthImgData1[depthImgData1 == inf]= 0
    depthImgData2[depthImgData2 == inf]= 0
    
    heightThreshold = 1.5 # 1.5m from the camera down to the conveyor
    
    areaCovered1 = torch.tensor([np.mean(depthImgData1 < heightThreshold)],dtype=torch.float32,device=env.device)
    areaCovered2 = torch.tensor([np.mean(depthImgData2 < heightThreshold)],dtype=torch.float32,device=env.device)
    
    
    diff = abs(areaCovered1 - areaCovered2)*100
    
    
    #Inverse relationship between area coverage and jointVel speed
    currentAction = env.action_manager.action
    prevAction = env.action_manager.prev_action
    actionDiff = currentAction - prevAction
    chunks = torch.chunk(currentAction, chunks=8, dim=1)
    Spdmeans = torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1) / 1326
    # print(Spdmeans)
    # print(chunks)
    
    if sensor1_cfg.name == "tiled_camera1":
        relationshipReward = (Spdmeans[:,0] + areaCovered1 - 1)*10
    elif sensor1_cfg.name == "tiled_camera2":
        relationshipReward = (Spdmeans[:,1] + areaCovered1 - 1)*10
    elif sensor1_cfg.name == "tiled_camera3":
        relationshipReward = (Spdmeans[:,2] + areaCovered1 - 1)*10
    elif sensor1_cfg.name == "tiled_camera4":
        relationshipReward = (Spdmeans[:,3] + areaCovered1 - 1)*10
    elif sensor1_cfg.name == "tiled_camera5":
        relationshipReward = (Spdmeans[:,4] + areaCovered1 - 1)*10
    elif sensor1_cfg.name == "tiled_camera6":
        relationshipReward = (Spdmeans[:,5] + areaCovered1 - 1)*10
    elif sensor1_cfg.name == "tiled_camera7":
        relationshipReward = (Spdmeans[:,6] + areaCovered1 - 1)*10
    elif sensor1_cfg.name == "tiled_camera8":
        relationshipReward = (Spdmeans[:,7] + areaCovered1 - 1)*10
    
    relationshipReward = -relationshipReward**2
    
    #print(relationshipReward)
    
    if diff <= 1: return (areaCovered1*100)**2 * 30.0 + relationshipReward
    elif diff > 1: return diff**2 * -30.0 + relationshipReward


def joint_vel_positive(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # print(asset.data.joint_vel[:,:], asset.data.joint_vel[:,:].size())
    # print(asset_cfg.joint_names, asset_cfg.joint_ids)
    
    # ######### Default #############
    # jointVel = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    # ###############################
    
    # print(asset.data, type(asset.data))
    # print(asset.data.soft_joint_vel_limits, asset.data.soft_joint_vel_limits.size())
    # print(asset.data.joint_vel_limits) ArticulationData does not have this. Weird
    # for k in jointVel: print(k)
    # print(asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids])
    #print(((jointVel < 1).sum().item())/(int(env.num_envs)*128.0))
    
    
    currentAction = env.action_manager.action
    prevAction = env.action_manager.prev_action
    actionDiff = currentAction - prevAction
    baseRew = torch.where(actionDiff > 0, actionDiff*1.0, actionDiff*10.0)
    
    negVelRew = torch.where(currentAction > 0, currentAction**2 + 10, ((currentAction**2) * -1.0) - 100)
    #print("SUM ", torch.sum(rew, dim=1))
    
    totalRew = baseRew + negVelRew
    
    return torch.sum(totalRew, dim=1)
    
    
    
    # sumEnvJointVel = torch.sum(asset.data.joint_vel[:, asset_cfg.joint_ids],dim=1)
    # # sumEnvDefaultJointVel = torch.sum(asset.data.default_joint_vel[:, asset_cfg.joint_ids],dim=1)
    # sumEnvSoftJointVelLim = torch.sum(asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids], dim=1)
    
    # #outLimNeg = torch.sub(sumEnvJointVel, sumEnvDefaultJointVel,alpha=1.0)
    # ratio = 300
    # outLimRew = torch.tanh(torch.div((sumEnvJointVel - sumEnvSoftJointVelLim * 0.5), ratio))
    
    # print("Joint Vel ",sumEnvJointVel)
    # print("Default Joint Vel ",sumEnvDefaultJointVel)
    # print("Soft Joint Vel Lim ",sumEnvSoftJointVelLim)
    
    #print(outLimRew)
    
    #return outLimRew*100.0


    
    # JointVelNegCount = (jointVel < 20).sum().item()
    # JointVelPosCount = (jointVel >= 20).sum().item()
    
    # if JointVelPosCount == None: JointVelPosCount = 0
    # if JointVelNegCount == None: JointVelNegCount = 0

    # if JointVelNegCount < JointVelPosCount:
    #     return torch.tensor((15.0 + ((JointVelPosCount - JointVelNegCount)/env.num_envs))**2, dtype=torch.float32, device=env.device)
    # elif JointVelNegCount == 0 and JointVelPosCount == 0:
    #     return torch.tensor((-5.0*10), dtype=torch.float32, device=env.device)
    # elif JointVelNegCount >= JointVelPosCount or JointVelNegCount > 0:
    #     return torch.tensor(-1.0*(-15.0 + ((JointVelPosCount - JointVelNegCount)/env.num_envs))**2, dtype=torch.float32, device=env.device)
    
    
    # generalJointVel = ((jointVel < 5).sum().item())/(int(env.num_envs)*128.0)
    # # print(generalJointVel)
    # return torch.tensor(generalJointVel, dtype=torch.float32, device=env.device)
    
    
    
    
    