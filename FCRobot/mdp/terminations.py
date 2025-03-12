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

    single_cam_data = convert_dict_to_backend(
                {k: v[0] for k, v in asset.data.output.items()}, backend="numpy")

    img_gray = cv2.cvtColor(single_cam_data['rgb'], cv2.COLOR_BGR2GRAY)
    thr, img_th = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
    white_pix = cv2.countNonZero(img_th)
    coverage = ((100*100 - white_pix)/(100*100)) * 100
    
    return coverage > 90.0