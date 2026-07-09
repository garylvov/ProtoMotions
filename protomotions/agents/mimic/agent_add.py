# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import torch

from protomotions.agents.amp.agent import AMP
from protomotions.envs.obs.humanoid import compute_humanoid_max_coords_observations
from protomotions.utils.rotations import calc_heading
from lightning.fabric import Fabric
from typing import Optional
from pathlib import Path
from protomotions.envs.base_env.env import BaseEnv

log = logging.getLogger(__name__)

# Number of extra differential features appended to ``mimic_target_poses_diff``
# when ``add_root_displacement_features`` is enabled: root xy error in the
# current heading frame (2) + wrapped heading error (1).
NUM_ROOT_DISPLACEMENT_FEATURES = 3


def compute_root_displacement_features(
    ref_root_pos: torch.Tensor,
    ref_root_rot: torch.Tensor,
    current_root_pos: torch.Tensor,
    current_root_rot: torch.Tensor,
    w_last: bool = True,
) -> torch.Tensor:
    """Differential root-displacement features for the ADD discriminator.

    Track D teacher-retrain objective 1 (minimize xy + heading displacement
    relative to the motion): explicit error channels so the ADD discriminator
    can auto-balance global drift toward zero.  Like the existing
    ``mimic_target_poses_diff`` (ref pose minus current pose), these are
    reference-minus-actual differentials that are exactly zero when tracking
    is perfect — matching ADD's zero-vector positive samples.

    Args:
        ref_root_pos: Reference root positions [num_envs, 3].
        ref_root_rot: Reference root rotations [num_envs, 4].
        current_root_pos: Current root positions [num_envs, 3].
        current_root_rot: Current root rotations [num_envs, 4].
        w_last: Quaternion layout (xyzw when True).

    Returns:
        [num_envs, 3] tensor: xy error expressed in the current root heading
        frame (2) followed by the wrapped heading error in radians (1).
    """
    xy_err_world = ref_root_pos[:, :2] - current_root_pos[:, :2]
    cur_heading = calc_heading(current_root_rot, w_last)
    cos_h = torch.cos(cur_heading)
    sin_h = torch.sin(cur_heading)
    # Rotate the world-frame error by -heading into the heading-local frame.
    xy_err_local = torch.stack(
        [
            cos_h * xy_err_world[:, 0] + sin_h * xy_err_world[:, 1],
            -sin_h * xy_err_world[:, 0] + cos_h * xy_err_world[:, 1],
        ],
        dim=-1,
    )
    ref_heading = calc_heading(ref_root_rot, w_last)
    heading_err = ref_heading - cur_heading
    # Wrap to [-pi, pi).
    heading_err = torch.remainder(heading_err + torch.pi, 2 * torch.pi) - torch.pi
    return torch.cat([xy_err_local, heading_err.unsqueeze(-1)], dim=-1)


class MimicADD(AMP):
    def __init__(
        self, fabric: Fabric, env: BaseEnv, config, root_dir: Optional[Path] = None
    ):
        super().__init__(fabric, env, config, root_dir)

    @property
    def _root_displacement_features_enabled(self) -> bool:
        """Config-gated Track D extension, default OFF (no behavior change)."""
        return bool(
            getattr(getattr(self, "config", None), "add_root_displacement_features", False)
        )

    # -----------------------------
    # Environment Interaction and Data Updates
    # -----------------------------
    def add_agent_info_to_obs(self, obs):
        obs = super().add_agent_info_to_obs(obs)

        motion_times = self.env.motion_manager.motion_times
        motion_ids = self.env.motion_manager.motion_ids
        ref_state = self.env.motion_lib.get_motion_state(motion_ids, motion_times)

        ref_state_gt = ref_state.rigid_body_pos.reshape(self.num_envs, -1, 3)
        ref_state_gt += (
            self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
                ref_state_gt
            )
        )
        ref_ground_heights = self.env.terrain.get_ground_heights(
            ref_state_gt[:, 0]
        ).clone()

        current_state = self.env.simulator.get_bodies_state()
        ground_heights = self.env.terrain.get_ground_heights(
            current_state.rigid_body_pos[:, 0]
        ).clone()

        # ADD uses local_obs=False for tracking diff observations
        local_obs = False
        root_height_obs = True
        observe_contacts = False  # ADD does not yet support contact based conditioning

        # Empty contact flags since observe_contacts is False
        empty_contacts = torch.zeros(
            self.num_envs, 0, dtype=torch.bool, device=ref_state_gt.device
        )

        ref_pose = compute_humanoid_max_coords_observations(
            body_pos=ref_state_gt,
            body_rot=ref_state.rigid_body_rot,
            body_vel=ref_state.rigid_body_vel,
            body_ang_vel=ref_state.rigid_body_ang_vel,
            ground_height=ref_ground_heights,
            body_contacts=empty_contacts,
            local_obs=local_obs,
            root_height_obs=root_height_obs,
            observe_contacts=observe_contacts,
            w_last=True,
        )

        current_pose = compute_humanoid_max_coords_observations(
            body_pos=current_state.rigid_body_pos,
            body_rot=current_state.rigid_body_rot,
            body_vel=current_state.rigid_body_vel,
            body_ang_vel=current_state.rigid_body_ang_vel,
            ground_height=ground_heights,
            body_contacts=empty_contacts,
            local_obs=local_obs,
            root_height_obs=root_height_obs,
            observe_contacts=observe_contacts,
            w_last=True,
        )

        tracking_diff_obs = ref_pose - current_pose
        tracking_diff_obs = tracking_diff_obs.view(self.num_envs, -1)

        if self._root_displacement_features_enabled:
            # Track D: append root xy displacement error (heading frame) and
            # wrapped heading error as extra differential channels.
            root_displacement_features = compute_root_displacement_features(
                ref_root_pos=ref_state_gt[:, 0],
                ref_root_rot=ref_state.rigid_body_rot.reshape(self.num_envs, -1, 4)[:, 0],
                current_root_pos=current_state.rigid_body_pos[:, 0],
                current_root_rot=current_state.rigid_body_rot[:, 0],
                w_last=True,
            )
            tracking_diff_obs = torch.cat(
                [tracking_diff_obs, root_displacement_features], dim=-1
            )

        obs["mimic_target_poses_diff"] = tracking_diff_obs
        # Cache the differential width so expert (zero) samples can be built
        # without re-deriving it from env internals (see get_expert_disc_obs).
        self._tracking_diff_dim = int(tracking_diff_obs.shape[-1])
        return obs

    def get_expert_disc_obs(self, num_samples: int):
        if getattr(getattr(self, "config", None), "reference_obs_components", True):
            expert_disc_obs = super().get_expert_disc_obs(num_samples)
        else:
            # ADD positive samples are pure zero differentials: when the
            # discriminator consumes only ``mimic_target_poses_diff`` there is
            # nothing to compute from the motion lib, so the base-AMP
            # requirement for reference_obs_components does not apply.
            expert_disc_obs = {}
        cached = getattr(self, "_tracking_diff_dim", None)
        if cached is not None:
            # Set by add_agent_info_to_obs during collection (always runs
            # before dataset augmentation) — already includes the appended
            # root-displacement channels when enabled.
            obs_dim = cached
        else:
            obs_manager = getattr(self.env, "observation_manager", None)
            hist = getattr(obs_manager, "observation_history_buffers", {}) if obs_manager else {}
            if "max_coords_obs" in hist:
                obs_dim = hist["max_coords_obs"].data.shape[-1]
            else:
                obs_dim = expert_disc_obs.get("max_coords_obs", expert_disc_obs.get("historical_max_coords_obs", torch.empty(0))).shape[-1] // 8
            if self._root_displacement_features_enabled:
                # Expert (positive) samples are zero differentials — extend by
                # the appended root-displacement channels to stay aligned.
                obs_dim += NUM_ROOT_DISPLACEMENT_FEATURES
        tracking_diff_obs = torch.zeros(
            [num_samples, obs_dim],
            device=self.device,
        )
        expert_disc_obs["mimic_target_poses_diff"] = tracking_diff_obs

        return expert_disc_obs
