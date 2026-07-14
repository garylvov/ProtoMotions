# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
from protomotions.envs.motion_manager.motion_manager import MotionManager
from protomotions.components.motion_lib import MotionLib
import torch
from typing import Optional


class MimicMotionManager(MotionManager):
    """Motion manager specialized for mimic environments.

    Extends the base MotionManager to handle mimic-specific motion sampling,
    including time progression and conditional resampling on reset.
    """

    config: MimicMotionManagerConfig

    def __init__(
        self,
        config: MimicMotionManagerConfig,
        num_envs: int,
        env_dt: float,
        device: torch.device,
        motion_lib: MotionLib,
        fixed_motion_ids_per_env: Optional[torch.Tensor] = None,
    ):
        """A motion manager that handles motion sampling and tracking for mimic environments.

        Args:
            config: Configuration object containing motion manager settings
            num_envs (int): Number of parallel environments
            env_dt (float): Environment timestep
            device (torch.device): Device to store tensors on
            motion_lib (MotionLib): Motion library containing reference motions
            fixed_motion_ids_per_env (Optional[torch.Tensor], optional): If provided, specifies fixed motion IDs to use for each environment. Defaults to None.
        """
        super().__init__(config, num_envs, env_dt, device, motion_lib, fixed_motion_ids_per_env)

    def get_done_tracks(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Check which motion tracks have reached their end time.

        Args:
            env_ids: Optional tensor of environment indices to check. If None, checks all environments.

        Returns:
            Boolean tensor indicating which motion tracks are done (True) or still playing (False)
        """
        end_times = self.motion_lib.motion_lengths[self.motion_ids]
        done_clip = (self.motion_times + self.env_dt) >= end_times
        if env_ids is not None:
            done_clip = done_clip[env_ids]
        return done_clip

    def post_physics_step(self):
        """Advance motion playback time by one environment timestep.

        Called after each physics simulation step to update the current time
        in each motion track. With the reference-freeze augmentation enabled
        (reference_freeze_prob_per_sec > 0), each env's reference clock
        randomly holds for a sampled duration and then resumes; everything
        downstream (mimic targets, masked-mimic conditioning, expert obs,
        tracking rewards, termination) consistently sees the frozen
        reference. Default (prob 0.0) is byte-identical stock behavior.
        """
        freeze_prob = getattr(self.config, "reference_freeze_prob_per_sec", 0.0)
        if freeze_prob <= 0.0:
            self.motion_times += self.env_dt
            return

        if not hasattr(self, "_freeze_time_left"):
            self._freeze_time_left = torch.zeros(self.num_envs, device=self.device)
        frozen = self._freeze_time_left > 0.0
        # start new freezes on currently-unfrozen envs
        lo, hi = self.config.reference_freeze_duration_range
        start = (~frozen) & (
            torch.rand(self.num_envs, device=self.device)
            < freeze_prob * self.env_dt
        )
        if start.any():
            n = int(start.sum())
            self._freeze_time_left[start] = lo + (hi - lo) * torch.rand(
                n, device=self.device
            )
            frozen = frozen | start
        # advance unfrozen reference clocks; tick down active freezes
        self.motion_times += self.env_dt * (~frozen).float()
        self._freeze_time_left.sub_(self.env_dt).clamp_(min=0.0)

    def sample_motions(
        self, env_ids: torch.Tensor, new_motion_ids: Optional[torch.Tensor] = None
    ):
        """Clear any active reference freeze for envs being reset."""
        if hasattr(self, "_freeze_time_left") and len(env_ids) > 0:
            self._freeze_time_left[env_ids] = 0.0
        return self._sample_motions_mimic(env_ids, new_motion_ids)

    def _sample_motions_mimic(
        self, env_ids: torch.Tensor, new_motion_ids: Optional[torch.Tensor] = None
    ):
        """Sample new motions for environments.

        Extends base class to handle mimic-specific resample_on_reset logic:
        only resample motions that have finished playing.

        Args:
            env_ids (Tensor): Indices of the environments to reset.
            new_motion_ids (Tensor, optional):
                Force new motion IDs for the reset environments.
                If provided, this overrides fixed motion IDs.
        """
        # Mimic-specific: Only resample motions that have finished if resample_on_reset is False
        reset_env_ids = env_ids
        if not self.config.resample_on_reset:
            done_tracks = self.get_done_tracks(env_ids)
            reset_env_ids = env_ids[done_tracks]

        # Only proceed if there are environments to reset
        if len(reset_env_ids) == 0:
            return

        # Call parent sample_motions (handles fixed motion IDs)
        super().sample_motions(reset_env_ids, new_motion_ids)
