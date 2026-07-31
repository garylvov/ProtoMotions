# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mimic control component for motion tracking tasks.

This component manages reference motion tracking, including:
- Motion manager for motion library sampling and playback
- Reference state computation and terrain correction
- Masked mimic conditioning state
- Visualization markers for target poses
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext, MimicContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.envs.obs.humanoid import dof_to_local
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarkerConfig,
    MarkerState,
)

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class MimicControlConfig(ControlComponentConfig):
    """Configuration for mimic control component.
    
    Attributes:
        bootstrap_on_episode_end: If True, don't terminate when motion clip ends.
        future_steps: Future reference poses to provide in context. If int N,
            provides N consecutive steps (1 to N). If list, provides specific step
            indices (e.g., [1, 3, 5, 9, 15] for non-uniform sampling).
    """
    _target_: str = "protomotions.envs.control.mimic_control.MimicControl"
    
    bootstrap_on_episode_end: bool = True
    future_steps: Union[int, List[int]] = 1


class MimicControl(ControlComponent):
    """Control component for motion tracking tasks.
    
    Provides context for mimic observations and rewards. Accesses the env's
    motion_manager rather than creating its own.
    """
    
    config: MimicControlConfig
    
    def __init__(self, config: MimicControlConfig, env: "BaseEnv"):
        """Initialize mimic control component.
        
        Args:
            config: Component configuration.
            env: Parent environment instance.
        """
        super().__init__(config, env)
    
    def step(self):
        """Control component step - motion manager is handled by env."""
        # Motion manager is updated by env.post_physics_step if needed
        pass
    
    def check_resets_and_terminations(self) -> Tuple[Tensor, Tensor]:
        """Check if motion clips have finished.
        
        Returns:
            Tuple of (reset_buf, terminate_buf) boolean tensors.
        """
        device = self.env.device
        num_envs = self.env.num_envs
        
        # Check if motion clip has finished (access via env)
        done_clip = self.env.motion_manager.get_done_tracks()
        reset_buf = done_clip
        
        # Only terminate if not bootstrapping
        if self.config.bootstrap_on_episode_end:
            terminate_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
        else:
            terminate_buf = done_clip
        
        return reset_buf, terminate_buf
    
    def populate_context(self, ctx: EnvContext) -> None:
        """Populate mimic-specific view in the EnvContext.
        
        Creates a MimicContext with:
        - ref_state: Single-step reference state at current time (for rewards)
        - future_*: Multi-step future reference poses [envs, future_steps, ...]
        
        Args:
            ctx: The EnvContext to populate with ctx.mimic.
        """
        num_envs = self.env.num_envs
        device = self.env.device
        motion_ids = self.env.motion_manager.motion_ids
        motion_times = self.env.motion_manager.motion_times

        # Online mirror augmentation: per-env flags (None when PM_MIRROR_PROB=0
        # -> byte-identical fast path, no mirror work). The SAME per-env flags key
        # both the current ref (rewards) and the future window (obs), so a mirrored
        # env sees a fully consistent mirrored reference this step.
        _mm = self.env.motion_manager
        mirror_flags = _mm.mirror_flags if _mm.mirror_prob > 0.0 else None
        # Same pattern for the per-episode time-reversal coin (None when
        # PM_REVERSE_PROB unset -> byte-identical fast path). The SAME per-env
        # flags key both the current ref (rewards) and the future window (obs),
        # so a reversed env sees a fully consistent reversed timeline this step.
        reverse_flags = (
            _mm.reverse_flags if getattr(_mm, "reverse_prob", 0.0) > 0.0 else None
        )

        # Get single-step reference state at current time (for rewards)
        ref_state = self.env.motion_lib.get_motion_state(
            motion_ids,
            motion_times,
            mirror_flags=mirror_flags,
            reverse_flags=reverse_flags,
        )
        
        # Apply terrain height correction to reference state
        ref_gt = ref_state.rigid_body_pos.clone()
        ref_gt += self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            ref_gt
        )
        ref_state.rigid_body_pos = ref_gt

        # --- Reference-conditioned payload modulation (anti-cheat) -------------
        # Push the per-wrist force scale into the simulator from the REFERENCE
        # motion's (terrain-corrected, world-frame) wrist->chest distance. This
        # is a NO-OP unless a wrench class sets reference_distance_modulation=True
        # (the simulator setter early-returns). The scale depends ONLY on the
        # exogenous demonstration (see simulator.reference_wrench_scale), so the
        # policy cannot shed load by moving; it takes effect on the next physics
        # step (the payload is quasi-static, so the one-step lag is negligible).
        # The optional secondary hardening (posture_gate) additionally consumes
        # the live robot pose passed as actual_body_pos.
        _set_ref_scale = getattr(
            self.env.simulator, "set_wrench_reference_scale_from_reference", None
        )
        if _set_ref_scale is not None:
            _body_names = self.env.robot_config.kinematic_info.body_names
            _actual_body_pos = (
                ctx.current.rigid_body_pos if getattr(ctx, "current", None) is not None
                else None
            )
            _set_ref_scale(ref_state.rigid_body_pos, _body_names, _actual_body_pos)

        # Build multi-step reference for observations
        dt = self.env.dt
        if isinstance(self.config.future_steps, int):
            step_indices = list(range(1, self.config.future_steps + 1))
        else:
            step_indices = self.config.future_steps
        future_steps = len(step_indices)
        
        time_offsets = dt * torch.tensor(step_indices, device=device, dtype=torch.float32)
        future_times = motion_times.unsqueeze(-1) + time_offsets  # [envs, N]
        
        motion_lengths = self.env.motion_lib.get_motion_length(motion_ids)
        future_times = torch.minimum(future_times, motion_lengths.unsqueeze(-1))
        
        flat_motion_ids = motion_ids.unsqueeze(-1).expand(
            num_envs, future_steps
        ).reshape(-1)
        flat_future_times = future_times.reshape(-1)

        # Same per-env flags, broadcast across the future window (envs-major, to
        # match flat_motion_ids' [num_envs, future_steps] flattening).
        flat_mirror_flags = (
            mirror_flags.unsqueeze(-1).expand(num_envs, future_steps).reshape(-1)
            if mirror_flags is not None
            else None
        )
        # Same per-env reverse flags across the future window: the fetch-level
        # time remap turns playback time t + k*dt into original time
        # L - t - k*dt, so the future obs correctly walk the original clip
        # BACKWARDS for reversed envs.
        flat_reverse_flags = (
            reverse_flags.unsqueeze(-1).expand(num_envs, future_steps).reshape(-1)
            if reverse_flags is not None
            else None
        )

        # Query motion lib for all future steps
        future_state = self.env.motion_lib.get_motion_state(
            flat_motion_ids,
            flat_future_times,
            mirror_flags=flat_mirror_flags,
            reverse_flags=flat_reverse_flags,
        )
        
        # Reshape to [envs, future_steps, ...] and apply terrain correction
        num_bodies = future_state.rigid_body_pos.shape[1]
        num_dofs = future_state.dof_pos.shape[1]
        
        # Body positions with terrain correction
        future_pos = future_state.rigid_body_pos.view(
            num_envs, future_steps, num_bodies, 3
        ).clone()
        offset = self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            future_pos[:, 0, :, :]  # Use first step for offset
        )
        future_pos += offset.unsqueeze(1)
        
        # Body rotations
        future_rot = future_state.rigid_body_rot.view(
            num_envs, future_steps, num_bodies, 4
        )
        
        # Body velocities
        future_vel = future_state.rigid_body_vel.view(
            num_envs, future_steps, num_bodies, 3
        )
        
        # Body angular velocities
        future_ang_vel = future_state.rigid_body_ang_vel.view(
            num_envs, future_steps, num_bodies, 3
        )
        
        # DOF positions and velocities
        future_dof_pos = future_state.dof_pos.view(
            num_envs, future_steps, num_dofs
        )
        future_dof_vel = future_state.dof_vel.view(
            num_envs, future_steps, num_dofs
        )
        
        hinge_axes_map = self.env.robot_config.kinematic_info.hinge_axes_map
        ref_lr = dof_to_local(ref_state.dof_pos, hinge_axes_map, True)
        
        # Populate the mimic view
        ctx.mimic = MimicContext(
            ref_state=ref_state,
            future_pos=future_pos,
            future_rot=future_rot,
            future_vel=future_vel,
            future_ang_vel=future_ang_vel,
            future_dof_pos=future_dof_pos,
            future_dof_vel=future_dof_vel,
            anchor_idx=self.env.robot_config.anchor_body_index,
            ref_lr=ref_lr,
        )
    
    def create_visualization_markers(self, headless: bool) -> Dict[str, VisualizationMarkerConfig]:
        """Create visualization markers for reference poses.
        
        Args:
            headless: If True, returns empty dict.
            
        Returns:
            Dictionary of marker configurations.
        """
        if headless:
            return {}
        
        visualization_markers = {}
        
        # Standard mimic: visualize all bodies
        body_names = self.env.robot_config.kinematic_info.body_names
        
        body_markers = []
        for body_name in body_names:
            if (
                self.env.robot_config.mimic_small_marker_bodies is not None
                and body_name in self.env.robot_config.mimic_small_marker_bodies
            ):
                body_markers.append(MarkerConfig(size="small"))
            else:
                body_markers.append(MarkerConfig(size="regular"))
        
        # Red markers for target poses
        body_markers_red_cfg = VisualizationMarkerConfig(
            type="sphere", color=(1.0, 0.0, 0.0), markers=body_markers
        )
        visualization_markers["body_markers_red"] = body_markers_red_cfg
        
        return visualization_markers
    
    def get_markers_state(self) -> Dict[str, MarkerState]:
        """Compute marker positions for reference poses.
        
        Returns:
            Dictionary mapping marker names to MarkerState.
        """
        if self.env.simulator.headless:
            return {}
        
        markers_state = {}
        
        # Get reference state at current time (access motion_manager via env).
        # Mirror the markers to match the (possibly mirrored) served reference.
        _mm = self.env.motion_manager
        _mflags = _mm.mirror_flags if _mm.mirror_prob > 0.0 else None
        _rflags = (
            _mm.reverse_flags if getattr(_mm, "reverse_prob", 0.0) > 0.0 else None
        )
        ref_state = self.env.motion_lib.get_motion_state(
            self.env.motion_manager.motion_ids,
            self.env.motion_manager.motion_times,
            mirror_flags=_mflags,
            reverse_flags=_rflags,
        )
        
        target_pos = ref_state.rigid_body_pos.clone()
        target_pos += (
            self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
                target_pos
            )
        )
        
        # Standard mimic: show all body markers in red
        target_pos = target_pos.view(self.env.num_envs, -1, 3)
        markers_state["body_markers_red"] = MarkerState(
            translation=target_pos,
            orientation=torch.zeros(
                self.env.num_envs, target_pos.shape[1], 4, device=self.env.device
            ),
        )
        
        return markers_state
