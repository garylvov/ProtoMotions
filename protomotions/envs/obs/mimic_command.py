# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observation compute kernel for heading-frame future-displacement commands.

Pure tensor function (kernel) for computing a heading-local future-displacement
command observation: for each selected future step, the reference anchor's
displacement from the current anchor, expressed in the current anchor's
heading-aligned frame (XYZ, Z preserved), plus an optional 2D heading(yaw)
delta encoding.

Use MdpComponent in experiment configs to bind kernel to context paths:

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.mimic_command import build_mimic_future_displacement_cmd

    observation_components = {
        "mimic_future_displacement_cmd": MdpComponent(
            compute_func=build_mimic_future_displacement_cmd,
            dynamic_vars={
                "current_state_anchor_pos": EnvContext.current.anchor_pos,
                "current_state_anchor_rot": EnvContext.current.anchor_rot,
                "mimic_ref_anchor_pos": EnvContext.mimic.future_anchor_pos,
                "mimic_ref_anchor_rot": EnvContext.mimic.future_anchor_rot,
            },
            static_params={"future_steps": [8], "include_heading_delta": True, "w_last": True},
        ),
    }
"""

from typing import List, Union

import torch
from torch import Tensor

from protomotions.utils import rotations
from protomotions.envs.obs.utils import heading_local_xyz_delta, select_step_indices


def build_mimic_future_displacement_cmd(
    current_state_anchor_pos: Tensor,
    current_state_anchor_rot: Tensor,
    mimic_ref_anchor_pos: Tensor,
    mimic_ref_anchor_rot: Tensor,
    future_steps: Union[int, List[int]] = 1,
    w_last: bool = True,
    include_heading_delta: bool = True,
) -> Tensor:
    """Build heading-local future-displacement command observation.

    For each selected future step, computes the reference anchor position's
    displacement from the current anchor position, rotated into the current
    anchor's heading-aligned frame (XYZ, with Z preserved as a raw delta so
    vertical motion such as deep-bend/pickup remains observable). Optionally
    appends a 2D (cos, sin) encoding of the relative heading(yaw) delta
    between the current and reference anchor orientations, following the
    ``quat_to_tan_norm`` pattern used by ``build_target_root_rot``: for a
    pure-yaw (heading-only) quaternion, ``quat_to_tan_norm`` reduces to a
    constant Z-normal, so only the first 2 (tangent XY / cos,sin) components
    are informative and are kept.

    This is a pure delta computation (ref future - current, heading-rotated):
    no world-frame absolute values are emitted and no state is accumulated
    across steps, making it safe for deployment.

    Args:
        current_state_anchor_pos: Current anchor position [envs, 3].
        current_state_anchor_rot: Current anchor rotation [envs, 4] (w-last by default).
        mimic_ref_anchor_pos: Reference anchor positions [envs, total_future_steps, 3].
        mimic_ref_anchor_rot: Reference anchor rotations [envs, total_future_steps, 4].
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific 1-indexed step numbers (e.g., [1, 3, 5]).
        w_last: If True, quaternions are in XYZW format, else WXYZ.
        include_heading_delta: If True, append the 2D heading(yaw) delta
            (cos, sin) per step.

    Returns:
        Command observation [envs, selected_steps * (3 or 5)]:
        per step [heading_local_xyz_delta(3), (heading_delta_cos_sin(2))].
    """
    num_envs = current_state_anchor_pos.shape[0]

    ref_pos = select_step_indices(mimic_ref_anchor_pos, future_steps)  # [envs, steps, 3]
    ref_rot = select_step_indices(mimic_ref_anchor_rot, future_steps)  # [envs, steps, 4]
    steps = ref_pos.shape[1]

    origin_pos_exp = (
        current_state_anchor_pos.unsqueeze(1).expand(-1, steps, -1).reshape(-1, 3)
    )
    origin_rot_exp = (
        current_state_anchor_rot.unsqueeze(1).expand(-1, steps, -1).reshape(-1, 4)
    )
    target_pos_flat = ref_pos.reshape(-1, 3)

    xyz_delta = heading_local_xyz_delta(
        origin_pos_exp, origin_rot_exp, target_pos_flat, w_last
    )
    xyz_delta = xyz_delta.reshape(num_envs, steps, 3)

    obs_components = [xyz_delta]

    if include_heading_delta:
        target_rot_flat = ref_rot.reshape(-1, 4)
        current_heading_quat = rotations.calc_heading_quat(origin_rot_exp, w_last)
        ref_heading_quat = rotations.calc_heading_quat(target_rot_flat, w_last)
        rel_heading_quat = rotations.quat_mul(
            rotations.quat_conjugate(current_heading_quat, w_last),
            ref_heading_quat,
            w_last,
        )
        heading_tan_norm = rotations.quat_to_tan_norm(rel_heading_quat, w_last)
        heading_delta = heading_tan_norm[..., :2].reshape(num_envs, steps, 2)
        obs_components.append(heading_delta)

    obs = torch.cat(obs_components, dim=-1)
    return obs.reshape(num_envs, -1)


__all__ = ["build_mimic_future_displacement_cmd"]
