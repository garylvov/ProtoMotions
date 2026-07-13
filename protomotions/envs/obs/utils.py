# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for observation processing."""

from typing import Union, List

import torch
from torch import Tensor

from protomotions.utils import rotations


def heading_local_xy_delta(
    origin_pos: Tensor,
    origin_rot: Tensor,
    target_pos: Tensor,
    w_last: bool = True,
) -> Tensor:
    """Return target XY displacement in the origin heading frame.

    The vertical component is intentionally discarded before rotation so this
    helper can be shared by target-reaching observations and odometer-style
    target-pose offsets.
    """
    rel_pos = target_pos - origin_pos
    rel_pos = rel_pos.clone()
    rel_pos[..., 2] = 0.0
    heading_inv = rotations.calc_heading_quat_inv(origin_rot, w_last)
    return rotations.quat_rotate(heading_inv, rel_pos, w_last)[..., :2]


def heading_local_xyz_delta(
    origin_pos: Tensor,
    origin_rot: Tensor,
    target_pos: Tensor,
    w_last: bool = True,
) -> Tensor:
    """Return target XYZ displacement in the origin heading frame, Z preserved.

    Like ``heading_local_xy_delta`` but keeps the raw (unrotated) vertical
    delta instead of discarding it. Only the XY plane components of the
    delta are rotated into the origin's heading-aligned frame (yaw-only
    rotation about the world Z axis leaves Z untouched anyway), so this is
    equivalent to rotating the full 3D delta by the heading-inverse quaternion
    and is a pure delta: heading-invariant under a shared yaw rotation of the
    origin and target, and translation-invariant (no world-frame absolutes).

    Used for commands/rewards where vertical motion (e.g. deep-bend, pickup)
    must be observable, unlike ``heading_local_xy_delta`` which is meant for
    ground-plane-only offsets.
    """
    rel_pos = target_pos - origin_pos
    rel_xy = rel_pos.clone()
    rel_xy[..., 2] = 0.0
    heading_inv = rotations.calc_heading_quat_inv(origin_rot, w_last)
    rotated = rotations.quat_rotate(heading_inv, rel_xy, w_last)
    rotated = rotated.clone()
    rotated[..., 2] = rel_pos[..., 2]
    return rotated


def select_step_indices(
    tensor: Tensor,
    steps: Union[int, List[int]],
    dim: int = 1
) -> Tensor:
    """Select steps from tensor by index.

    Supports both consecutive steps (int) and arbitrary step indices (list).
    Uses 1-indexed step numbers that are converted to 0-indexed tensor positions.

    Args:
        tensor: Input tensor with steps along dim.
        steps: If int N, selects first N steps (like arange(1, N+1)).
               If list, selects specific 1-indexed steps (e.g., [1, 3, 5] -> indices [0, 2, 4]).
        dim: Dimension containing steps.

    Returns:
        Tensor with selected steps.
    """
    if isinstance(steps, int):
        return tensor.narrow(dim, 0, steps)
    else:
        indices = torch.tensor([s - 1 for s in steps], device=tensor.device, dtype=torch.long)
        return tensor.index_select(dim, indices)
