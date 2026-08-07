# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracking reward compute kernels for motion imitation.

Pure tensor functions (kernels) for computing tracking rewards.
Use MdpComponent in experiment configs to bind kernels to context paths:

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.tracking import compute_gt_rew
    
    reward_components = {
        "gt_rew": MdpComponent(
            compute_func=compute_gt_rew,
            dynamic_vars={
                "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
                "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            },
            static_params={"coefficient": -100.0},
        ),
    }

Includes:
- Standard AMP/DeepMimic-style tracking rewards (gt, gr, gv, gav, rh)
- BeyondMimic-style rewards (global/relative position, orientation, velocity)
"""

import torch
from torch import Tensor
from typing import Optional, Tuple

from protomotions.utils.rotations import (
    quat_angle_diff_norm,
    calc_heading,
    calc_heading_quat_inv,
    quat_rotate,
    quat_mul,
)
from protomotions.envs.rewards.base import mean_squared_error_exp, rotation_error_exp
from protomotions.envs.obs.utils import heading_local_xyz_delta


# =============================================================================
# Standard Tracking Reward Kernels
# =============================================================================

def compute_gt_rew(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    coefficient: float = -100.0,
) -> Tensor:
    """Position tracking reward (exponential MSE).
    
    Args:
        current_rigid_body_pos: Current body positions [num_envs, num_bodies, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        coefficient: Exponential coefficient for error.
    
    Returns:
        Reward tensor [num_envs].
    """
    return mean_squared_error_exp(
        current_rigid_body_pos,
        ref_rigid_body_pos,
        coefficient,
    )


def compute_gr_rew(
    current_rigid_body_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    coefficient: float = -5.0,
) -> Tensor:
    """Rotation tracking reward (exponential quaternion error).
    
    Args:
        current_rigid_body_rot: Current body rotations [num_envs, num_bodies, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4] (w-last).
        coefficient: Exponential coefficient for error.
    
    Returns:
        Reward tensor [num_envs].
    """
    return rotation_error_exp(
        current_rigid_body_rot,
        ref_rigid_body_rot,
        coefficient,
    )


def compute_gv_rew(
    current_rigid_body_vel: Tensor,
    ref_rigid_body_vel: Tensor,
    coefficient: float = -0.5,
) -> Tensor:
    """Velocity tracking reward (exponential MSE).
    
    Args:
        current_rigid_body_vel: Current body velocities [num_envs, num_bodies, 3].
        ref_rigid_body_vel: Reference body velocities [num_envs, num_bodies, 3].
        coefficient: Exponential coefficient for error.
    
    Returns:
        Reward tensor [num_envs].
    """
    return mean_squared_error_exp(
        current_rigid_body_vel,
        ref_rigid_body_vel,
        coefficient,
    )


def compute_gav_rew(
    current_rigid_body_ang_vel: Tensor,
    ref_rigid_body_ang_vel: Tensor,
    coefficient: float = -0.1,
) -> Tensor:
    """Angular velocity tracking reward (exponential MSE).
    
    Args:
        current_rigid_body_ang_vel: Current angular velocities [num_envs, num_bodies, 3].
        ref_rigid_body_ang_vel: Reference angular velocities [num_envs, num_bodies, 3].
        coefficient: Exponential coefficient for error.
    
    Returns:
        Reward tensor [num_envs].
    """
    return mean_squared_error_exp(
        current_rigid_body_ang_vel,
        ref_rigid_body_ang_vel,
        coefficient,
    )


def compute_rh_rew(
    current_root_height: Tensor,
    ref_rigid_body_pos: Tensor,
    coefficient: float = -100.0,
) -> Tensor:
    """Root height tracking reward (exponential MSE).
    
    Args:
        current_root_height: Current root height [num_envs] or [num_envs, 1].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        coefficient: Exponential coefficient for error.
    
    Returns:
        Reward tensor [num_envs].
    """
    # Extract reference root height (z-coordinate of root body)
    ref_root_height = ref_rigid_body_pos[:, 0, 2]
    
    return mean_squared_error_exp(
        current_root_height,
        ref_root_height,
        coefficient,
    )


# =============================================================================
# BeyondMimic-style Reward Kernels
# =============================================================================

def _dual_sigma_exp(
    error: Tensor,
    sigma: float,
    fine_weight: float = 0.0,
    fine_sigma: Optional[float] = None,
) -> Tensor:
    """Gaussian tracking kernel with an OPTIONAL narrow (fine) companion.

    The single-sigma Gaussian ``exp(-e^2 / sigma^2)`` used by every position
    tracking term here is nearly FLAT for errors far below ``sigma``: at
    sigma=0.3 m a 2 cm error scores 0.9956 and its gradient is ~0.44 /m, so a
    few-centimetre static-hold drift is effectively free while the term still
    reads ~1.0. Tightening ``sigma`` alone fixes the dead zone but destroys the
    CAPTURE RANGE -- at sigma=0.05 m a 30 cm error scores e^-36 ~ 2e-16 with a
    vanishing gradient, so a policy that is far off gets no signal to come back.

    This kernel keeps the coarse Gaussian EXACTLY as-is (capture range and
    large-error gradient untouched) and optionally ADDS a narrow companion::

        r = exp(-e^2 / sigma_coarse^2) + fine_weight * exp(-e^2 / sigma_fine^2)

    The companion is essentially zero beyond a few ``fine_sigma``, so it changes
    nothing about large-error behavior; near zero it supplies the precision
    gradient the coarse term lacks. Because the MdpComponent ``weight`` w
    multiplies the whole kernel, the effective decomposition is
    ``w * coarse + (w * fine_weight) * fine`` -- i.e. ``fine_weight`` is the
    fine term's weight RELATIVE to the coarse one, and the term's maximum value
    rises from w to ``w * (1 + fine_weight)``.

    RESUME RULE (Rule 10): ``fine_weight`` defaults to 0.0 and the OFF path
    returns LITERALLY the original expression ``torch.exp(-error / (sigma**2))``
    -- not an algebraically-equal rewrite -- so a frozen/unpickled config whose
    static_params lack the new keys is BYTE-IDENTICAL (bit-exact in float32, no
    ulp drift). Behavior changes ONLY when the explicit env knobs are set.

    Args:
        error: SQUARED error already reduced to [num_envs] (e.g. squared
            distance in m^2, or mean squared per-DOF error in rad^2).
        sigma: Coarse Gaussian kernel width (capture range). Unchanged.
        fine_weight: Relative weight of the narrow companion. 0.0 (default) =
            companion absent = byte-identical to the single-sigma kernel.
            Must be >= 0.
        fine_sigma: Narrow Gaussian kernel width. Required (and must be > 0)
            whenever ``fine_weight`` is non-zero.

    Returns:
        Reward tensor with the same shape as ``error``.
    """
    if not fine_weight:
        # BYTE-IDENTICAL OFF PATH -- do not "simplify" this into the dual form.
        return torch.exp(-error / (sigma ** 2))
    if fine_weight < 0.0:
        raise ValueError(
            f"fine_weight must be >= 0 (got {fine_weight}); the fine companion "
            "is a REWARD bonus near zero error, a negative weight would carve a "
            "hole at the target."
        )
    if fine_sigma is None:
        raise ValueError(
            f"fine_sigma is required when fine_weight={fine_weight} is non-zero "
            "(no default is assumed: the narrow width is the whole point)."
        )
    if fine_sigma <= 0.0:
        raise ValueError(f"fine_sigma must be > 0 (got {fine_sigma}).")
    return torch.exp(-error / (sigma ** 2)) + fine_weight * torch.exp(
        -error / (fine_sigma ** 2)
    )


def _weighted_body_mean(error: Tensor, weights: Optional[Tensor]) -> Tensor:
    """Reduce a [num_envs, num_bodies] per-body error to [num_envs].

    Uniform mean when ``weights`` is None (unchanged behavior). Otherwise a
    weighted mean normalized by the sum of weights: sum(w_i * err_i) / sum(w_i).
    """
    if weights is None:
        return error.mean(dim=-1)
    w = torch.as_tensor(weights, dtype=error.dtype, device=error.device)
    return (error * w).sum(dim=-1) / w.sum()


def compute_global_position_error_exp(
    x: Tensor,
    ref_x: Tensor,
    sigma: float,
    indices: Optional[Tensor] = None,
    weights: Optional[Tensor] = None,
    fine_weight: float = 0.0,
    fine_sigma: Optional[float] = None,
) -> Tensor:
    """Position error: exp(-||x - ref_x||^2 / sigma^2).

    Args:
        x: Current positions [num_envs, num_bodies, 3] or [num_envs, 3].
        ref_x: Reference positions (same shape as x).
        sigma: Gaussian kernel width.
        indices: Optional body indices to select [num_bodies_subset].
        weights: Optional per-body weight multipliers [num_bodies_subset],
            aligned with ``indices`` (or with the body dim if ``indices`` is
            None). None = uniform mean (default, backward compatible).
        fine_weight: Optional relative weight of a NARROW Gaussian companion
            added on top of the coarse one (see ``_dual_sigma_exp``). 0.0
            (default) = absent = byte-identical single-sigma kernel.
        fine_sigma: Narrow companion width; required when ``fine_weight`` != 0.

    Returns:
        Reward tensor [num_envs].
    """
    if indices is not None and x.dim() == 3:
        x = x[:, indices]
        ref_x = ref_x[:, indices]

    error = (x - ref_x).pow(2).sum(dim=-1)
    if error.dim() == 2:
        error = _weighted_body_mean(error, weights)
    return _dual_sigma_exp(error, sigma, fine_weight, fine_sigma)


def compute_global_anchor_pos_rew(
    current_anchor_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    anchor_idx: int,
    sigma: float = 0.3,
    fine_weight: float = 0.0,
    fine_sigma: Optional[float] = None,
) -> Tensor:
    """Global anchor position reward (BeyondMimic style).

    NOTE (scope): the "anchor" is the ROOT/pelvis body only -- this term does
    NOT see the hands. Sharpening it tightens WORLD-frame root placement, not
    hand precision (see ``compute_relative_body_pos_rew`` for the hand).

    Args:
        current_anchor_pos: Current anchor position [num_envs, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        anchor_idx: Index of anchor body.
        sigma: Gaussian kernel width.
        fine_weight: Optional relative weight of a NARROW Gaussian companion
            (see ``_dual_sigma_exp``). 0.0 (default) = absent = byte-identical.
        fine_sigma: Narrow companion width; required when ``fine_weight`` != 0.

    Returns:
        Reward: exp(-||anchor_pos - ref_anchor_pos||^2 / sigma^2), plus the
        optional narrow companion.
    """
    ref_anchor_pos = ref_rigid_body_pos[:, anchor_idx, :]
    return compute_global_position_error_exp(
        current_anchor_pos,
        ref_anchor_pos,
        sigma,
        fine_weight=fine_weight,
        fine_sigma=fine_sigma,
    )


def compute_global_body_pos_rew(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    sigma: float = 0.3,
    body_indices: Optional[Tensor] = None,
    body_weights: Optional[Tensor] = None,
    fine_weight: float = 0.0,
    fine_sigma: Optional[float] = None,
    reference_still_mask: Optional[Tensor] = None,
) -> Tensor:
    """WORLD-frame body position tracking -- the missing hand-in-the-world term.

    WHY THIS EXISTS (frame audit, 2026-08-04). Every body-position term in the
    canonical teacher stack is ANCHOR-RELATIVE:
    ``compute_relative_body_pos_rew`` (bound as both ``relative_body_pos`` and
    the wrist-only ``wrist_relative_body_pos``) subtracts the pelvis position
    from every body on BOTH the current and the reference side and then rotates
    into the pelvis's YAW-aligned frame::

        current_rel_pos = current_rigid_body_pos - current_anchor_pos
        ref_rel_pos     = ref_rigid_body_pos     - ref_anchor_pos

    Consequences, exactly:

    - Pelvis TRANSLATION cancels IDENTICALLY on both sides. A hand riding a
      swaying base is scored as PERFECT. This is invisible to the reward.
    - Pelvis YAW cancels too (``calc_heading_quat_inv`` is yaw-only).
    - Pelvis PITCH/ROLL do NOT cancel, so tilt is partially visible -- but tilt
      is the minority share of the drift.

    The remaining position terms (``global_anchor_pos``,
    ``heading_local_anchor_drift``) score the ROOT only, and ``dof_pos_track``
    scores JOINT angles -- which actively OPPOSES compensation, since bending
    the arm to cancel base sway is a deviation from the reference posture.

    Net: NOTHING in the stack rewards the hand for being in the right place in
    the WORLD. This kernel does. It scores the world-frame per-body position
    error directly, so an arm that actively cancels base motion is PAID for it,
    which is the stated design target ("the pelvis may sway; the hand must
    stay put"). No pelvis stabilization is implied or required.

    Restrict it to the hand/wrist bodies via ``body_indices``; the pelvis is
    deliberately NOT a recommended member of that set.

    WORLD-FRAME CAVEAT: with ``realign_motion_with_humanoid_on_each_step=False``
    the reference and the robot can separate in world coordinates over an
    episode. That separation is already bounded by the ``anchor_pos_drift``
    termination (0.4 m) and priced by ``global_anchor_pos``, which lives in the
    same world frame at weight 0.5 -- so this term inherits an established,
    bounded convention rather than a new one. During a static hold (the case
    that motivated it) the separation is exactly what we want measured.

    Args:
        current_rigid_body_pos: Current body positions [num_envs, num_bodies, 3]
            in WORLD coordinates.
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3]
            in WORLD coordinates.
        sigma: Coarse Gaussian width (m). Preserves capture range.
        body_indices: Body indices to score (e.g. the two wrist bodies).
            None = all bodies.
        body_weights: Optional per-body multipliers aligned with
            ``body_indices``. None = uniform mean.
        fine_weight: Optional relative weight of the NARROW companion
            (see ``_dual_sigma_exp``). 0.0 = absent.
        fine_sigma: Narrow companion width (m); required when
            ``fine_weight`` != 0.
        reference_still_mask: Optional [num_envs] bool/float mask. When given,
            the reward is zeroed outside held-reference envs (the HOLD-FIX
            gate). None (default) = the term applies at ALL times, which is the
            recommended setting: base sway displaces the hand during motion
            too, and an on/off gate introduces a reward discontinuity at the
            gate boundary for the policy to exploit.

    Returns:
        Reward tensor [num_envs].
    """
    reward = compute_global_position_error_exp(
        current_rigid_body_pos,
        ref_rigid_body_pos,
        sigma,
        body_indices,
        body_weights,
        fine_weight=fine_weight,
        fine_sigma=fine_sigma,
    )
    if reference_still_mask is not None:
        reward = reward * reference_still_mask.to(reward.dtype)
    return reward


def compute_global_orientation_error_exp(
    q: Tensor,
    ref_q: Tensor,
    sigma: float,
    indices: Optional[Tensor] = None,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Orientation error: exp(-angle_diff^2 / sigma^2).

    Args:
        q: Current orientations [num_envs, num_bodies, 4] or [num_envs, 4] (w-last).
        ref_q: Reference orientations (same shape as q).
        sigma: Gaussian kernel width.
        indices: Optional body indices to select [num_bodies_subset].
        weights: Optional per-body weight multipliers [num_bodies_subset],
            aligned with ``indices`` (or with the body dim if ``indices`` is
            None). None = uniform mean (default, backward compatible).

    Returns:
        Reward tensor [num_envs].
    """
    if indices is not None and q.dim() == 3:
        q = q[:, indices]
        ref_q = ref_q[:, indices]

    error = quat_angle_diff_norm(q, ref_q, w_last=True)
    if error.dim() == 2:
        error = _weighted_body_mean(error, weights)
    return torch.exp(-error / (sigma ** 2))


def compute_global_anchor_ori_rew(
    current_anchor_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    anchor_idx: int,
    sigma: float = 0.4,
) -> Tensor:
    """Global anchor orientation reward (BeyondMimic style).
    
    Args:
        current_anchor_rot: Current anchor rotation [num_envs, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4] (w-last).
        anchor_idx: Index of anchor body.
        sigma: Gaussian kernel width.
    
    Returns:
        Reward: exp(-angle_diff^2 / sigma^2).
    """
    ref_anchor_rot = ref_rigid_body_rot[:, anchor_idx, :]
    return compute_global_orientation_error_exp(current_anchor_rot, ref_anchor_rot, sigma)


def compute_anchor_relative_local_body_pos(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    current_anchor_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    current_anchor_pos: Tensor,
    anchor_idx: int,
) -> Tuple[Tensor, Tensor]:
    """Put current and reference body positions in the ANCHOR-RELATIVE,
    HEADING-LOCAL frame -- the frame ``wrist_relative_body_pos`` scores in.

    Extracted VERBATIM out of ``compute_relative_body_pos_rew`` (2026-08-07) so
    that the reward and the new ``env/wrist_relative_body_pos/err_m_mean`` stat
    read the SAME arithmetic, in the same order, and can never drift. Not one
    operation was reordered, retyped or vectorised in the move: the stat must
    measure the error the reward actually prices, and a "harmless" rewrite is
    exactly how a metric silently stops describing its term.
    ``test_wrist_relative_body_pos_stat.py`` pins the reward's output as
    bitwise-equal to an inline copy of the pre-refactor expression.

    Args:
        current_rigid_body_pos: Current body positions [num_envs, num_bodies, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        current_anchor_rot: Current anchor rotation [num_envs, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4].
        current_anchor_pos: Current anchor position [num_envs, 3].
        anchor_idx: Index of anchor body.

    Returns:
        ``(current_rel_pos_local, ref_rel_pos_local)``, each
        [num_envs, num_bodies, 3].
    """
    # Extract reference anchor pos and rot
    ref_anchor_pos = ref_rigid_body_pos[:, anchor_idx, :]
    ref_anchor_rot = ref_rigid_body_rot[:, anchor_idx, :]
    
    # Compute heading rotations (yaw-only)
    current_heading_rot_inv = calc_heading_quat_inv(current_anchor_rot, w_last=True)
    ref_heading_rot_inv = calc_heading_quat_inv(ref_anchor_rot, w_last=True)
    
    # Compute relative positions in world frame
    current_rel_pos = current_rigid_body_pos - current_anchor_pos.unsqueeze(1)
    ref_rel_pos = ref_rigid_body_pos - ref_anchor_pos.unsqueeze(1)
    
    # Rotate to anchor's local frame
    current_rel_pos_flat = current_rel_pos.reshape(-1, 3)
    current_heading_rot_inv_exp = current_heading_rot_inv.unsqueeze(1).expand(
        -1, current_rigid_body_pos.shape[1], -1
    ).reshape(-1, 4)
    current_rel_pos_local = quat_rotate(
        current_heading_rot_inv_exp, current_rel_pos_flat, w_last=True
    ).reshape(current_rigid_body_pos.shape)
    
    ref_rel_pos_flat = ref_rel_pos.reshape(-1, 3)
    ref_heading_rot_inv_exp = ref_heading_rot_inv.unsqueeze(1).expand(
        -1, ref_rigid_body_pos.shape[1], -1
    ).reshape(-1, 4)
    ref_rel_pos_local = quat_rotate(
        ref_heading_rot_inv_exp, ref_rel_pos_flat, w_last=True
    ).reshape(ref_rigid_body_pos.shape)
    return current_rel_pos_local, ref_rel_pos_local


def compute_relative_body_pos_rew(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    current_anchor_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    current_anchor_pos: Tensor,
    anchor_idx: int,
    sigma: float = 0.3,
    body_indices: Optional[Tensor] = None,
    body_weights: Optional[Tensor] = None,
    fine_weight: float = 0.0,
    fine_sigma: Optional[float] = None,
) -> Tensor:
    """Relative body position reward (BeyondMimic style).

    Computes reward based on body positions relative to anchor in anchor's local frame.

    THIS is the kernel that actually governs HAND/WRIST placement in the
    canonical teacher stack: bound once over all bodies ("relative_body_pos")
    and once restricted to the two wrist bodies ("wrist_relative_body_pos",
    ``body_indices=[left_wrist_yaw_link, right_wrist_yaw_link]``). The frame is
    ANCHOR-RELATIVE and HEADING-LOCAL, i.e. hand position measured in the
    robot's own pelvis/heading frame -- NOT world coordinates.

    Args:
        current_rigid_body_pos: Current body positions [num_envs, num_bodies, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        current_anchor_rot: Current anchor rotation [num_envs, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4] (w-last).
        current_anchor_pos: Current anchor position [num_envs, 3].
        anchor_idx: Index of anchor body.
        sigma: Gaussian kernel width.
        body_indices: Optional body indices to select [num_bodies_subset].
        body_weights: Optional per-body weight multipliers [num_bodies_subset],
            aligned with ``body_indices``. None = uniform mean over the
            selected bodies (default, backward compatible).
        fine_weight: Optional relative weight of a NARROW Gaussian companion
            (see ``_dual_sigma_exp``). 0.0 (default) = absent = byte-identical.
        fine_sigma: Narrow companion width; required when ``fine_weight`` != 0.

    Returns:
        Reward: exp(-||rel_pos - ref_rel_pos||^2 / sigma^2), plus the optional
        narrow companion.
    """
    current_rel_pos_local, ref_rel_pos_local = (
        compute_anchor_relative_local_body_pos(
            current_rigid_body_pos,
            ref_rigid_body_pos,
            current_anchor_rot,
            ref_rigid_body_rot,
            current_anchor_pos,
            anchor_idx,
        )
    )
    return compute_global_position_error_exp(
        current_rel_pos_local,
        ref_rel_pos_local,
        sigma,
        body_indices,
        body_weights,
        fine_weight=fine_weight,
        fine_sigma=fine_sigma,
    )


def compute_relative_body_ori_rew(
    current_rigid_body_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    current_anchor_rot: Tensor,
    anchor_idx: int,
    sigma: float = 0.4,
    body_indices: Optional[Tensor] = None,
    body_weights: Optional[Tensor] = None,
) -> Tensor:
    """Relative body orientation reward (BeyondMimic style).

    Computes reward based on body orientations relative to anchor.

    Args:
        current_rigid_body_rot: Current body rotations [num_envs, num_bodies, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4] (w-last).
        current_anchor_rot: Current anchor rotation [num_envs, 4] (w-last).
        anchor_idx: Index of anchor body.
        sigma: Gaussian kernel width.
        body_indices: Optional body indices to select [num_bodies_subset].
        body_weights: Optional per-body weight multipliers [num_bodies_subset],
            aligned with ``body_indices``. None = uniform mean over the
            selected bodies (default, backward compatible).

    Returns:
        Reward: exp(-angle_diff^2 / sigma^2).
    """
    # Extract reference anchor rotation
    ref_anchor_rot = ref_rigid_body_rot[:, anchor_idx, :]
    
    # Compute heading rotations (yaw-only)
    current_heading_rot_inv = calc_heading_quat_inv(current_anchor_rot, w_last=True)
    ref_heading_rot_inv = calc_heading_quat_inv(ref_anchor_rot, w_last=True)
    
    # Compute relative rotations
    current_heading_rot_inv_exp = current_heading_rot_inv.unsqueeze(1).expand(
        -1, current_rigid_body_rot.shape[1], -1
    )
    current_rel_rot = quat_mul(current_heading_rot_inv_exp, current_rigid_body_rot, w_last=True)
    
    ref_heading_rot_inv_exp = ref_heading_rot_inv.unsqueeze(1).expand(
        -1, ref_rigid_body_rot.shape[1], -1
    )
    ref_rel_rot = quat_mul(ref_heading_rot_inv_exp, ref_rigid_body_rot, w_last=True)
    
    return compute_global_orientation_error_exp(
        current_rel_rot, ref_rel_rot, sigma, body_indices, body_weights
    )


def compute_dof_pos_track_rew(
    current_dof_pos: Tensor,
    ref_dof_pos: Tensor,
    sigma: float = 0.35,
    indices: Optional[Tensor] = None,
    fine_weight: float = 0.0,
    fine_sigma: Optional[float] = None,
) -> Tensor:
    """Joint-space (DOF) position tracking reward: exp(-mean(dof_err^2) / sigma^2).

    Pins the policy to the REFERENCE posture in JOINT space, not only in the
    Cartesian body-position space the BeyondMimic ``relative_body_pos`` reward
    scores. Because the arm/leg kinematics are redundant, many joint
    configurations satisfy a given set of body positions, so a policy can track
    bodies while drifting to a LOOSE per-DOF posture that only stands up in one
    simulator -- the sim2sim posture-divergence failure mode (loose optimum
    joint err ~1.5 rad vs ~0.30 rad for a transferring policy). This kernel
    scores the mean squared per-DOF error against the reference joint positions
    with a Gaussian kernel, mirroring ``compute_global_position_error_exp`` (the
    sigma form): a tight posture earns ~1 and a loose one falls off sharply, so
    the reward stops being indifferent between the two.

    Args:
        current_dof_pos: Current joint positions [num_envs, num_dofs] (rad).
        ref_dof_pos: Reference joint positions [num_envs, num_dofs] (rad).
        sigma: Gaussian kernel width (rad). Smaller = tighter posture demand.
        indices: Optional DOF indices to restrict to a subset.
        fine_weight: Optional relative weight of a NARROW Gaussian companion
            (see ``_dual_sigma_exp``). 0.0 (default) = absent = byte-identical.
        fine_sigma: Narrow companion width (rad); required when
            ``fine_weight`` != 0.

    Returns:
        Reward tensor [num_envs] in (0, 1] (or (0, 1 + fine_weight] when the
        narrow companion is enabled).
    """
    if indices is not None:
        current_dof_pos = current_dof_pos[:, indices]
        ref_dof_pos = ref_dof_pos[:, indices]

    error = (current_dof_pos - ref_dof_pos).pow(2).mean(dim=-1)
    return _dual_sigma_exp(error, sigma, fine_weight, fine_sigma)


def compute_global_body_lin_vel_rew(
    current_rigid_body_vel: Tensor,
    ref_rigid_body_vel: Tensor,
    sigma: float = 1.0,
) -> Tensor:
    """Global body linear velocity reward (BeyondMimic style).
    
    Args:
        current_rigid_body_vel: Current body velocities [num_envs, num_bodies, 3].
        ref_rigid_body_vel: Reference body velocities [num_envs, num_bodies, 3].
        sigma: Gaussian kernel width.
    
    Returns:
        Reward: exp(-||vel - ref_vel||^2 / sigma^2).
    """
    return compute_global_position_error_exp(current_rigid_body_vel, ref_rigid_body_vel, sigma)


def compute_global_body_ang_vel_rew(
    current_rigid_body_ang_vel: Tensor,
    ref_rigid_body_ang_vel: Tensor,
    sigma: float = 3.14,
) -> Tensor:
    """Global body angular velocity reward (BeyondMimic style).
    
    Args:
        current_rigid_body_ang_vel: Current angular velocities [num_envs, num_bodies, 3].
        ref_rigid_body_ang_vel: Reference angular velocities [num_envs, num_bodies, 3].
        sigma: Gaussian kernel width.
    
    Returns:
        Reward: exp(-||ang_vel - ref_ang_vel||^2 / sigma^2).
    """
    return compute_global_position_error_exp(
        current_rigid_body_ang_vel, ref_rigid_body_ang_vel, sigma
    )


def compute_gt_rel_rew(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    current_anchor_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    anchor_idx: int,
    coefficient: float = -100.0,
    body_indices=None,
) -> Tensor:
    """Heading-local anchor-relative body position tracking reward.

    Subtracts the anchor position from all body positions and rotates into the
    heading-aligned frame before computing exponential MSE.  Invariant to global
    XY translation and yaw heading, so it remains well-defined when
    ``realign_motion_with_humanoid_on_each_step=False``.

    Args:
        current_rigid_body_pos: Current body positions [num_envs, num_bodies, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        current_anchor_rot: Current anchor rotation [num_envs, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4] (w-last).
        anchor_idx: Index of the anchor body.
        coefficient: Exponential coefficient for error.
        body_indices: Optional list of body indices to restrict to a subset.

    Returns:
        Reward tensor [num_envs].
    """
    ref_anchor_pos = ref_rigid_body_pos[:, anchor_idx, :]
    ref_anchor_rot = ref_rigid_body_rot[:, anchor_idx, :]
    current_anchor_pos = current_rigid_body_pos[:, anchor_idx, :]

    current_heading_inv = calc_heading_quat_inv(current_anchor_rot, w_last=True)
    ref_heading_inv = calc_heading_quat_inv(ref_anchor_rot, w_last=True)

    current_rel = current_rigid_body_pos - current_anchor_pos.unsqueeze(1)
    ref_rel = ref_rigid_body_pos - ref_anchor_pos.unsqueeze(1)

    if body_indices is not None:
        current_rel = current_rel[:, body_indices]
        ref_rel = ref_rel[:, body_indices]

    N, B, _ = current_rel.shape
    cur_h = current_heading_inv.unsqueeze(1).expand(-1, B, -1).reshape(-1, 4)
    ref_h = ref_heading_inv.unsqueeze(1).expand(-1, B, -1).reshape(-1, 4)
    current_local = quat_rotate(cur_h, current_rel.reshape(-1, 3), w_last=True).reshape(N, B, 3)
    ref_local = quat_rotate(ref_h, ref_rel.reshape(-1, 3), w_last=True).reshape(N, B, 3)

    return mean_squared_error_exp(current_local, ref_local, coefficient)


def compute_anchor_xy_rew(
    current_anchor_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    anchor_idx: int,
    coefficient: float = -20.0,
) -> Tensor:
    """Anchor XY position tracking reward (exponential MSE).

    Analogous to ``compute_rh_rew`` but for XY coordinates.  Provides a loose
    global XY position signal when ``realign_motion_with_humanoid_on_each_step``
    is off.  The coefficient should be kept small relative to ``compute_rh_rew``
    since odometer-based XY is inherently noisier than height.

    Args:
        current_anchor_pos: Current anchor position [num_envs, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        anchor_idx: Index of the anchor body in ref_rigid_body_pos.
        coefficient: Exponential coefficient for error.

    Returns:
        Reward tensor [num_envs].
    """
    ref_anchor_xy = ref_rigid_body_pos[:, anchor_idx, :2]
    current_xy = current_anchor_pos[:, :2]
    return mean_squared_error_exp(current_xy, ref_anchor_xy, coefficient)


def compute_root_xy_displacement_rew(
    current_root_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    coefficient: float = -20.0,
) -> Tensor:
    """Root XY displacement tracking reward (exponential MSE).

    Track D Option-B fallback: exp-kernel on the root xy error relative to the
    reference motion (objective 1 — minimize xy displacement).  The kernel is
    invariant to the frame the error is expressed in (only the magnitude
    matters), so no heading-frame rotation is needed here.

    Args:
        current_root_pos: Current root position [num_envs, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3]
            (root is body 0).
        coefficient: Exponential coefficient for error.

    Returns:
        Reward tensor [num_envs] in (0, 1].
    """
    ref_root_xy = ref_rigid_body_pos[:, 0, :2]
    current_xy = current_root_pos[:, :2]
    return mean_squared_error_exp(current_xy, ref_root_xy, coefficient)


def compute_root_heading_rew(
    current_root_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    coefficient: float = -2.0,
) -> Tensor:
    """Root heading tracking reward (exponential squared wrapped-angle error).

    Track D Option-B fallback: exp-kernel on the wrapped heading (yaw) error
    relative to the reference motion (objective 1 — minimize heading
    displacement).

    Args:
        current_root_rot: Current root rotation [num_envs, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4]
            (w-last, root is body 0).
        coefficient: Exponential coefficient for error.

    Returns:
        Reward tensor [num_envs] in (0, 1].
    """
    ref_heading = calc_heading(ref_rigid_body_rot[:, 0], w_last=True)
    cur_heading = calc_heading(current_root_rot, w_last=True)
    heading_err = ref_heading - cur_heading
    heading_err = torch.remainder(heading_err + torch.pi, 2 * torch.pi) - torch.pi
    return heading_err.pow(2).mul(coefficient).exp()


def compute_heading_local_anchor_drift_rew(
    current_anchor_pos: Tensor,
    current_anchor_rot: Tensor,
    ref_anchor_pos: Tensor,
    sigma: float = 0.3,
    fine_weight: float = 0.0,
    fine_sigma: Optional[float] = None,
) -> Tensor:
    """Heading-local anchor drift reward (the reward twin of the future-
    displacement command observation, ``build_mimic_future_displacement_cmd``).

    Computes the CURRENT-time drift between the actual and reference anchor
    position in the current anchor's heading-aligned frame (XYZ, Z
    preserved) and scores it with a Gaussian kernel: exp(-||drift||^2 / sigma^2).
    Zero drift gives a reward of 1; the reward falls off as drift grows,
    with ``sigma`` controlling how quickly.

    Args:
        current_anchor_pos: Current anchor position [num_envs, 3].
        current_anchor_rot: Current anchor rotation [num_envs, 4] (w-last).
        ref_anchor_pos: Reference (current-time) anchor position [num_envs, 3].
        sigma: Gaussian kernel width.
        fine_weight: Optional relative weight of a NARROW Gaussian companion
            (see ``_dual_sigma_exp``). 0.0 (default) = absent = byte-identical.
        fine_sigma: Narrow companion width; required when ``fine_weight`` != 0.

    Returns:
        Reward tensor [num_envs].
    """
    drift = heading_local_xyz_delta(
        current_anchor_pos, current_anchor_rot, ref_anchor_pos, w_last=True
    )
    error = drift.pow(2).sum(dim=-1)
    return _dual_sigma_exp(error, sigma, fine_weight, fine_sigma)


__all__ = [
    # Standard tracking rewards
    "compute_gt_rew",
    "compute_gr_rew",
    "compute_gv_rew",
    "compute_gav_rew",
    "compute_rh_rew",
    # Heading-local relative tracking (realign=OFF compatible)
    "compute_gt_rel_rew",
    "compute_anchor_xy_rew",
    # Track D root displacement rewards (Option-B fallback, dormant)
    "compute_root_xy_displacement_rew",
    "compute_root_heading_rew",
    # BeyondMimic-style rewards
    "compute_global_position_error_exp",
    "compute_global_anchor_pos_rew",
    "compute_global_orientation_error_exp",
    "compute_global_anchor_ori_rew",
    "compute_relative_body_pos_rew",
    "compute_global_body_pos_rew",
    "compute_relative_body_ori_rew",
    "compute_dof_pos_track_rew",
    "compute_global_body_lin_vel_rew",
    "compute_global_body_ang_vel_rew",
    # Heading-local anchor drift reward (twin of build_mimic_future_displacement_cmd)
    "compute_heading_local_anchor_drift_rew",
]
