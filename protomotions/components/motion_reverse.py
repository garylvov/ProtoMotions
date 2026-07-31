# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Online reference-motion TIME-REVERSAL augmentation.

Motivation
----------
The corpus is walking-heavy in the FORWARD direction, and the policy is weak at
backwards walking. Playing a sampled reference clip time-reversed turns walking
forward into walking backward (and pick-up into put-down -- considered a
feature: reversed manipulation clips supply put-down data) without storing a
second copy of the corpus. Modeled exactly on the online sagittal mirror
augmentation (components/motion_mirror.py): per env-episode Bernoulli coin
(PM_REVERSE_PROB), ref-side transform applied at the motion-lib fetch level.

Semantics
---------
A time-reversed clip of length ``L`` evaluated at playback time ``t`` is the
original clip evaluated at time ``L - t``, with all time-derivatives negated:

* query time                     : ``t -> L - t`` (BEFORE frame-blend, inside
      ``get_motion_state``; the existing ``clip(0, L)`` bound then maps
      out-of-range future times past the reversed clip's end to the original
      clip's first frame -- the correct "hold last frame" behavior).
* poses (root/body pos+rot, dof pos): untouched -- the time remap alone serves
      the frames in reversed order (with the same inter-frame interpolation).
* linear velocities (``rigid_body_vel``)      : negate (d/dt sign flip).
* angular velocities (``rigid_body_ang_vel``) : negate.
* dof velocities (``dof_vel``)                : negate.
* contacts (``rigid_body_contacts``): instantaneous flags -> time remap only,
      no sign, no permutation.

Placement / composition
-----------------------
Applied inside ``MotionLib.get_motion_state`` -- the single entry point every
ref consumer uses (current-frame rewards, the future-steps obs window, the
historical window, RSI init states, markers). Because callers pass their own
(forward-advancing) times and the remap happens at fetch time, future-target
lookups automatically see a consistent reversed timeline: playback time
``t + k*dt`` fetches original time ``L - t - k*dt``, i.e. the frame ``k`` steps
EARLIER in the original clip -- exactly the reversed clip's future.

Composition with mirror: the mirror transform is a per-frame spatial map
(applied post-blend in the same function); time reversal is a temporal map plus
a uniform velocity negation. Full negation commutes with the mirror's
sign-flips and L/R permutation, so mirror(reverse(x)) == reverse(mirror(x)) and
the two coins are drawn independently (a clip can be both mirrored and
reversed).

Default OFF: with ``PM_REVERSE_PROB`` unset (and config ``reverse_prob=0``)
the per-env mask stays all-False, every call site passes ``reverse_flags=None``
and both hooks in ``get_motion_state`` are bypassed at the python level, so
training is byte-identical to no-reverse.
"""

import torch


def reverse_motion_times(
    motion_times: torch.Tensor,
    motion_lengths: torch.Tensor,
    flags: torch.Tensor,
) -> torch.Tensor:
    """Remap playback times onto the reversed timeline for flagged rows.

    Args:
        motion_times: query times [B] (may exceed clip length for future
            lookups; downstream frame-blend clips to [0, L]).
        motion_lengths: per-row clip lengths [B] (already gathered by
            motion_ids).
        flags: bool [B]; True rows are served time-reversed.

    Returns:
        New times tensor: ``L - t`` on flagged rows, ``t`` elsewhere. NOT
        clamped here -- ``_calc_frame_blend_from_id_and_time`` clips to
        ``[0, L]``, which maps past-the-end reversed lookups to the original
        first frame (the reversed clip's final frame), the correct hold.
    """
    flags = flags.to(motion_times.device)
    return torch.where(flags, motion_lengths - motion_times, motion_times)


def reverse_robot_state_velocities(state, flags: torch.Tensor):
    """Negate all time-derivative fields of a RobotState on flagged rows.

    HOT PATH -- called every step for the future-frame ref window (like
    ``mirror_robot_state``). Fully vectorized and sync-free: a per-row +/-1
    multiply (``-1`` on reversed rows, ``+1`` elsewhere) touches only the
    velocity fields; unreversed rows and all pose/contact fields are untouched.
    Callers must only invoke this when ``reverse_prob > 0`` (python-level
    bypass keeps PM_REVERSE_PROB unset byte-identical with zero added work).
    """
    if flags is None:
        return state
    dev = None
    for _f in ("rigid_body_vel", "dof_vel", "rigid_body_ang_vel"):
        _v = getattr(state, _f, None)
        if _v is not None:
            dev = _v.device
            break
    if dev is None:
        return state
    fb = flags.to(dev).view(-1, 1)  # [B,1] bool for column broadcasts
    fsign = torch.where(
        fb, torch.as_tensor(-1.0, device=dev), torch.as_tensor(1.0, device=dev)
    )
    # dof_vel is [B, D] -> fsign [B,1] broadcasts; body fields are [B, Bn, 3]
    # -> need [B,1,1].
    v = getattr(state, "dof_vel", None)
    if v is not None:
        v *= fsign
    fsign3 = fsign.view(-1, 1, 1)
    for field in ("rigid_body_vel", "rigid_body_ang_vel"):
        v = getattr(state, field, None)
        if v is None:
            continue
        v *= fsign3
    return state
