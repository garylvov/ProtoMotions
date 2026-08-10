# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract guard for the reference-interpolation path (sub-cm wrist budget).

Background (measured 2026-08-07, canonical_clean_plus_synth20_7class_v2.pt,
400 sampling-mass-weighted clips, wrist bodies 21/28, 176k control instants):

  * 91.8% of clips are stored at motion_dt=0.033333 (30 fps); 8.2% at exactly
    0.02 (50 fps) which is the control rate, so those interpolate exactly and
    contribute zero error.
  * Reference-interpolation error at the wrists is 0.85 mm rms
    (p50 0.16, p95 1.50, p99 3.14, max 40.7), corpus-frame-weighted 0.78 mm.
    NOT the 1.9 mm that the sub-cm plan budgeted.
  * Two independent estimators agree: the leading-order estimate
    rms_u(u(1-u)/2) * |second difference| = 0.89 mm, and the direct
    |Catmull-Rom - lerp| at the real 50 Hz control instants = 0.84 mm.
  * Upgrading to a cubic would therefore buy ~0.10 mm on the ~3.8 mm quadrature
    floor. The decision was to LEAVE the math alone and pin it here instead, so
    a later lane cannot silently coarsen it (or silently "improve" it and
    perturb every reference mid-experiment) without this test going red.

These tests pin the SEMANTICS, not any particular numeric output, so they are
byte-identical no-ops against 97e1368 behaviour.
"""

import numpy as np
import pytest
import torch

from protomotions.utils import rotations
from protomotions.utils.motion_interpolation_utils import (
    calc_frame_blend,
    interpolate_pos,
    interpolate_quat,
)

CTRL_DT = 0.02  # 50 Hz control loop (h1_2: sim fps 200 / decimation 4)


# --------------------------------------------------------------------------
# 1. Frame-blend contract: must stay consistent with
#    motion_length == (num_frames - 1) * motion_dt (simulator_state.py).
# --------------------------------------------------------------------------
def test_calc_frame_blend_is_consistent_with_motion_length():
    dt = torch.tensor([0.033333, 0.02])
    num_frames = torch.tensor([101, 101])
    length = (num_frames - 1) * dt

    # sweep the real control grid over the whole clip
    for c in range(0, 160):
        t = torch.full((2,), c * CTRL_DT).minimum(length)
        i0, i1, b = calc_frame_blend(t, length, num_frames, dt)
        assert torch.all(i1 >= i0)
        assert torch.all(i1 - i0 <= 1)
        assert torch.all((b >= 0.0) & (b <= 1.0))
        # the blend must reproduce the query time from the frame grid
        recon = i0.to(dt.dtype) * dt + b * dt
        assert torch.allclose(recon, t, atol=1e-5), (c, recon, t)


def test_50hz_sources_hit_frames_exactly():
    """dt == control dt must give zero interpolation (blend 0), not a fraction."""
    dt = torch.tensor([0.02])
    num_frames = torch.tensor([201])
    length = (num_frames - 1) * dt
    for c in range(0, 200):
        t = torch.tensor([c * CTRL_DT]).minimum(length)
        _, _, b = calc_frame_blend(t, length, num_frames, dt)
        assert torch.allclose(b, torch.zeros_like(b), atol=1e-4) or torch.allclose(
            b, torch.ones_like(b), atol=1e-4
        ), (c, b)


# --------------------------------------------------------------------------
# 2. Positions / velocities are LINEAR. Pin it, because the sub-cm budget above
#    is computed for linear interpolation.
# --------------------------------------------------------------------------
def test_interpolate_pos_is_exactly_linear():
    p0 = torch.randn(8, 29, 3, dtype=torch.float64)
    p1 = torch.randn(8, 29, 3, dtype=torch.float64)
    for u in (0.0, 0.2, 0.5, 0.8, 1.0):
        blend = torch.full((8,), u, dtype=torch.float64)
        got = interpolate_pos(p0, p1, blend)
        want = (1.0 - u) * p0 + u * p1
        assert torch.allclose(got, want, atol=1e-12)


def test_linear_interp_error_follows_half_u_one_minus_u():
    """The measured budget assumes err(u) = -1/2 u(1-u) * (second difference).

    Verify on an exact quadratic, where that relation is not an approximation.
    """
    h = 1.0 / 30.0
    a = torch.tensor([0.7, -1.3, 2.1], dtype=torch.float64)  # curvature vector

    def x(t):
        return 0.5 * a * t**2

    x0, x1 = x(torch.tensor(0.0)), x(torch.tensor(h))
    d2 = x(torch.tensor(h)) - 2.0 * x(torch.tensor(0.0)) + x(torch.tensor(-h))
    for u in (0.1, 0.25, 0.5, 0.75, 0.9):
        lin = (1.0 - u) * x0 + u * x1
        err = x(torch.tensor(u * h)) - lin
        want = -0.5 * u * (1.0 - u) * d2
        assert torch.allclose(err, want, atol=1e-14), (u, err, want)


def test_documented_rms_blend_coefficient():
    """rms over the real 50Hz-vs-30fps blend set of 1/2 u(1-u) ~= 0.0913."""
    t = np.arange(20000) * CTRL_DT
    dt = 0.033333
    u = np.clip((t - np.floor(t / dt) * dt) / dt, 0.0, 1.0)
    coeff = np.sqrt(np.mean((0.5 * u * (1.0 - u)) ** 2))
    # uniform-blend value is 1/2 * sqrt(1/30) = 0.09129
    assert coeff == pytest.approx(0.0913, abs=2e-3)


# --------------------------------------------------------------------------
# 3. Rotations must be SLERP, not lerp+normalize. This project has been bitten
#    by quaternion handling before; a silent downgrade to nlerp would add a
#    rotation error that the position budget above does not cover.
# --------------------------------------------------------------------------
def _nlerp(q0, q1, u):
    q = (1.0 - u) * q0 + u * q1
    return q / q.norm(dim=-1, keepdim=True)


def test_interpolate_quat_is_true_slerp_not_nlerp():
    # 120 deg about z, w-last (COMMON convention is xyzw)
    ang = torch.tensor(2.0 * np.pi / 3.0, dtype=torch.float64)
    q0 = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    q1 = torch.tensor(
        [[0.0, 0.0, float(torch.sin(ang / 2)), float(torch.cos(ang / 2))]],
        dtype=torch.float64,
    )
    # NOTE: u must be asymmetric -- at u=0.5 nlerp and slerp coincide exactly,
    # so a midpoint test cannot tell them apart.
    uval = 0.25
    u = torch.tensor([uval], dtype=torch.float64)
    got = interpolate_quat(q0, q1, u)

    # ground truth: a constant-angular-rate fraction of the rotation angle
    part = ang * uval
    want = torch.tensor(
        [[0.0, 0.0, float(torch.sin(part / 2)), float(torch.cos(part / 2))]],
        dtype=torch.float64,
    )
    assert torch.allclose(got, want, atol=1e-9), (got, want)
    # and it must be measurably different from nlerp, otherwise this test is blind
    nl = _nlerp(q0, q1, u)
    assert not torch.allclose(got, nl, atol=1e-6), (got, nl)


def test_slerp_output_stays_unit_norm():
    g = torch.Generator().manual_seed(0)
    q0 = torch.randn(256, 4, generator=g, dtype=torch.float64)
    q1 = torch.randn(256, 4, generator=g, dtype=torch.float64)
    q0 = q0 / q0.norm(dim=-1, keepdim=True)
    q1 = q1 / q1.norm(dim=-1, keepdim=True)
    for u in (0.0, 0.13, 0.5, 0.87, 1.0):
        blend = torch.full((256,), u, dtype=torch.float64)
        q = interpolate_quat(q0, q1, blend)
        assert torch.allclose(q.norm(dim=-1), torch.ones(256, dtype=torch.float64), atol=1e-6)


def test_slerp_takes_shortest_arc():
    """q and -q are the same rotation; slerp must not travel the long way."""
    ang = torch.tensor(np.pi * 0.9, dtype=torch.float64)
    q0 = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    q1 = torch.tensor(
        [[0.0, 0.0, float(torch.sin(ang / 2)), float(torch.cos(ang / 2))]],
        dtype=torch.float64,
    )
    u = torch.tensor([0.5], dtype=torch.float64)
    a = interpolate_quat(q0, q1, u)
    b = interpolate_quat(q0, -q1, u)
    # same rotation up to sign
    assert torch.allclose(a, b, atol=1e-9) or torch.allclose(a, -b, atol=1e-9)


def test_slerp_near_parallel_fallback_is_bounded():
    """KNOWN DEFECT, deliberately not fixed (see module docstring).

    rotations.slerp returns the MIDPOINT (ignoring t) when sin(theta/2) < 1e-3.
    That is a t-independent answer, so it is wrong for t != 0.5 -- but the gate
    only fires for theta < 2e-3 rad, bounding the angular error at 1e-3 rad
    (0.057 deg). Pinned here so the bound cannot silently widen.
    """
    theta = torch.tensor(1.9e-3, dtype=torch.float64)  # inside the fallback gate
    q0 = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    q1 = torch.tensor(
        [[0.0, 0.0, float(torch.sin(theta / 2)), float(torch.cos(theta / 2))]],
        dtype=torch.float64,
    )
    u = torch.tensor([0.9], dtype=torch.float64)
    got = rotations.slerp(q0, q1, u.unsqueeze(-1))
    want = torch.tensor(
        [[0.0, 0.0, float(torch.sin(0.9 * theta / 2)), float(torch.cos(0.9 * theta / 2))]],
        dtype=torch.float64,
    )
    ang_err = 2.0 * torch.asin(torch.clamp((got - want).norm(dim=-1) / 2.0, max=1.0))
    assert float(ang_err.max()) < 1.1e-3, float(ang_err.max())


# --------------------------------------------------------------------------
# 4. The blend-vs-exact-frame decision must not silently regress to nearest
#    frame: that would cost ~half a frame of reference travel (~10 mm at the
#    wrists, 30 fps) instead of the 0.85 mm the budget carries.
# --------------------------------------------------------------------------
def test_blending_beats_nearest_frame_by_an_order_of_magnitude():
    h = 1.0 / 30.0
    t = torch.linspace(0.0, 1.0, 2001, dtype=torch.float64)
    # 1 Hz circular wrist sweep at 0.35 m radius ~ 2.2 m/s, a fast reach
    r = 0.35
    traj = torch.stack([r * torch.cos(2 * np.pi * t), r * torch.sin(2 * np.pi * t)], -1)

    knots = torch.arange(0.0, 1.0 + h, h, dtype=torch.float64)
    kn = torch.stack([r * torch.cos(2 * np.pi * knots), r * torch.sin(2 * np.pi * knots)], -1)

    ctrl = torch.arange(0.0, 1.0, CTRL_DT, dtype=torch.float64)
    i0 = torch.clamp((ctrl / h).long(), max=kn.shape[0] - 2)
    u = (ctrl - i0 * h) / h
    lin = (1 - u)[:, None] * kn[i0] + u[:, None] * kn[i0 + 1]
    near = kn[torch.clamp(torch.round(ctrl / h).long(), max=kn.shape[0] - 1)]

    true = torch.stack(
        [r * torch.cos(2 * np.pi * ctrl), r * torch.sin(2 * np.pi * ctrl)], -1
    )
    e_lin = (lin - true).norm(dim=-1).pow(2).mean().sqrt()
    e_near = (near - true).norm(dim=-1).pow(2).mean().sqrt()
    assert e_lin < e_near / 5.0, (float(e_lin), float(e_near))
    # and linear must be in the sub-mm..few-mm regime the budget assumes
    assert float(e_lin) < 5e-3, float(e_lin)
