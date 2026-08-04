# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for reference-conditioned (anti-cheat) payload wrench modulation
(night13 t8, 2026-07-15).

The "bag" payload is HEAVY when the arms are near the chest (short lever arm,
mechanically sane) and LIGHT when outstretched. The load is scaled by the
REFERENCE motion's wrist->chest distance (the mocap target), which is EXOGENOUS
to the policy: the robot cannot change the demonstration by moving, so it cannot
shed load by extending its arms. A naive scale on the robot's LIVE wrist->chest
distance would be reward-hackable (extend arms -> lighter); conditioning on the
reference makes the anti-cheat guarantee structural.

Exercises the scheduler + reference-scale plumbing in isolation (no backend): a
harness object borrows the scheduler/setter methods from the abstract
``Simulator`` and stubs the two backend hooks. Mirrors
``test_ramped_force_dr.py``.

Covers:
  1. near-chest reference (d <= near)   -> scale ~ 1.0        -> force ~ F_max.
  2. outstretched reference (d >= far)  -> scale ~ far_scale  -> force ~ F_max*far_scale.
  3. ANTI-CHEAT: reference fixed, live/robot pose varied arbitrarily -> applied
     wrench IDENTICAL; the pure scale function literally cannot take live pose.
  4. Backward-compat: reference_distance_modulation=False -> buffer all ones ->
     forces byte-for-byte identical to the pre-feature path.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

from protomotions.simulator.base_simulator.config import (
    WrenchDomainRandomizationConfig,
)
from protomotions.simulator.base_simulator.simulator import (
    Simulator,
    reference_wrench_scale,
)


# Reference body layout used throughout: chest at index 0, two wrists after.
BODY_NAMES = [
    "torso_link",           # chest / anchor (index 0)
    "left_wrist_yaw_link",  # index 1
    "right_wrist_yaw_link",  # index 2
    "left_ankle_link",      # unrelated body (index 3)
]


class _Host:
    """Minimal harness carrying only the attributes the scheduler/setter touch."""

    def __init__(self, wrench_cfg=None, num_envs=4, dt=0.02):
        self.config = SimpleNamespace(
            domain_randomization=SimpleNamespace(
                wrench=wrench_cfg, sustained_wrench=None
            )
        )
        self.device = torch.device("cpu")
        self.num_envs = num_envs
        self.dt = dt
        self.apply_calls = 0

    def _resolve_wrench_bodies(self, body_names):
        self._union_names = list(body_names)
        return len(body_names)

    def _apply_external_wrenches(self, forces, torques):
        self.apply_calls += 1


# Borrow the real scheduler + reference-scale implementation.
for _name in (
    "_init_wrench_randomization",
    "_schedule_wrench",
    "_reset_wrench_class",
    "_summed_wrench_buffers",
    "_update_wrench_randomization",
    "_reset_wrench_randomization",
    "set_wrench_reference_scale_from_reference",
    "_perturb_gain_multiplier",
    # PM_DR_* env-gated override helpers (pre-existing borrow gap: the
    # harness broke when _summed_wrench_buffers grew these calls).
    "_wrench_dr_env_mask",
    "_wrench_dr_force_scale",
):
    setattr(_Host, _name, getattr(Simulator, _name))
_Host._sample_wrench_vectors = staticmethod(Simulator._sample_wrench_vectors)


def _bag_cfg(**kw):
    """A persistent downward wrist payload with reference modulation ON."""
    defaults = dict(
        force_magnitude_range=(111.0, 111.0),  # ~25 lb per wrist
        duration_range=(1.0, 1.0),
        interval_range=(1.0, 1.0),
        body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
        direction_mode="downward",
        reference_distance_modulation=True,
        distance_ref_body="torso_link",
        distance_wrist_bodies=["left_wrist_yaw_link", "right_wrist_yaw_link"],
        distance_near_m=0.25,
        distance_far_m=0.55,
        distance_far_scale=0.20,
    )
    defaults.update(kw)
    return WrenchDomainRandomizationConfig(**defaults)


def _ref_pose(num_envs, wrist_offset):
    """Reference body positions [num_envs, num_bodies, 3] with both wrists placed
    ``wrist_offset`` (a length-3 vector, meters) from the chest at the origin."""
    pos = torch.zeros(num_envs, len(BODY_NAMES), 3)
    off = torch.as_tensor(wrist_offset, dtype=torch.float32)
    pos[:, 1, :] = off   # left wrist
    pos[:, 2, :] = off   # right wrist
    return pos


def _load_both_wrists(host, F=111.0):
    """Deterministically set a pure -z payload of magnitude F on both wrist cols,
    bypassing the stochastic sampler so the test asserts the SCALE exactly."""
    sched = host._wrench_scheds[0]
    sched["forces"][:] = 0.0
    for name in ("left_wrist_yaw_link", "right_wrist_yaw_link"):
        col = host._wrench_union_names.index(name)
        sched["forces"][:, col, 2] = -F
    return sched


def test_near_chest_reference_full_load():
    """Reference wrist near the chest (d <= near) -> scale ~1.0 -> force ~F_max."""
    host = _Host(wrench_cfg=_bag_cfg())
    host._init_wrench_randomization()
    F = 111.0
    _load_both_wrists(host, F)
    # d = 0.10 m (< near 0.25) -> full scale.
    ref_pos = _ref_pose(host.num_envs, [0.10, 0.0, 0.0])
    host.set_wrench_reference_scale_from_reference(ref_pos, BODY_NAMES)

    lcol = host._wrench_union_names.index("left_wrist_yaw_link")
    assert torch.allclose(
        host._wrench_ref_scale[:, lcol], torch.ones(host.num_envs)
    ), "near-chest reference -> scale 1.0"

    f, _t = host._summed_wrench_buffers()
    applied = f[:, lcol].norm(dim=-1)
    torch.testing.assert_close(applied, torch.full((host.num_envs,), F), atol=1e-4, rtol=0)


def test_outstretched_reference_floor_load():
    """Reference wrist outstretched (d >= far) -> scale ~far_scale -> force ~F*far_scale."""
    cfg = _bag_cfg(distance_far_scale=0.20)
    host = _Host(wrench_cfg=cfg)
    host._init_wrench_randomization()
    F = 111.0
    _load_both_wrists(host, F)
    # d = 0.70 m (> far 0.55) -> floor scale 0.20.
    ref_pos = _ref_pose(host.num_envs, [0.70, 0.0, 0.0])
    host.set_wrench_reference_scale_from_reference(ref_pos, BODY_NAMES)

    lcol = host._wrench_union_names.index("left_wrist_yaw_link")
    torch.testing.assert_close(
        host._wrench_ref_scale[:, lcol], torch.full((host.num_envs,), 0.20),
        atol=1e-6, rtol=0,
    )
    f, _t = host._summed_wrench_buffers()
    applied = f[:, lcol].norm(dim=-1)
    torch.testing.assert_close(
        applied, torch.full((host.num_envs,), F * 0.20), atol=1e-3, rtol=0
    )


def test_midrange_reference_linear_interpolation():
    """Between near and far the scale interpolates linearly (sanity of the ramp)."""
    host = _Host(wrench_cfg=_bag_cfg())
    host._init_wrench_randomization()
    _load_both_wrists(host, 111.0)
    # d = 0.40 m, midway between near 0.25 and far 0.55 -> halfway between 1.0 and 0.2.
    ref_pos = _ref_pose(host.num_envs, [0.40, 0.0, 0.0])
    host.set_wrench_reference_scale_from_reference(ref_pos, BODY_NAMES)
    lcol = host._wrench_union_names.index("left_wrist_yaw_link")
    # Raw clamped-linear falloff: (far - d)/(far - near) = (0.55-0.40)/0.30 = 0.5,
    # which is above the far_scale floor (0.20) so it passes through unclamped.
    expected = (0.55 - 0.40) / (0.55 - 0.25)  # = 0.5
    torch.testing.assert_close(
        host._wrench_ref_scale[:, lcol], torch.full((host.num_envs,), expected),
        atol=1e-6, rtol=0,
    )


def test_anti_cheat_actual_pose_cannot_change_load():
    """THE KEY TEST. Hold the REFERENCE fixed; vary the actual/robot pose across
    a wide range. The applied wrench is IDENTICAL every time -> the policy cannot
    game the load by moving. Also assert the pure scale function's signature
    literally cannot accept a live/actual pose."""
    # The pure function accepts ONLY reference positions + scalar knobs.
    params = list(inspect.signature(reference_wrench_scale).parameters)
    assert params == [
        "ref_wrist_pos", "ref_chest_pos", "near_m", "far_m", "far_scale",
    ], "pure scale fn takes reference positions only (no actual/live pose)"
    assert not any(
        tok in p for p in params for tok in ("actual", "robot", "live", "current")
    ), "no parameter references the live robot pose"

    host = _Host(wrench_cfg=_bag_cfg())  # posture_gate OFF (default)
    host._init_wrench_randomization()
    F = 111.0
    _load_both_wrists(host, F)

    # FIXED reference at a mid distance so the scale is a non-trivial value.
    ref_pos = _ref_pose(host.num_envs, [0.40, 0.05, -0.10])
    lcol = host._wrench_union_names.index("left_wrist_yaw_link")

    baseline = None
    torch.manual_seed(0)
    for _ in range(50):
        # Arbitrary, wildly-varying "actual" robot pose (would let a naive
        # live-distance scheme shed load) -- must have ZERO effect here.
        actual = torch.randn(host.num_envs, len(BODY_NAMES), 3) * 5.0
        host.set_wrench_reference_scale_from_reference(ref_pos, BODY_NAMES, actual)
        f, _t = host._summed_wrench_buffers()
        applied = f[:, lcol].clone()
        if baseline is None:
            baseline = applied
        else:
            torch.testing.assert_close(
                applied, baseline, atol=0.0, rtol=0.0,
            )  # byte-for-byte identical regardless of live pose

    # And the value equals the reference-only computation.
    d = torch.tensor(0.40 ** 2 + 0.05 ** 2 + 0.10 ** 2).sqrt()
    expected_scale = ((0.55 - d) / (0.55 - 0.25)).clamp(0.20, 1.0)
    torch.testing.assert_close(
        baseline.norm(dim=-1), torch.full((host.num_envs,), float(F * expected_scale)),
        atol=1e-3, rtol=0,
    )


def test_posture_gate_is_opt_in_secondary_hardening():
    """posture_gate (opt-in) DOES consume live pose: with it ON, a wrist braced
    far from the reference wrist attenuates the load; with it OFF (default) live
    pose is ignored (covered by the anti-cheat test). Confirms the secondary
    hardening is real yet off by default."""
    cfg = _bag_cfg(posture_gate=True, posture_gate_tol_m=0.15)
    host = _Host(wrench_cfg=cfg)
    host._init_wrench_randomization()
    F = 111.0
    _load_both_wrists(host, F)
    ref_pos = _ref_pose(host.num_envs, [0.30, 0.0, 0.0])  # mid distance
    lcol = host._wrench_union_names.index("left_wrist_yaw_link")

    # Actual wrist ON the reference -> gate 1.0 (only reference scale applies).
    actual_on = ref_pos.clone()
    host.set_wrench_reference_scale_from_reference(ref_pos, BODY_NAMES, actual_on)
    on_ref = host._wrench_ref_scale[:, lcol].clone()

    # Actual wrist braced 0.30 m away (>= 2*tol) -> gate 0.0 -> load fully cut.
    actual_far = ref_pos.clone()
    actual_far[:, 1, 0] += 0.30
    host.set_wrench_reference_scale_from_reference(ref_pos, BODY_NAMES, actual_far)
    braced = host._wrench_ref_scale[:, lcol].clone()

    assert bool((on_ref > 0).all())
    torch.testing.assert_close(braced, torch.zeros(host.num_envs), atol=1e-6, rtol=0)


def test_backward_compat_modulation_off_leaves_forces_identical():
    """reference_distance_modulation=False -> buffer all ones -> forces are the
    exact pre-feature summed forces (setter is a no-op even if called)."""
    cfg = _bag_cfg(reference_distance_modulation=False)
    host = _Host(wrench_cfg=cfg)
    host._init_wrench_randomization()
    F = 111.0
    sched = _load_both_wrists(host, F)

    # Buffer initialized to ones.
    assert torch.equal(
        host._wrench_ref_scale, torch.ones_like(host._wrench_ref_scale)
    )

    # Even if the env calls the setter, an unmodulated class leaves it ones.
    ref_pos = _ref_pose(host.num_envs, [0.70, 0.0, 0.0])  # would be floor if ON
    host.set_wrench_reference_scale_from_reference(ref_pos, BODY_NAMES)
    assert torch.equal(
        host._wrench_ref_scale, torch.ones_like(host._wrench_ref_scale)
    ), "modulation OFF -> buffer stays ones"

    # Summed force equals the raw scheduler force (identity multiply by ones).
    f, _t = host._summed_wrench_buffers()
    torch.testing.assert_close(f, sched["forces"], atol=0.0, rtol=0.0)


def test_pure_function_clamps_and_shapes():
    """reference_wrench_scale clamps to [far_scale, 1.0] and preserves batch shape."""
    near, far, floor = 0.25, 0.55, 0.20
    chest = torch.zeros(7, 3)
    # Distances 0.0 .. 1.0 along x.
    dists = torch.linspace(0.0, 1.0, 7).unsqueeze(-1)
    wrist = torch.cat([dists, torch.zeros(7, 2)], dim=-1)
    s = reference_wrench_scale(wrist, chest, near, far, floor)
    assert s.shape == (7,)
    assert float(s.min()) >= floor - 1e-7
    assert float(s.max()) <= 1.0 + 1e-7
    # Monotonically non-increasing with distance.
    assert bool((s[1:] <= s[:-1] + 1e-7).all())
