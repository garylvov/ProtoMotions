# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit smokes for the ramped persistent-force DR (Track D 2026-07-10).

Exercises the wrench scheduler in isolation (no backend): a harness object
borrows the scheduler methods from the abstract ``Simulator`` and stubs the
two backend hooks (``_resolve_wrench_bodies`` / ``_apply_external_wrenches``).

Covers: ramp profile shape + continuity (ease-in -> hold at sampled magnitude
-> ease-out -> zero), previous step-profile behavior with zero ramps,
per-env randomization, direction modes, independent per-body scheduling
(asymmetric/simultaneous events), and reset behavior.
"""

from types import SimpleNamespace

import pytest
import torch

from protomotions.simulator.base_simulator.config import (
    WrenchDomainRandomizationConfig,
)
from protomotions.simulator.base_simulator.simulator import Simulator


class _Host:
    """Minimal harness carrying only the attributes the scheduler touches."""

    def __init__(self, wrench_cfg=None, sustained_cfg=None, num_envs=4, dt=0.02):
        self.config = SimpleNamespace(
            domain_randomization=SimpleNamespace(
                wrench=wrench_cfg, sustained_wrench=sustained_cfg
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


# Borrow the real scheduler implementation.
for _name in (
    "_init_wrench_randomization",
    "_schedule_wrench",
    "_reset_wrench_class",
    "_summed_wrench_buffers",
    "_update_wrench_randomization",
    "_reset_wrench_randomization",
):
    setattr(_Host, _name, getattr(Simulator, _name))
_Host._sample_wrench_vectors = staticmethod(Simulator._sample_wrench_vectors)


def _ramped_cfg(**kw):
    defaults = dict(
        force_magnitude_range=(50.0, 50.0),
        duration_range=(1.0, 1.0),
        interval_range=(0.5, 0.5),
        body_names=["torso_link"],
        ramp_in_range=(0.4, 0.4),
        ramp_out_range=(0.4, 0.4),
        direction_mode="horizontal",
    )
    defaults.update(kw)
    return WrenchDomainRandomizationConfig(**defaults)


def _step_trace(host, sched, steps):
    """Advance the scheduler, recording per-step force magnitude of env 0."""
    mags = []
    for _ in range(steps):
        host._update_wrench_randomization()
        mags.append(float(sched["forces"][0].norm()))
    return mags


def test_ramp_profile_shape_and_continuity():
    torch.manual_seed(0)
    host = _Host(wrench_cfg=_ramped_cfg())
    host._init_wrench_randomization()
    sched = host._wrench_scheds[0]
    dt = host.dt

    # interval 0.5s, ramp 0.4s, hold 1.0s, ramp 0.4s -> total event 1.8s.
    mags = _step_trace(host, sched, int(3.0 / dt))
    peak = max(mags)
    assert peak == pytest.approx(50.0, abs=1e-3), "hold reaches sampled magnitude"

    # Continuity: cosine ease bounds the per-step increment.
    deltas = [abs(b - a) for a, b in zip(mags, mags[1:])]
    max_slope_per_step = 50.0 * (torch.pi / 2) * (dt / 0.4)  # cosine max slope
    assert max(deltas) <= max_slope_per_step * 1.05, "no step discontinuity"

    # Shape of the FIRST event (the scheduler starts another one after the
    # 0.5 s interval): starts ~0, sustained hold, falls back to exactly 0.
    active = [i for i, m in enumerate(mags) if m > 1e-6]
    first = active[0]
    last = first
    while last + 1 < len(mags) and mags[last + 1] > 1e-6:
        last += 1
    event = mags[first:last + 1]
    assert mags[first] < 10.0, "ease-in starts near zero"
    hold_steps = sum(1 for m in event if m > 49.999)
    assert hold_steps >= int(0.9 / dt), "sustained hold phase present"
    assert last + 1 < len(mags) and mags[last + 1] == 0.0, "returns to zero"
    # Event duration ~ ramp_in + hold + ramp_out.
    assert (last - first + 1) * dt == pytest.approx(1.8, abs=3 * dt)


def test_zero_ramps_reproduce_step_profile():
    torch.manual_seed(0)
    cfg = _ramped_cfg(ramp_in_range=(0.0, 0.0), ramp_out_range=(0.0, 0.0),
                      direction_mode="uniform")
    host = _Host(wrench_cfg=cfg)
    host._init_wrench_randomization()
    sched = host._wrench_scheds[0]
    assert not sched["ramped"]

    mags = _step_trace(host, sched, int(2.5 / host.dt))
    nonzero = [m for m in mags if m > 0]
    # Step profile: instantly at full magnitude, no intermediate values.
    assert all(m == pytest.approx(50.0, abs=1e-4) for m in nonzero)
    # First contiguous event lasts exactly duration_range (1.0 s).
    active = [i for i, m in enumerate(mags) if m > 0]
    first = active[0]
    last = first
    while last + 1 < len(mags) and mags[last + 1] > 0:
        last += 1
    assert (last - first + 1) == pytest.approx(int(1.0 / host.dt), abs=2)


def test_per_env_and_per_event_randomization():
    torch.manual_seed(1)
    cfg = _ramped_cfg(
        force_magnitude_range=(10.0, 90.0),
        duration_range=(0.5, 4.0),
        ramp_in_range=(0.2, 1.0),
        ramp_out_range=(0.2, 1.0),
    )
    host = _Host(wrench_cfg=cfg, num_envs=16)
    host._init_wrench_randomization()
    sched = host._wrench_scheds[0]
    for _ in range(int(1.6 / host.dt)):
        host._update_wrench_randomization()
    assert bool(sched["active"].all())
    tgt = sched["target_forces"][:, 0]  # single-body class -> column 0
    mags = tgt.norm(dim=-1)
    assert float(mags.std()) > 5.0, "magnitudes vary per env"
    dirs = tgt / mags.clamp_min(1e-9).unsqueeze(-1)
    assert float(dirs.std(dim=0).norm()) > 0.1, "directions vary per env"
    assert float(sched["ramp_in"].std()) > 0.05, "ramp-in durations vary"
    assert float((sched["end_time"] - sched["start_time"]).std()) > 0.1


def test_direction_modes():
    torch.manual_seed(2)
    down = Simulator._sample_wrench_vectors(
        1000, (98.0, 98.0), torch.device("cpu"), mode="downward")
    # Payload spec: z locked to the FULL sampled magnitude (constant -z),
    # horizontal at 10-18% of magnitude.
    torch.testing.assert_close(down[:, 2], torch.full((1000,), -98.0))
    h = down[:, :2].norm(dim=-1)
    assert bool((h >= 98.0 * 0.10 - 1e-4).all()), "horizontal >= 10%"
    assert bool((h <= 98.0 * 0.18 + 1e-4).all()), "horizontal <= 18%"

    horiz = Simulator._sample_wrench_vectors(
        1000, (50.0, 50.0), torch.device("cpu"), mode="horizontal")
    h = horiz / horiz.norm(dim=-1, keepdim=True)
    assert float(h[:, 2].abs().mean()) < 0.30, "mostly horizontal"

    uni = Simulator._sample_wrench_vectors(
        1000, (50.0, 50.0), torch.device("cpu"), mode="uniform")
    u = uni / uni.norm(dim=-1, keepdim=True)
    assert 0.4 < float(u[:, 2].abs().mean()) < 0.6, "uniform sphere baseline"


def test_independent_bodies_schedule_per_body():
    torch.manual_seed(3)
    cfg = _ramped_cfg(
        body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
        force_magnitude_range=(10.0, 98.0),
        duration_range=(0.5, 2.0),
        interval_range=(0.2, 1.5),
        direction_mode="downward",
        independent_bodies=True,
    )
    host = _Host(wrench_cfg=cfg, num_envs=8)
    host._init_wrench_randomization()
    # One scheduler entry PER BODY.
    assert len(host._wrench_scheds) == 2
    assert all(len(s["cols"]) == 1 for s in host._wrench_scheds)

    saw_one, saw_both = False, False
    for _ in range(int(8.0 / host.dt)):
        host._update_wrench_randomization()
        f, _t = host._summed_wrench_buffers()
        loaded = (f[:, :, :].norm(dim=-1) > 1.0)  # [envs, union_bodies]
        n_loaded = loaded.sum(dim=-1)
        if bool((n_loaded == 1).any()):
            saw_one = True
        if bool((n_loaded == 2).any()):
            saw_both = True
        if saw_one and saw_both:
            break
    assert saw_one, "asymmetric single-hand load occurs"
    assert saw_both, "simultaneous both-hands load occurs"


def test_reset_clears_active_forces_and_state():
    torch.manual_seed(4)
    host = _Host(wrench_cfg=_ramped_cfg())
    host._init_wrench_randomization()
    sched = host._wrench_scheds[0]
    for _ in range(int(1.0 / host.dt)):
        host._update_wrench_randomization()
    assert bool(sched["active"].any())
    assert float(sched["forces"].abs().sum()) > 0.0

    env_ids = torch.arange(host.num_envs)
    host._reset_wrench_randomization(env_ids)
    assert not bool(sched["active"].any())
    assert float(sched["forces"].abs().sum()) == 0.0
    assert float(sched["target_forces"].abs().sum()) == 0.0
    assert float(sched["ramp_in"].abs().sum()) == 0.0

    # After the reset the scheduler keeps producing events.
    for _ in range(int(1.0 / host.dt)):
        host._update_wrench_randomization()
    assert bool(sched["active"].any())


def _seeded_trace(seed, num_envs=8, steps=None, **cfg_kw):
    """Full applied-force trace [steps, envs, bodies, 3] under a fixed seed."""
    torch.manual_seed(seed)
    cfg = _ramped_cfg(
        force_magnitude_range=(10.0, 90.0),
        duration_range=(0.5, 4.0),
        ramp_in_range=(0.2, 1.0),
        ramp_out_range=(0.2, 1.0),
        **cfg_kw,
    )
    host = _Host(wrench_cfg=cfg, num_envs=num_envs)
    host._init_wrench_randomization()
    steps = steps or int(3.0 / host.dt)
    frames = []
    for _ in range(steps):
        host._update_wrench_randomization()
        f, _t = host._summed_wrench_buffers()
        frames.append(f.clone())
    return torch.stack(frames)


def test_seeded_reproducibility():
    a = _seeded_trace(123)
    b = _seeded_trace(123)
    torch.testing.assert_close(a, b)
    c = _seeded_trace(321)
    assert not torch.allclose(a, c), "different seed gives a different schedule"
    assert float(a.abs().sum()) > 0.0, "traces are non-trivial"


def test_stage_scale_multiplies_magnitudes():
    """magnitude_scale is the DR-ladder stage knob: same sampling stream
    under the same seed, every applied force multiplied by the scale."""
    full = _seeded_trace(5, magnitude_scale=1.0)
    half = _seeded_trace(5, magnitude_scale=0.5)
    zero = _seeded_trace(5, magnitude_scale=0.0)
    torch.testing.assert_close(half, full * 0.5)
    assert float(zero.abs().max()) == 0.0, "scale 0 silences the family"
    assert float(full.abs().max()) > 0.0


def test_stage_scale_persistent_cohort_and_downward_bias():
    """Scale also applies to the reset-time persistent cohort, and the
    payload gravity-bias split survives scaling (z stays the dominant,
    locked -z share)."""
    torch.manual_seed(7)
    cfg = _ramped_cfg(
        force_magnitude_range=(98.0, 98.0),
        direction_mode="downward",
        persistent_fraction=1.0,
        magnitude_scale=0.5,
    )
    host = _Host(wrench_cfg=cfg, num_envs=32)
    host._init_wrench_randomization()
    sched = host._wrench_scheds[0]
    f = sched["forces"][:, 0]  # single-body class -> column 0
    torch.testing.assert_close(f[:, 2], torch.full((32,), -49.0))
    h = f[:, :2].norm(dim=-1)
    assert bool((h >= 49.0 * 0.10 - 1e-4).all())
    assert bool((h <= 49.0 * 0.18 + 1e-4).all())


def test_action_rate_grace_mask_wrench_phases():
    """Grace mask True during ramp-in + plateau of a flagged class, False
    during ease-out and idle; None when nothing is configured."""
    torch.manual_seed(5)
    cfg = _ramped_cfg(action_rate_grace=True)
    host = _Host(wrench_cfg=cfg, num_envs=2)
    host._init_wrench_randomization()
    host._push_grace_steps = 0  # no push source
    sched = host._wrench_scheds[0]

    saw_grace_in_event, saw_no_grace_ease_out = False, False
    for _ in range(int(3.0 / host.dt)):
        host._update_wrench_randomization()
        mask = Simulator.get_action_rate_grace_mask(host)
        assert mask is not None
        active = bool(sched["active"][0])
        if active:
            t_left = float(sched["end_time"][0] - sched["time"][0])
            in_ease_out = t_left < float(sched["ramp_out"][0])
            if not in_ease_out:
                assert bool(mask[0]), "grace ON during ramp-in/plateau"
                saw_grace_in_event = True
            else:
                assert not bool(mask[0]), "grace OFF during ease-out"
                saw_no_grace_ease_out = True
        else:
            assert not bool(mask[0]), "grace OFF while idle"
    assert saw_grace_in_event and saw_no_grace_ease_out

    # Unflagged class -> no grace source -> None.
    host2 = _Host(wrench_cfg=_ramped_cfg(), num_envs=2)
    host2._init_wrench_randomization()
    host2._push_grace_steps = 0
    assert Simulator.get_action_rate_grace_mask(host2) is None


def test_push_grace_countdown_and_reset():
    """Post-push grace countdown arms per env and clears on reset."""
    host = _Host(wrench_cfg=_ramped_cfg(action_rate_grace=False), num_envs=4)
    host._init_wrench_randomization()
    host._push_grace_steps = 60
    host._push_grace_left = torch.zeros(4, dtype=torch.long)
    host._push_grace_left[1] = 60  # env 1 just got pushed
    mask = Simulator.get_action_rate_grace_mask(host)
    assert mask is not None
    assert bool(mask[1]) and not bool(mask[0])
    # Reset clears the countdown for the reset env.
    Simulator._reset_wrench_randomization(host, torch.tensor([1]))
    mask = Simulator.get_action_rate_grace_mask(host)
    assert not bool(mask[1])
