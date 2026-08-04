# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit smokes for GAIN-DR (actuator PD-gain scale domain randomization).

Covers (CPU-only):
- ActuatorGainDomainRandomizationConfig validation (DOF selection XOR, range
  sanity, defaults).
- DomainRandomizationConfig carries the new optional actuator_gain slot
  (None default = off, pre-field pickles safe via getattr).
- Simulator._process_actuator_gain_domain_randomization sampling: shapes,
  bounds, per-env spread, optional effort-limit samples, DOF index match.
- Apply-path composition logic (replicated math): stiffness/damping scales
  compose multiplicatively on the nominal per-DOF gains; untouched DOFs keep
  default gains; envs differ from each other.
"""

from types import SimpleNamespace

import pytest
import torch

from protomotions.simulator.base_simulator.config import (
    ActuatorGainDomainRandomizationConfig,
    DomainRandomizationConfig,
)
from protomotions.simulator.base_simulator.simulator import Simulator

DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
]
NUM_ENVS = 512


def _mock_sim():
    return SimpleNamespace(
        num_envs=NUM_ENVS,
        robot_config=SimpleNamespace(
            kinematic_info=SimpleNamespace(dof_names=DOF_NAMES)
        ),
    )


def _process(cfg):
    return Simulator._process_actuator_gain_domain_randomization(_mock_sim(), cfg)


def test_config_requires_dof_selection():
    with pytest.raises(ValueError):
        ActuatorGainDomainRandomizationConfig()
    with pytest.raises(ValueError):
        ActuatorGainDomainRandomizationConfig(
            dof_names=["left_knee_joint"], dof_indices=[1]
        )


def test_config_rejects_bad_ranges():
    with pytest.raises(ValueError):
        ActuatorGainDomainRandomizationConfig(
            dof_names=[".*"], stiffness_scale_range=(1.3, 0.7)
        )
    with pytest.raises(ValueError):
        ActuatorGainDomainRandomizationConfig(
            dof_names=[".*"], damping_scale_range=(1.3, 0.7)
        )
    with pytest.raises(ValueError):
        ActuatorGainDomainRandomizationConfig(
            dof_names=[".*"], effort_limit_scale_range=(0.0, 1.05)
        )


def test_config_defaults_and_dr_slot():
    cfg = ActuatorGainDomainRandomizationConfig(dof_names=[".*"])
    assert cfg.stiffness_scale_range == (0.7, 1.3)
    assert cfg.damping_scale_range == (0.7, 1.3)
    assert cfg.effort_limit_scale_range is None
    dr = DomainRandomizationConfig()
    assert getattr(dr, "actuator_gain", None) is None  # off by default
    dr = DomainRandomizationConfig(actuator_gain=cfg)
    assert dr.actuator_gain is cfg


def test_process_samples_shapes_bounds_and_spread():
    cfg = ActuatorGainDomainRandomizationConfig(
        dof_names=[".*"],
        stiffness_scale_range=(0.7, 1.3),
        damping_scale_range=(0.8, 1.2),
    )
    out = _process(cfg)
    assert out["dof_indices"] == list(range(len(DOF_NAMES)))

    stiffness_scales = out["stiffness_scales"]
    assert stiffness_scales.shape == (NUM_ENVS, len(DOF_NAMES))
    assert (stiffness_scales >= 0.7).all() and (stiffness_scales <= 1.3).all()
    assert stiffness_scales.std() > 0.02
    assert abs(stiffness_scales.mean().item() - 1.0) < 0.03

    damping_scales = out["damping_scales"]
    assert damping_scales.shape == (NUM_ENVS, len(DOF_NAMES))
    assert (damping_scales >= 0.8).all() and (damping_scales <= 1.2).all()
    assert damping_scales.std() > 0.02

    # Envs differ from each other (not a broadcast collapse).
    assert not torch.allclose(stiffness_scales[0], stiffness_scales[1])
    assert out["effort_limit_scales"] is None


def test_process_samples_effort_limit_when_configured():
    cfg = ActuatorGainDomainRandomizationConfig(
        dof_names=["left_knee_joint", "right_knee_joint"],
        stiffness_scale_range=(0.7, 1.3),
        damping_scale_range=(0.7, 1.3),
        effort_limit_scale_range=(0.9, 1.1),
    )
    out = _process(cfg)
    assert out["dof_indices"] == [
        DOF_NAMES.index("left_knee_joint"),
        DOF_NAMES.index("right_knee_joint"),
    ]
    el = out["effort_limit_scales"]
    assert el.shape == (NUM_ENVS, 2)
    assert (el >= 0.9).all() and (el <= 1.1).all()


def test_apply_composition_math():
    """Replicates the isaaclab apply-path composition on CPU tensors."""
    cfg = ActuatorGainDomainRandomizationConfig(
        dof_names=["left_knee_joint", "right_knee_joint"],
        stiffness_scale_range=(0.7, 1.3),
        damping_scale_range=(0.7, 1.3),
    )
    out = _process(cfg)
    default_stiffness = torch.tensor([100.0, 80.0, 40.0, 100.0, 80.0, 40.0]).repeat(
        NUM_ENVS, 1
    )
    default_damping = torch.tensor([5.0, 4.0, 2.0, 5.0, 4.0, 2.0]).repeat(NUM_ENVS, 1)

    new_stiffness = default_stiffness.clone()
    new_damping = default_damping.clone()
    # identity ordering (find_joints preserve_order on identical names)
    new_stiffness[:, out["dof_indices"]] *= out["stiffness_scales"]
    new_damping[:, out["dof_indices"]] *= out["damping_scales"]

    knee = DOF_NAMES.index("left_knee_joint")
    ratio = new_stiffness[:, knee] / default_stiffness[:, knee]
    assert (ratio >= 0.7 - 1e-6).all() and (ratio <= 1.3 + 1e-6).all()
    assert torch.allclose(ratio, out["stiffness_scales"][:, 0], atol=1e-6)

    # Untouched DOFs (not in dof_indices) keep the default gains exactly.
    hip = DOF_NAMES.index("left_hip_pitch_joint")
    assert torch.equal(new_stiffness[:, hip], default_stiffness[:, hip])
    assert torch.equal(new_damping[:, hip], default_damping[:, hip])

    # readback check: readback == base * scale for the randomized DOFs
    damping_ratio = new_damping[:, knee] / default_damping[:, knee]
    assert torch.allclose(damping_ratio, out["damping_scales"][:, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# MARIONETTE mode (2026-08-04): widened gain range + perturbation coupling
# ---------------------------------------------------------------------------


def test_process_respects_widened_marionette_bounds():
    """Sampler honors the intended marionette range (0.2, 1.3) exactly."""
    cfg = ActuatorGainDomainRandomizationConfig(
        dof_names=[".*"],
        stiffness_scale_range=(0.2, 1.3),
        damping_scale_range=(0.2, 1.3),
    )
    out = _process(cfg)
    for key in ("stiffness_scales", "damping_scales"):
        s = out[key]
        assert (s >= 0.2).all() and (s <= 1.3).all()
        # The widened band is actually explored, not collapsed to the old one.
        assert s.min() < 0.5
        assert s.std() > 0.05


def test_env_gain_scale_is_geometric_mean_of_stiffness():
    cfg = ActuatorGainDomainRandomizationConfig(
        dof_names=[".*"], stiffness_scale_range=(0.2, 1.3)
    )
    out = _process(cfg)
    g = out["env_gain_scale"]
    assert g.shape == (NUM_ENVS,)
    expected = torch.exp(torch.log(out["stiffness_scales"]).mean(dim=1))
    assert torch.allclose(g, expected, atol=1e-6)
    assert (g >= 0.2 - 1e-6).all() and (g <= 1.3 + 1e-6).all()
    # No NaN/inf anywhere near the low-stiffness (0.2) end of the band.
    assert torch.isfinite(g).all()


def _coupling_host(env_gain_scale=None, num_envs=4):
    host = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=num_envs,
        _domain_randomization=(
            None
            if env_gain_scale is None
            else {"actuator_gain": {"env_gain_scale": env_gain_scale}}
        ),
    )
    host._perturb_gain_multiplier = (
        Simulator._perturb_gain_multiplier.__get__(host)
    )
    return host


def test_coupling_multiplier_math(monkeypatch):
    g = torch.tensor([0.3, 0.2, 1.0, 1.3])

    # exp unset -> None (full no-op, byte-identical path).
    monkeypatch.delenv("PM_PERTURB_GAIN_EXP", raising=False)
    monkeypatch.delenv("PM_PERTURB_SCALE_MIN", raising=False)
    assert _coupling_host(g)._perturb_gain_multiplier() is None

    # exp explicitly 0 -> still None (coupling OFF).
    monkeypatch.setenv("PM_PERTURB_GAIN_EXP", "0")
    assert _coupling_host(g)._perturb_gain_multiplier() is None

    # exp=1 (linear): g=0.3 -> 0.3; default min=0.25 clamps g=0.2 -> 0.25;
    # cap at 1.0 clamps g=1.3 -> 1.0.
    monkeypatch.setenv("PM_PERTURB_GAIN_EXP", "1.0")
    mult = _coupling_host(g)._perturb_gain_multiplier()
    assert torch.allclose(mult, torch.tensor([0.3, 0.25, 1.0, 1.0]), atol=1e-6)

    # explicit min clamp
    monkeypatch.setenv("PM_PERTURB_SCALE_MIN", "0.5")
    mult = _coupling_host(g)._perturb_gain_multiplier()
    assert torch.allclose(mult, torch.tensor([0.5, 0.5, 1.0, 1.0]), atol=1e-6)

    # sub-linear exponent
    monkeypatch.setenv("PM_PERTURB_SCALE_MIN", "0.0001")
    monkeypatch.setenv("PM_PERTURB_GAIN_EXP", "2.0")
    mult = _coupling_host(g)._perturb_gain_multiplier()
    assert torch.allclose(mult, torch.tensor([0.09, 0.04, 1.0, 1.0]), atol=1e-6)

    # exp set but gain DR inactive -> None + loud WARNING (no crash).
    monkeypatch.setenv("PM_PERTURB_GAIN_EXP", "1.0")
    assert _coupling_host(None)._perturb_gain_multiplier() is None

    # multiplier is cached: second call returns the identical tensor object.
    host = _coupling_host(g)
    assert host._perturb_gain_multiplier() is host._perturb_gain_multiplier()


def test_push_velocities_scaled_by_coupling(monkeypatch):
    """_apply_push_if_due multiplies due-env impulses by the per-env factor."""
    num_envs = 4
    g = torch.tensor([0.3, 0.2, 1.0, 1.3])

    def make_host():
        host = SimpleNamespace(
            device=torch.device("cpu"),
            num_envs=num_envs,
            _domain_randomization={"actuator_gain": {"env_gain_scale": g}},
            _push_enabled=True,
            _simulation_time=torch.ones(num_envs),
            _push_next_time=torch.zeros(num_envs),  # all due
            _push_interval_range=(1.0, 1.0),
            _push_max_lin_vel=torch.tensor([1.2, 1.2, 1.2]),
            _push_max_ang_vel=torch.tensor([0.0, 0.0, 0.0]),
            _push_grace_steps=0,
            config=SimpleNamespace(
                domain_randomization=SimpleNamespace(
                    push=SimpleNamespace(magnitude_scale=1.0)
                )
            ),
            captured={},
        )
        host._apply_push_if_due = Simulator._apply_push_if_due.__get__(host)
        host._schedule_push = Simulator._schedule_push.__get__(host)
        host._perturb_gain_multiplier = (
            Simulator._perturb_gain_multiplier.__get__(host)
        )
        host._apply_root_velocity_impulse = (
            lambda lin, ang, ids: host.captured.update(
                lin=lin.clone(), ang=ang.clone(), ids=ids.clone()
            )
        )
        return host

    # Coupled run vs uncoupled run with the same RNG seed: exact ratio check.
    monkeypatch.setenv("PM_PERTURB_GAIN_EXP", "1.0")
    monkeypatch.setenv("PM_PERTURB_SCALE_MIN", "0.25")
    torch.manual_seed(1234)
    coupled = make_host()
    coupled._apply_push_if_due()

    monkeypatch.delenv("PM_PERTURB_GAIN_EXP", raising=False)
    torch.manual_seed(1234)
    plain = make_host()
    plain._apply_push_if_due()

    expected_mult = torch.tensor([0.3, 0.25, 1.0, 1.0])
    assert torch.allclose(
        coupled.captured["lin"], plain.captured["lin"] * expected_mult[:, None],
        atol=1e-6,
    )
    # Unset env: due-env velocities can reach the full configured magnitude.
    assert plain.captured["lin"].abs().max() <= 1.2 + 1e-6


def test_summed_wrench_buffers_scaled_by_coupling(monkeypatch):
    """Wrench + sustained-burst forces/torques all ride the summed buffers,
    so one per-env multiply there covers every wrench class."""
    num_envs = 3
    g = torch.tensor([0.2, 0.5, 1.0])
    forces = torch.ones(num_envs, 2, 3) * 10.0  # two body columns
    torques = torch.ones(num_envs, 2, 3) * 4.0
    sustained_f = torch.ones(num_envs, 2, 3) * 1.0  # second class (bag)

    def make_host():
        host = SimpleNamespace(
            device=torch.device("cpu"),
            num_envs=num_envs,
            _domain_randomization={"actuator_gain": {"env_gain_scale": g}},
            _wrench_scheds=[
                {"forces": forces.clone(), "torques": torques.clone()},
                {"forces": sustained_f.clone(), "torques": torch.zeros_like(torques)},
            ],
        )
        for name in (
            "_summed_wrench_buffers",
            "_wrench_dr_force_scale",
            "_wrench_dr_env_mask",
            "_perturb_gain_multiplier",
        ):
            setattr(host, name, getattr(Simulator, name).__get__(host))
        return host

    # Unset -> byte-identical to the plain sum.
    for var in ("PM_PERTURB_GAIN_EXP", "PM_PERTURB_SCALE_MIN",
                "PM_DR_FORCE_SCALE", "PM_DR_ENV_FRACTION"):
        monkeypatch.delenv(var, raising=False)
    f0, t0 = make_host()._summed_wrench_buffers()
    assert torch.equal(f0, forces + sustained_f)
    assert torch.equal(t0, torques)

    # Linear coupling: env0 clamps at min 0.25, env1 0.5, env2 1.0.
    monkeypatch.setenv("PM_PERTURB_GAIN_EXP", "1.0")
    f1, t1 = make_host()._summed_wrench_buffers()
    expected = torch.tensor([0.25, 0.5, 1.0])[:, None, None]
    assert torch.allclose(f1, (forces + sustained_f) * expected, atol=1e-6)
    assert torch.allclose(t1, torques * expected, atol=1e-6)
