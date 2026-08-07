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

import pickle
from types import SimpleNamespace

import pytest
import torch

from protomotions.simulator.base_simulator.config import (
    ActuatorGainDomainRandomizationConfig,
    DomainRandomizationConfig,
    H1_2_GAIN_DR_GROUP_PATTERNS,
    resolve_gain_dr_groups,
)
from protomotions.simulator.base_simulator.gain_dr_env_gates import (
    GAIN_DR_ENV_VARS,
    apply_gain_dr_env_overrides,
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

# The full H1-2 27-DOF actuated set, verbatim from the v56 run's
# "[gain-dr] enabled: dofs=[...]" log line (canonical_teacher_20260804_v56).
H1_2_DOF_NAMES = [
    "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
NUM_ENVS = 512


def _mock_sim(dof_names=None):
    return SimpleNamespace(
        num_envs=NUM_ENVS,
        robot_config=SimpleNamespace(
            kinematic_info=SimpleNamespace(dof_names=dof_names or DOF_NAMES)
        ),
    )


def _process(cfg, dof_names=None):
    return Simulator._process_actuator_gain_domain_randomization(
        _mock_sim(dof_names), cfg
    )


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


# ---------------------------------------------------------------------------
# PER-GROUP GAIN-DR (2026-08-04): stiff legs + compliant waist/arms
# ---------------------------------------------------------------------------

LEG_NAMES = [n for n in H1_2_DOF_NAMES if "hip" in n or "knee" in n or "ankle" in n]
ARM_NAMES = [
    n for n in H1_2_DOF_NAMES
    if "shoulder" in n or "elbow" in n or "wrist" in n
]


def _h1_2_group_cfg(**kwargs):
    kwargs.setdefault("dof_names", [".*"])
    return ActuatorGainDomainRandomizationConfig(**kwargs)


def _cols(out, name):
    """Column index of ``name`` inside the sample matrices."""
    return [
        i for i, idx in enumerate(out["dof_indices"])
        if H1_2_DOF_NAMES[idx] == name
    ][0]


def test_h1_2_group_partition_is_exhaustive_and_disjoint():
    groups = resolve_gain_dr_groups(H1_2_DOF_NAMES, H1_2_GAIN_DR_GROUP_PATTERNS)
    assert len(groups["legs"]) == 12
    assert len(groups["waist"]) == 1
    assert len(groups["arms"]) == 14
    assert sum(len(v) for v in groups.values()) == len(H1_2_DOF_NAMES)
    assert [H1_2_DOF_NAMES[c] for c in groups["waist"]] == ["torso_joint"]
    assert sorted(H1_2_DOF_NAMES[c] for c in groups["legs"]) == sorted(LEG_NAMES)
    assert sorted(H1_2_DOF_NAMES[c] for c in groups["arms"]) == sorted(ARM_NAMES)


def test_group_partition_fails_loud_on_unmatched_dof():
    with pytest.raises(ValueError, match="unmatched"):
        resolve_gain_dr_groups(
            H1_2_DOF_NAMES + ["head_yaw_joint"], H1_2_GAIN_DR_GROUP_PATTERNS
        )


def test_group_partition_fails_loud_on_double_matched_dof():
    bad = dict(H1_2_GAIN_DR_GROUP_PATTERNS)
    bad["arms"] = list(bad["arms"]) + [".*_knee_joint"]
    with pytest.raises(ValueError, match="double_matched"):
        resolve_gain_dr_groups(H1_2_DOF_NAMES, bad)


def test_per_group_ranges_respected_per_joint():
    """Legs nominal 0.7-1.3, waist+arms marionette 0.2-1.3."""
    cfg = _h1_2_group_cfg(
        stiffness_scale_range=(0.7, 1.3),
        damping_scale_range=(0.7, 1.3),
        group_stiffness_scale_ranges={
            "legs": (0.7, 1.3), "waist": (0.2, 1.3), "arms": (0.2, 1.3),
        },
        group_damping_scale_ranges={
            "legs": (0.7, 1.3), "waist": (0.2, 1.3), "arms": (0.2, 1.3),
        },
    )
    out = _process(cfg, H1_2_DOF_NAMES)
    k = out["stiffness_scales"]
    d = out["damping_scales"]
    for name in LEG_NAMES:
        c = _cols(out, name)
        assert k[:, c].min() >= 0.7 - 1e-6, name
        assert k[:, c].max() <= 1.3 + 1e-6, name
        assert d[:, c].min() >= 0.7 - 1e-6, name
    for name in ARM_NAMES + ["torso_joint"]:
        c = _cols(out, name)
        assert k[:, c].max() <= 1.3 + 1e-6, name
        # The widened band is actually explored on the soft groups.
        assert k[:, c].min() < 0.5, name
        assert d[:, c].min() < 0.5, name
    # group membership is reported back for downstream verification
    assert set(out["group_columns"]) == {"legs", "waist", "arms"}


def test_unset_group_ranges_are_byte_identical_to_global_behavior():
    """RULE-10: no per-group config => the legacy scalar sampling expression,
    same RNG stream, bit-for-bit."""
    cfg = ActuatorGainDomainRandomizationConfig(
        dof_names=[".*"],
        stiffness_scale_range=(0.2, 1.3),
        damping_scale_range=(0.2, 1.3),
    )
    torch.manual_seed(20260804)
    out = _process(cfg, H1_2_DOF_NAMES)

    torch.manual_seed(20260804)
    n = len(H1_2_DOF_NAMES)
    legacy_k = torch.rand(NUM_ENVS, n) * (1.3 - 0.2) + 0.2
    legacy_d = torch.rand(NUM_ENVS, n) * (1.3 - 0.2) + 0.2
    assert torch.equal(out["stiffness_scales"], legacy_k)
    assert torch.equal(out["damping_scales"], legacy_d)
    # ... and the aggregate keeps the all-DOF geometric mean.
    assert out["env_gain_scale_source"] == "all"
    assert torch.equal(
        out["env_gain_scale"], torch.exp(torch.log(legacy_k).mean(dim=1))
    )
    assert out["group_columns"] == {}


def test_per_group_ranges_equal_to_global_reproduce_global_sample():
    """Per-group mode with every band == the global band draws the same
    numbers (to float32 rounding: the per-column bound vectors round
    ``u*(hi-lo)`` one ulp differently from the scalar form, which is exactly
    why the OFF path keeps the literal scalar expression)."""
    common = dict(
        dof_names=[".*"],
        stiffness_scale_range=(0.2, 1.3),
        damping_scale_range=(0.2, 1.3),
    )
    torch.manual_seed(7)
    plain = _process(
        ActuatorGainDomainRandomizationConfig(**common), H1_2_DOF_NAMES
    )
    torch.manual_seed(7)
    grouped = _process(
        ActuatorGainDomainRandomizationConfig(
            **common,
            group_stiffness_scale_ranges={g: (0.2, 1.3) for g in
                                          ("legs", "waist", "arms")},
            group_damping_scale_ranges={g: (0.2, 1.3) for g in
                                        ("legs", "waist", "arms")},
            env_gain_scale_group="all",
        ),
        H1_2_DOF_NAMES,
    )
    assert torch.allclose(
        plain["stiffness_scales"], grouped["stiffness_scales"], atol=1e-6
    )
    assert torch.allclose(
        plain["damping_scales"], grouped["damping_scales"], atol=1e-6
    )
    assert torch.allclose(
        plain["env_gain_scale"], grouped["env_gain_scale"], atol=1e-6
    )


def test_constant_zeta_derives_damping_from_stiffness():
    cfg = _h1_2_group_cfg(
        stiffness_scale_range=(0.2, 1.3),
        damping_scale_range=(0.2, 1.3),
        constant_damping_ratio=True,
    )
    out = _process(cfg, H1_2_DOF_NAMES)
    k = out["stiffness_scales"]
    d = out["damping_scales"]
    assert torch.allclose(d, torch.sqrt(k), atol=1e-7)
    # zeta = d / (2*sqrt(k*m)) is INVARIANT under (k -> s*k, d -> sqrt(s)*d).
    k0, d0, m = 200.0, 5.0, 1.0
    zeta_nominal = d0 / (2.0 * (k0 * m) ** 0.5)
    zeta = (d * d0) / (2.0 * torch.sqrt(k * k0 * m))
    assert torch.allclose(zeta, torch.full_like(zeta, zeta_nominal), atol=1e-5)
    # Contrast: the DEFAULT (independent sample) does NOT hold zeta.
    plain = _process(
        _h1_2_group_cfg(
            stiffness_scale_range=(0.2, 1.3), damping_scale_range=(0.2, 1.3)
        ),
        H1_2_DOF_NAMES,
    )
    zeta_plain = (plain["damping_scales"] * d0) / (
        2.0 * torch.sqrt(plain["stiffness_scales"] * k0 * m)
    )
    assert zeta_plain.std() > 1e-3


def test_constant_zeta_default_off_is_byte_identical():
    cfg = _h1_2_group_cfg(
        stiffness_scale_range=(0.2, 1.3), damping_scale_range=(0.2, 1.3)
    )
    assert cfg.constant_damping_ratio is False
    torch.manual_seed(11)
    out = _process(cfg, H1_2_DOF_NAMES)
    torch.manual_seed(11)
    n = len(H1_2_DOF_NAMES)
    _ = torch.rand(NUM_ENVS, n) * (1.3 - 0.2) + 0.2
    legacy_d = torch.rand(NUM_ENVS, n) * (1.3 - 0.2) + 0.2
    assert torch.equal(out["damping_scales"], legacy_d)


def test_env_gain_scale_keys_off_legs_when_per_group_active():
    """A nominal-leg env must NOT get its pushes discounted for floppy arms."""
    cfg = _h1_2_group_cfg(
        stiffness_scale_range=(0.7, 1.3),
        group_stiffness_scale_ranges={
            "legs": (0.7, 1.3), "waist": (0.2, 0.3), "arms": (0.2, 0.3),
        },
        group_damping_scale_ranges={
            "legs": (0.7, 1.3), "waist": (0.2, 0.3), "arms": (0.2, 0.3),
        },
    )
    out = _process(cfg, H1_2_DOF_NAMES)
    assert out["env_gain_scale_source"] == "legs"  # AUTO resolution
    g = out["env_gain_scale"]
    leg_cols = out["group_columns"]["legs"]
    assert torch.allclose(
        g, torch.exp(torch.log(out["stiffness_scales"][:, leg_cols]).mean(dim=1)),
        atol=1e-6,
    )
    # Leg-keyed aggregate stays in the leg band; the all-DOF mean would have
    # been dragged far below it by the 0.2-0.3 upper body.
    assert g.min() >= 0.7 - 1e-6
    all_mean = torch.exp(torch.log(out["stiffness_scales"]).mean(dim=1))
    assert all_mean.max() < 0.7

    # Explicit 'all' restores the old aggregate.
    cfg.env_gain_scale_group = "all"
    out_all = _process(cfg, H1_2_DOF_NAMES)
    assert out_all["env_gain_scale_source"] == "all"
    assert out_all["env_gain_scale"].max() < 0.7


def test_env_gain_scale_group_validation():
    with pytest.raises(ValueError, match="env_gain_scale_group"):
        _h1_2_group_cfg(env_gain_scale_group="tail")
    with pytest.raises(ValueError, match="unknown group"):
        _h1_2_group_cfg(group_stiffness_scale_ranges={"tail": (0.2, 1.0)})
    with pytest.raises(ValueError, match="0 < min <= max"):
        _h1_2_group_cfg(group_stiffness_scale_ranges={"arms": (1.3, 0.2)})


# ---------------------------------------------------------------------------
# Env-var gate (shared by the fresh-build and resume re-apply paths)
# ---------------------------------------------------------------------------


def _clear_gain_env(monkeypatch):
    for var in GAIN_DR_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _gate(cfg, label="RESUME"):
    lines = []
    changed = apply_gain_dr_env_overrides(cfg, log_fn=lines.append, label=label)
    return changed, lines


def test_gate_is_hard_noop_when_no_knob_present(monkeypatch):
    _clear_gain_env(monkeypatch)
    cfg = _h1_2_group_cfg(
        stiffness_scale_range=(0.7, 1.3), damping_scale_range=(0.7, 1.3)
    )
    changed, lines = _gate(cfg)
    assert changed is False
    assert lines == []  # not one line logged
    assert cfg.stiffness_scale_range == (0.7, 1.3)
    assert cfg.group_stiffness_scale_ranges is None
    assert cfg.constant_damping_ratio is False
    assert cfg.env_gain_scale_group is None


def test_gate_global_only_matches_legacy_semantics(monkeypatch):
    _clear_gain_env(monkeypatch)
    monkeypatch.setenv("PM_GAIN_DR_LOW", "0.2")
    cfg = _h1_2_group_cfg(
        stiffness_scale_range=(0.7, 1.3), damping_scale_range=(0.7, 1.3)
    )
    changed, lines = _gate(cfg)
    assert changed is True
    assert cfg.stiffness_scale_range == (0.2, 1.3)
    assert cfg.damping_scale_range == (0.2, 1.3)
    # No per-group knob => per-group mode stays OFF.
    assert cfg.group_stiffness_scale_ranges is None
    assert any("PM_GAIN_DR_LOW,PM_GAIN_DR_HIGH" in ln for ln in lines)
    assert any("stiffness_scale_range = (0.2, 1.3)" in ln for ln in lines)
    assert any("damping_scale_range = (0.2, 1.3)" in ln for ln in lines)


def test_gate_per_group_knobs_stiff_legs_soft_upper(monkeypatch):
    _clear_gain_env(monkeypatch)
    monkeypatch.setenv("PM_GAIN_DR_LOW", "0.7")
    monkeypatch.setenv("PM_GAIN_DR_HIGH", "1.3")
    monkeypatch.setenv("PM_GAIN_DR_LOW_WAIST", "0.2")
    monkeypatch.setenv("PM_GAIN_DR_LOW_ARMS", "0.2")
    cfg = _h1_2_group_cfg(
        stiffness_scale_range=(0.7, 1.3), damping_scale_range=(0.7, 1.3)
    )
    changed, lines = _gate(cfg, label="FRESH-BUILD")
    assert changed is True
    assert cfg.group_stiffness_scale_ranges == {
        "legs": (0.7, 1.3), "waist": (0.2, 1.3), "arms": (0.2, 1.3),
    }
    assert cfg.group_damping_scale_ranges == cfg.group_stiffness_scale_ranges
    # Groups without an explicit knob inherit the global band, never drift.
    assert cfg.group_stiffness_scale_ranges["legs"] == (0.7, 1.3)
    # Proof lines name the group and both endpoints.
    for group in ("legs", "waist", "arms"):
        assert any(
            ln.startswith("FRESH-BUILD") and f"GROUP '{group}'" in ln
            for ln in lines
        ), group
    assert any("AUTO -> resolves to 'legs'" in ln for ln in lines)

    # And the sampler honors it end to end.
    out = _process(cfg, H1_2_DOF_NAMES)
    assert out["env_gain_scale_source"] == "legs"
    assert out["stiffness_scales"][:, out["group_columns"]["arms"]].min() < 0.5
    assert out["stiffness_scales"][:, out["group_columns"]["legs"]].min() >= 0.7 - 1e-6


def test_gate_arms_only_knobs_leave_legs_and_waist_randomized(monkeypatch):
    """LIVE-RUN GUARD (mm_canonical_v1): the launcher exports ONLY the two
    arm knobs -- no global PM_GAIN_DR_LOW/HIGH at all.

    This is the bug the per-group feature exists to prevent: widening the arms
    must NOT silently drop legs/waist out of gain DR. The sibling test above
    also sets the global knobs, so ``base`` there comes from the env; here it
    must come from the CONFIG's own global band (gain_dr_env_gates.py:208
    ``base = tuple(old)``), which is the path a real resume takes.
    """
    _clear_gain_env(monkeypatch)
    monkeypatch.setenv("PM_GAIN_DR_LOW_ARMS", "0.7")
    monkeypatch.setenv("PM_GAIN_DR_HIGH_ARMS", "3.0")
    cfg = _h1_2_group_cfg(
        stiffness_scale_range=(0.7, 1.3), damping_scale_range=(0.7, 1.3)
    )
    changed, lines = _gate(cfg, label="RESUME")
    assert changed is True

    # All THREE groups are written; the two unmentioned ones inherit the
    # config's global band rather than dropping out.
    assert cfg.group_stiffness_scale_ranges == {
        "legs": (0.7, 1.3), "waist": (0.7, 1.3), "arms": (0.7, 3.0),
    }
    # The arm knob has no _KD_ twin, so damping falls back to it: arm damping
    # is widened to 3.0x too, legs/waist stay at the global band.
    assert cfg.group_damping_scale_ranges == {
        "legs": (0.7, 1.3), "waist": (0.7, 1.3), "arms": (0.7, 3.0),
    }
    # Global bands are untouched, and the effort axis stays OFF.
    assert cfg.stiffness_scale_range == (0.7, 1.3)
    assert cfg.damping_scale_range == (0.7, 1.3)
    assert cfg.effort_limit_scale_range is None
    assert cfg.group_effort_limit_scale_ranges is None
    assert cfg.constant_damping_ratio is False
    for group in ("legs", "waist", "arms"):
        assert any(f"GROUP '{group}'" in ln for ln in lines), group

    # End to end through the sampler, per DOF.
    out = _process(cfg, H1_2_DOF_NAMES)
    k, d = out["stiffness_scales"], out["damping_scales"]
    for group in ("legs", "waist"):
        cols = out["group_columns"][group]
        assert k[:, cols].max() <= 1.3 + 1e-6, group
        assert d[:, cols].max() <= 1.3 + 1e-6, group
        # Still genuinely randomized -- not collapsed to 1.0.
        assert k[:, cols].std() > 0.05, group
    arms = out["group_columns"]["arms"]
    assert k[:, arms].max() > 1.3  # the widened band is actually explored
    assert d[:, arms].max() > 1.3
    assert k[:, arms].min() >= 0.7 - 1e-6
    assert out["effort_limit_scales"] is None


def test_gate_constant_zeta_and_scale_source_knobs(monkeypatch):
    _clear_gain_env(monkeypatch)
    monkeypatch.setenv("PM_GAIN_DR_CONSTANT_ZETA", "1")
    monkeypatch.setenv("PM_GAIN_DR_ENV_SCALE_SOURCE", "legs")
    cfg = _h1_2_group_cfg()
    changed, lines = _gate(cfg)
    assert changed is True
    assert cfg.constant_damping_ratio is True
    assert cfg.env_gain_scale_group == "legs"
    assert any("sqrt(stiffness scale)" in ln for ln in lines)

    monkeypatch.setenv("PM_GAIN_DR_ENV_SCALE_SOURCE", "hands")
    with pytest.raises(ValueError, match="PM_GAIN_DR_ENV_SCALE_SOURCE"):
        _gate(_h1_2_group_cfg())


def test_gate_rejects_inverted_group_range(monkeypatch):
    _clear_gain_env(monkeypatch)
    monkeypatch.setenv("PM_GAIN_DR_LOW_ARMS", "1.4")
    monkeypatch.setenv("PM_GAIN_DR_HIGH_ARMS", "0.5")
    with pytest.raises(ValueError, match="0 < low <= high"):
        _gate(_h1_2_group_cfg())


def test_gate_warns_loud_when_actuator_gain_absent(monkeypatch):
    _clear_gain_env(monkeypatch)
    monkeypatch.setenv("PM_GAIN_DR_LOW_ARMS", "0.2")
    changed, lines = _gate(None)
    assert changed is False
    assert len(lines) == 1 and "SKIPPED" in lines[0]
    assert "PM_GAIN_DR_LOW_ARMS" in lines[0]


def test_gate_tolerates_pre_field_pickled_config(monkeypatch):
    """A frozen config unpickled from a pre-per-group checkpoint has no
    per-group instance attributes; the gate must still fire cleanly."""
    _clear_gain_env(monkeypatch)
    monkeypatch.setenv("PM_GAIN_DR_LOW_ARMS", "0.2")
    cfg = _h1_2_group_cfg(
        stiffness_scale_range=(0.7, 1.3), damping_scale_range=(0.7, 1.3)
    )
    for attr in (
        "group_dof_patterns",
        "group_stiffness_scale_ranges",
        "group_damping_scale_ranges",
        "constant_damping_ratio",
        "env_gain_scale_group",
    ):
        cfg.__dict__.pop(attr, None)  # class defaults are all that remain
    changed, _ = _gate(cfg)
    assert changed is True
    assert cfg.group_stiffness_scale_ranges["arms"] == (0.2, 1.3)
    assert _process(cfg, H1_2_DOF_NAMES)["env_gain_scale_source"] == "legs"


def test_sampler_resumes_from_real_pre_field_pickle():
    """RESUME GUARD: a config pickled BEFORE the per-group fields existed must
    sample byte-identically to the modern single-config form.

    The gate-side twin above pokes ``__dict__``; this one does a real
    ``pickle`` round-trip of an old-shape instance and drives the SAMPLER,
    which is the path a ``resolved_configs.pt`` resume actually takes.
    ``_process`` reads ``group_stiffness_scale_ranges`` /
    ``group_damping_scale_ranges`` as plain attributes, so it survives only
    because those fields carry immutable ``default=None`` CLASS defaults.
    Converting any of them to ``default_factory`` would move the default off
    the class and break every in-flight checkpoint with an AttributeError --
    this test is the tripwire for that refactor.
    """
    modern = ActuatorGainDomainRandomizationConfig(
        dof_names=[".*"],
        stiffness_scale_range=(0.7, 1.3),
        damping_scale_range=(0.7, 1.3),
    )
    old = ActuatorGainDomainRandomizationConfig(
        dof_names=[".*"],
        stiffness_scale_range=(0.7, 1.3),
        damping_scale_range=(0.7, 1.3),
    )
    per_group_fields = [
        "group_dof_patterns",
        "group_stiffness_scale_ranges",
        "group_damping_scale_ranges",
        "group_effort_limit_scale_ranges",
        "constant_damping_ratio",
        "env_gain_scale_group",
    ]
    for attr in per_group_fields:
        old.__dict__.pop(attr, None)
    old = pickle.loads(pickle.dumps(old))
    # The pickled payload really is pre-field: no per-group instance state.
    assert not (set(per_group_fields) & set(old.__dict__))

    torch.manual_seed(1234)
    expected = _process(modern, H1_2_DOF_NAMES)
    torch.manual_seed(1234)
    got = _process(old, H1_2_DOF_NAMES)

    assert got["dof_indices"] == expected["dof_indices"]
    assert got["group_columns"] == {}
    for key in ("stiffness_scales", "damping_scales", "env_gain_scale"):
        assert torch.equal(got[key], expected[key]), key
    assert got["effort_limit_scales"] is None


# ---------------------------------------------------------------------------
# KINESTHETIC TEACHING (2026-08-04): effort-limit axis + independent per-group
# damping. Effort DR has NEVER run in production -- these are its first tests.
# ---------------------------------------------------------------------------


def test_per_group_effort_limit_ranges_and_apply_math():
    cfg = _h1_2_group_cfg(
        group_effort_limit_scale_ranges={"arms": (0.1, 0.4)},
    )
    out = _process(cfg, H1_2_DOF_NAMES)
    e = out["effort_limit_scales"]
    assert e is not None and e.shape == (NUM_ENVS, len(H1_2_DOF_NAMES))
    arm_cols = out["group_columns"]["arms"]
    leg_cols = out["group_columns"]["legs"]
    assert e[:, arm_cols].min() >= 0.1 - 1e-6
    assert e[:, arm_cols].max() <= 0.4 + 1e-6
    # Unmentioned groups keep NOMINAL effort limits exactly (no-op 1.0 band).
    assert torch.equal(e[:, leg_cols], torch.ones_like(e[:, leg_cols]))
    assert torch.equal(
        e[:, out["group_columns"]["waist"]],
        torch.ones_like(e[:, out["group_columns"]["waist"]]),
    )

    # Replicates the isaaclab apply path: new_effort[:, ids] *= scales.
    default_effort = torch.full((NUM_ENVS, len(H1_2_DOF_NAMES)), 120.0)
    new_effort = default_effort.clone()
    new_effort[:, out["dof_indices"]] *= e
    assert new_effort[:, arm_cols].max() <= 120.0 * 0.4 + 1e-4
    assert torch.equal(new_effort[:, leg_cols], default_effort[:, leg_cols])


def test_effort_axis_stays_off_when_unset():
    cfg = _h1_2_group_cfg(
        group_stiffness_scale_ranges={"arms": (0.2, 1.3)},
        group_damping_scale_ranges={"arms": (0.2, 1.3)},
    )
    assert cfg.effort_limit_scale_range is None
    assert cfg.group_effort_limit_scale_ranges is None
    # A gain-only marionette config must NOT start writing effort limits.
    assert _process(cfg, H1_2_DOF_NAMES)["effort_limit_scales"] is None


def test_per_group_damping_is_independent_of_stiffness():
    """Kinesthetic teaching wants LOW kd at the arms specifically -- kd is what
    a human dragging a hand fights directly."""
    cfg = _h1_2_group_cfg(
        stiffness_scale_range=(0.7, 1.3),
        damping_scale_range=(0.7, 1.3),
        group_stiffness_scale_ranges={"arms": (0.7, 1.3)},   # kp untouched
        group_damping_scale_ranges={"arms": (0.05, 0.2)},    # kd crushed
    )
    out = _process(cfg, H1_2_DOF_NAMES)
    arm_cols = out["group_columns"]["arms"]
    leg_cols = out["group_columns"]["legs"]
    assert out["stiffness_scales"][:, arm_cols].min() >= 0.7 - 1e-6
    assert out["damping_scales"][:, arm_cols].max() <= 0.2 + 1e-6
    assert out["damping_scales"][:, leg_cols].min() >= 0.7 - 1e-6


def test_gate_effort_knobs_global_and_per_group(monkeypatch):
    _clear_gain_env(monkeypatch)
    monkeypatch.setenv("PM_EFFORT_DR_LOW_ARMS", "0.1")
    monkeypatch.setenv("PM_EFFORT_DR_HIGH_ARMS", "0.4")
    cfg = _h1_2_group_cfg()
    changed, lines = _gate(cfg)
    assert changed is True
    assert cfg.group_effort_limit_scale_ranges == {
        "legs": (1.0, 1.0), "waist": (1.0, 1.0), "arms": (0.1, 0.4),
    }
    # Axis turned ON with the no-op global band so the sampler emits a matrix.
    assert cfg.effort_limit_scale_range == (1.0, 1.0)
    # Gains untouched: an effort-only config must not soften the PD gains.
    assert cfg.group_stiffness_scale_ranges is None
    assert cfg.group_damping_scale_ranges is None
    assert any("NEVER run in production" in ln for ln in lines)

    out = _process(cfg, H1_2_DOF_NAMES)
    assert out["effort_limit_scales"][:, out["group_columns"]["arms"]].max() <= 0.4 + 1e-6

    # Global effort knob alone: no group mode, uniform band.
    _clear_gain_env(monkeypatch)
    monkeypatch.setenv("PM_EFFORT_DR_LOW", "0.5")
    monkeypatch.setenv("PM_EFFORT_DR_HIGH", "1.0")
    cfg2 = _h1_2_group_cfg()
    _gate(cfg2)
    assert cfg2.effort_limit_scale_range == (0.5, 1.0)
    assert cfg2.group_effort_limit_scale_ranges is None


def test_gate_kd_knobs_override_stiffness_pair_for_damping(monkeypatch):
    _clear_gain_env(monkeypatch)
    monkeypatch.setenv("PM_GAIN_DR_LOW", "0.7")
    monkeypatch.setenv("PM_GAIN_DR_HIGH", "1.3")
    monkeypatch.setenv("PM_GAIN_DR_LOW_ARMS", "0.2")     # kp+kd for arms
    monkeypatch.setenv("PM_GAIN_DR_KD_LOW_ARMS", "0.05")  # kd only, wins
    monkeypatch.setenv("PM_GAIN_DR_KD_HIGH_ARMS", "0.2")
    cfg = _h1_2_group_cfg(
        stiffness_scale_range=(0.7, 1.3), damping_scale_range=(0.7, 1.3)
    )
    changed, lines = _gate(cfg)
    assert changed is True
    assert cfg.group_stiffness_scale_ranges["arms"] == (0.2, 1.3)
    assert cfg.group_damping_scale_ranges["arms"] == (0.05, 0.2)
    assert cfg.group_stiffness_scale_ranges["legs"] == (0.7, 1.3)
    assert cfg.group_damping_scale_ranges["legs"] == (0.7, 1.3)
    assert any("'arms' damping range = (0.05, 0.2)" in ln for ln in lines)

    out = _process(cfg, H1_2_DOF_NAMES)
    arm_cols = out["group_columns"]["arms"]
    assert out["damping_scales"][:, arm_cols].max() <= 0.2 + 1e-6
    assert out["stiffness_scales"][:, arm_cols].max() > 0.5


def test_gate_effort_knob_validation(monkeypatch):
    _clear_gain_env(monkeypatch)
    monkeypatch.setenv("PM_EFFORT_DR_LOW_ARMS", "0.9")
    monkeypatch.setenv("PM_EFFORT_DR_HIGH_ARMS", "0.1")
    with pytest.raises(ValueError, match="0 < low <= high"):
        _gate(_h1_2_group_cfg())


def test_gate_effort_knobs_listed_in_env_var_registry():
    for var in (
        "PM_EFFORT_DR_LOW", "PM_EFFORT_DR_HIGH",
        "PM_EFFORT_DR_LOW_ARMS", "PM_EFFORT_DR_HIGH_ARMS",
        "PM_GAIN_DR_KD_LOW_ARMS", "PM_GAIN_DR_KD_HIGH_ARMS",
        "PM_GAIN_DR_LOW_LEGS", "PM_GAIN_DR_HIGH_WAIST",
        "PM_GAIN_DR_CONSTANT_ZETA", "PM_GAIN_DR_ENV_SCALE_SOURCE",
    ):
        assert var in GAIN_DR_ENV_VARS, var


def test_isaaclab_apply_path_wires_the_effort_axis():
    """The effort axis has never run in production, so guard its wiring by
    source inspection (importing the IsaacLab backend needs a live Kit app).

    Verified 2026-08-04: for ControlType.BUILT_IN_PD / ImplicitActuatorCfg
    (this fleet), write_joint_effort_limit_to_sim -> root_physx_view.
    set_dof_max_forces IS the torque saturation the PD solver applies. Its
    IsaacLab docstring warns it does not update python-side actuator models
    (#128) -- harmless for implicit actuators, but a robot moved to
    IdealPDActuatorCfg would get only half the clip.
    """
    from pathlib import Path

    import protomotions

    src = (
        Path(protomotions.__file__).parent / "simulator" / "isaaclab" / "simulator.py"
    ).read_text()
    assert 'gain_dr.get("effort_limit_scales") is not None' in src
    assert "get_dof_max_forces()" in src
    assert "write_joint_effort_limit_to_sim(new_effort)" in src
    assert "new_effort[:, lab_dof_ids] *= effort_scales" in src
    assert "default_effort_limit_mean" in src  # proof line
