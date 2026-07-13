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
