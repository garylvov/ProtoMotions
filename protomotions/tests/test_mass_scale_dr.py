# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit smokes for MASS-DR (body-mass scale domain randomization).

Covers (CPU-only):
- MassScaleDomainRandomizationConfig validation (body selection XOR, range
  sanity, defaults).
- DomainRandomizationConfig carries the new optional mass_scale slot (None
  default = off, pre-field pickles safe via getattr).
- Simulator._process_mass_scale_domain_randomization sampling: shapes,
  bounds, per-env spread, optional all-links samples, main-body index match.
- Apply-path composition logic (replicated math): main-body multiplier
  composes with the all-links multiplier; untouched bodies keep default mass.
"""

from types import SimpleNamespace

import pytest
import torch

from protomotions.simulator.base_simulator.config import (
    DomainRandomizationConfig,
    MassScaleDomainRandomizationConfig,
)
from protomotions.simulator.base_simulator.simulator import Simulator

BODY_NAMES = ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link"]
NUM_ENVS = 512


def _mock_sim():
    return SimpleNamespace(
        num_envs=NUM_ENVS,
        robot_config=SimpleNamespace(
            kinematic_info=SimpleNamespace(body_names=BODY_NAMES)
        ),
    )


def _process(cfg):
    return Simulator._process_mass_scale_domain_randomization(_mock_sim(), cfg)


def test_config_requires_body_selection():
    with pytest.raises(ValueError):
        MassScaleDomainRandomizationConfig()
    with pytest.raises(ValueError):
        MassScaleDomainRandomizationConfig(body_names=["torso_link"], body_indices=[1])


def test_config_rejects_bad_ranges():
    with pytest.raises(ValueError):
        MassScaleDomainRandomizationConfig(
            body_names=["torso_link"], mass_scale_range=(1.3, 1.0)
        )
    with pytest.raises(ValueError):
        MassScaleDomainRandomizationConfig(
            body_names=["torso_link"], all_links_scale_range=(0.0, 1.05)
        )


def test_config_defaults_and_dr_slot():
    cfg = MassScaleDomainRandomizationConfig(body_names=["torso_link"])
    assert cfg.mass_scale_range == (1.0, 1.3)
    assert cfg.all_links_scale_range is None
    dr = DomainRandomizationConfig()
    assert getattr(dr, "mass_scale", None) is None  # off by default
    dr = DomainRandomizationConfig(mass_scale=cfg)
    assert dr.mass_scale is cfg


def test_process_samples_shapes_bounds_and_spread():
    cfg = MassScaleDomainRandomizationConfig(
        body_names=["torso_link"], mass_scale_range=(1.0, 1.3)
    )
    out = _process(cfg)
    assert out["body_indices"] == [BODY_NAMES.index("torso_link")]
    scales = out["scales"]
    assert scales.shape == (NUM_ENVS, 1)
    assert (scales >= 1.0).all() and (scales <= 1.3).all()
    # Per-env spread: 512 U(1.0,1.3) samples must not collapse.
    assert scales.std() > 0.02
    assert abs(scales.mean().item() - 1.15) < 0.03
    assert out["all_links_scales"] is None


def test_process_samples_all_links_when_configured():
    cfg = MassScaleDomainRandomizationConfig(
        body_names=["torso_link"],
        mass_scale_range=(1.0, 1.3),
        all_links_scale_range=(0.95, 1.05),
    )
    out = _process(cfg)
    al = out["all_links_scales"]
    assert al.shape == (NUM_ENVS, len(BODY_NAMES))
    assert (al >= 0.95).all() and (al <= 1.05).all()


def test_apply_composition_math():
    """Replicates the isaaclab apply-path composition on CPU tensors."""
    cfg = MassScaleDomainRandomizationConfig(
        body_names=["torso_link"],
        mass_scale_range=(1.0, 1.3),
        all_links_scale_range=(0.95, 1.05),
    )
    out = _process(cfg)
    default_masses = torch.tensor([5.0, 20.0, 1.0, 1.0]).repeat(NUM_ENVS, 1)
    masses = default_masses.clone()
    # identity ordering (find_bodies preserve_order on identical names)
    masses *= out["all_links_scales"]
    masses[:, out["body_indices"]] *= out["scales"]

    torso = BODY_NAMES.index("torso_link")
    ratio = masses[:, torso] / default_masses[:, torso]
    assert (ratio >= 1.0 * 0.95 - 1e-6).all() and (ratio <= 1.3 * 1.05 + 1e-6).all()
    # non-main bodies only see the all-links factor
    pelvis_ratio = masses[:, 0] / default_masses[:, 0]
    assert (pelvis_ratio >= 0.95 - 1e-6).all() and (pelvis_ratio <= 1.05 + 1e-6).all()
    # composition really applied both factors on the torso
    expected = out["all_links_scales"][:, torso] * out["scales"][:, 0]
    assert torch.allclose(ratio, expected, atol=1e-6)
