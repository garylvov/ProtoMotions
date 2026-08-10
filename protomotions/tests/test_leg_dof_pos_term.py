# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard tests for the v66' DOF-subset plumbing: factory passthrough, the
joint-space eval kernel, and the kernel-form / sigma pins.

The last two tests are the important ones. The kernel argument is
``e / sigma^2`` with ``e`` a MEAN SQUARED error -- NOT ``(d / sigma)^2`` -- and
that distinction was actually gotten backwards earlier in this campaign; it
changes the sizing by a square. And sigma=0.40 here is WIDER than the shared
``dof_pos_track``'s 0.35, which looks like a typo and is not: narrowing it to
0.35 puts the p90 deep-crouch frames at 0.031, below the 0.05 underflow floor,
i.e. dead -- the same class of defect as v57's ``hold_joint_quiet`` reading
exactly 0.0 for 5,241 steps. Both are pinned so nobody "simplifies" them.
"""
import math

import pytest
import torch

from protomotions.envs.component_factories import (
    dof_pos_metric_factory,
    dof_pos_track_rew_factory,
)
from protomotions.envs.rewards import tracking
from protomotions.envs.terminations import dof_pos_max_error

# H1-2 DOF order; the four v66' leg DOFs.
LEG_DOF_INDICES = [3, 9, 2, 8]  # l_knee, r_knee, l_hip_roll, r_hip_roll
V66P_SIGMA = 0.40
SHARED_DOF_POS_TRACK_SIGMA = 0.35
UNDERFLOW_FLOOR = 0.05

# Measured on v62_ep7200, 25 non-fallen deep_hinge_crouch clips, full per-frame
# DOF trajectories. e_B = mean squared error over the four leg DOFs (rad^2).
E_B_DEEP_MEDIAN = 0.1673   # at t*, the deepest frame of each clip
E_B_ALL_MEDIAN = 0.0539    # over all frames
E_B_DEEP_P90 = 0.4255      # sigma=0.40 -> 0.070; sigma=0.35 -> 0.031 (dead)
E_ALL_27_DOF_DEEP = 0.0095  # same frames, but averaged over all 27 DOFs


def test_factory_default_is_byte_identical():
    """RESUME RULE: no dof_indices => no new static param => the shared
    dof_pos_track term is bit-for-bit what it was before v66'."""
    comp = dof_pos_track_rew_factory(weight=1.0, sigma=SHARED_DOF_POS_TRACK_SIGMA)
    assert "indices" not in comp.static_params
    assert "dof_indices" not in comp.static_params
    assert comp.static_params == {"weight": 1.0, "sigma": SHARED_DOF_POS_TRACK_SIGMA}


def test_factory_passes_dof_indices_through_to_the_kernel_argument():
    """The factory-level name is dof_indices (so it can never be confused with
    the BODY-index kwarg the Cartesian factories take); the kernel's parameter
    is ``indices``. Guard the mapping, because a silently-dropped subset would
    leave a term that looks right and scores all 27 DOFs."""
    comp = dof_pos_track_rew_factory(
        weight=1.3, sigma=V66P_SIGMA, dof_indices=LEG_DOF_INDICES
    )
    assert comp.static_params["indices"] == LEG_DOF_INDICES
    assert comp.static_params["sigma"] == V66P_SIGMA
    assert comp.static_params["weight"] == 1.3
    # NO dual-sigma companion, deliberately (see the teacher gate).
    assert "fine_weight" not in comp.static_params
    assert "fine_sigma" not in comp.static_params
    assert comp.compute_func is tracking.compute_dof_pos_track_rew


def test_subset_reward_matches_a_hand_computed_value():
    """Subset math, against a number worked out by hand rather than by rerunning
    the implementation against itself."""
    ref = torch.zeros(1, 27)
    current = torch.zeros(1, 27)
    # Leg DOFs: errors 0.4, 0.2, 0.1, 0.1 rad. Everything else large, to prove
    # the other 23 DOFs genuinely do not enter the subset term.
    current[0, 3] = 0.4
    current[0, 9] = 0.2
    current[0, 2] = 0.1
    current[0, 8] = 0.1
    current[0, 0] = 5.0
    current[0, 13] = -5.0

    # e = mean of squared errors over the FOUR selected DOFs:
    #   (0.16 + 0.04 + 0.01 + 0.01) / 4 = 0.22 / 4 = 0.055
    e_hand = 0.055
    expected = math.exp(-e_hand / V66P_SIGMA**2)

    rew = tracking.compute_dof_pos_track_rew(
        current, ref, sigma=V66P_SIGMA, indices=torch.tensor(LEG_DOF_INDICES)
    )
    assert torch.allclose(rew, torch.tensor([expected]), atol=1e-6)

    # And it is genuinely restricted: the full-27 term sees the two large errors
    # and collapses, while the subset term does not.
    rew_all = tracking.compute_dof_pos_track_rew(current, ref, sigma=V66P_SIGMA)
    assert rew_all < 1e-4 < rew.item()


def test_kernel_argument_is_e_over_sigma_squared_not_d_over_sigma_squared():
    """THE FORM PIN. exp(-e/sigma^2) with e = mean(d^2), NOT exp(-(d/sigma)^2).

    The two agree only when the subset holds a single DOF, so a test built on
    one DOF would pass under either reading. This uses four unequal errors,
    where mean(d^2) and any per-DOF (d/sigma)^2 differ, and additionally pins
    that the exponent is LINEAR in e (halving e must exactly square-root the
    reward), which the wrong form does not satisfy.
    """
    ref = torch.zeros(1, 27)
    current = torch.zeros(1, 27)
    for idx, err in zip(LEG_DOF_INDICES, (0.5, 0.3, 0.2, 0.1)):
        current[0, idx] = err
    idx_t = torch.tensor(LEG_DOF_INDICES)

    e = (0.25 + 0.09 + 0.04 + 0.01) / 4  # 0.0975
    rew = tracking.compute_dof_pos_track_rew(
        current, ref, sigma=V66P_SIGMA, indices=idx_t
    ).item()
    assert rew == pytest.approx(math.exp(-e / V66P_SIGMA**2), rel=1e-6)

    # The WRONG form -- exp(-mean((d/sigma)^2)) is the same thing, but
    # exp(-(mean|d|/sigma)^2) is not -- must NOT match.
    mean_abs = (0.5 + 0.3 + 0.2 + 0.1) / 4
    wrong = math.exp(-((mean_abs / V66P_SIGMA) ** 2))
    assert abs(rew - wrong) > 1e-3

    # Linearity of the exponent in e: r(e/2) == sqrt(r(e)).
    half = current.clone()
    half[0, idx_t] = current[0, idx_t] / math.sqrt(2.0)
    rew_half = tracking.compute_dof_pos_track_rew(
        half, ref, sigma=V66P_SIGMA, indices=idx_t
    ).item()
    assert rew_half == pytest.approx(math.sqrt(rew), rel=1e-6)


def test_sigma_040_is_the_measured_optimum_and_035_is_dead():
    """THE SIGMA PIN. 0.40 is WIDER than the shared term's 0.35 on purpose.

    sigma_opt = sqrt(e) maximises exp(-e/sigma^2)/sigma^2. Three independent
    facts have to hold for 0.40 to be right, and this asserts all three so that
    "simplifying" it toward 0.35 fails loudly:
      1. 0.40 is sigma_opt at the deep-frame median (sqrt(0.1673) = 0.409);
      2. 0.40 keeps the p90 DEEP frames above the underflow floor and 0.35 does
         not -- 0.35 is already a dead channel where it matters most;
      3. the gain comes from the DOF RESTRICTION, not the width: over four DOFs
         e is ~17.6x what it is over all 27, which is what moves the term off
         the Gaussian's flat top.
    """
    assert math.sqrt(E_B_DEEP_MEDIAN) == pytest.approx(0.409, abs=0.002)

    gradient = lambda e, s: math.exp(-e / s**2) / s**2
    for worse in (0.20, 0.25, 0.30, 0.35, 0.50, 0.60):
        assert gradient(E_B_DEEP_MEDIAN, V66P_SIGMA) > gradient(
            E_B_DEEP_MEDIAN, worse
        ), f"sigma {worse} beat {V66P_SIGMA} at the deep-frame median"

    live_at_040 = math.exp(-E_B_DEEP_P90 / V66P_SIGMA**2)
    dead_at_035 = math.exp(-E_B_DEEP_P90 / SHARED_DOF_POS_TRACK_SIGMA**2)
    assert live_at_040 > UNDERFLOW_FLOOR, (
        f"sigma {V66P_SIGMA} underflowed at the p90 deep frame: {live_at_040:.4f}"
    )
    assert dead_at_035 < UNDERFLOW_FLOOR, (
        "0.35 was expected to be BELOW the underflow floor at the p90 deep "
        f"frame; got {dead_at_035:.4f}. If this ever passes, re-derive the "
        "sizing rather than assuming 0.35 became safe."
    )

    # The restriction, not the width, is the lever.
    assert E_B_DEEP_MEDIAN / E_ALL_27_DOF_DEEP > 17.0
    # Deep frames are where the term must bite: 3.10x the all-frame error.
    assert E_B_DEEP_MEDIAN / E_B_ALL_MEDIAN == pytest.approx(
        3.10, abs=0.05
    )


def test_dof_pos_max_error_default_covers_all_dofs_and_subset_is_bounded_by_it():
    """The EVAL SURFACE kernel. Default (no dof_indices) must be the whole-robot
    number, and any subset must be <= it -- the property that makes the subset
    metric directly comparable with the whole-body one."""
    ref = torch.zeros(2, 27)
    current = torch.zeros(2, 27)
    current[0, 3] = 0.4      # a leg DOF is the worst
    current[0, 20] = 0.1
    current[1, 3] = 0.1
    current[1, 20] = -0.9    # a NON-leg DOF is the worst

    full = dof_pos_max_error(current, ref)
    assert torch.allclose(full, torch.tensor([0.4, 0.9]), atol=1e-6)

    subset = dof_pos_max_error(current, ref, dof_indices=torch.tensor(LEG_DOF_INDICES))
    assert torch.allclose(subset, torch.tensor([0.4, 0.1]), atol=1e-6)
    assert bool((subset <= full + 1e-9).all())


def test_dof_pos_metric_factory_binds_the_joint_space_context():
    """READER/WRITER LAW: the metric must read the SAME two quantities the
    reward does, or the eval surface and the reward can drift apart."""
    comp = dof_pos_metric_factory(dof_indices=LEG_DOF_INDICES)
    assert comp.compute_func is dof_pos_max_error
    assert comp.static_params["dof_indices"] == LEG_DOF_INDICES
    assert set(comp.dynamic_vars) == {"current_dof_pos", "ref_dof_pos"}

    reward = dof_pos_track_rew_factory(dof_indices=LEG_DOF_INDICES)
    assert set(comp.dynamic_vars) == set(reward.dynamic_vars)

    plain = dof_pos_metric_factory()
    assert plain.static_params == {}
