# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard tests for the v64 BODY-subset plumbing behind ``foot_relative_body_pos``:
the ``body_indices`` argument on ``relative_body_pos_max_error`` (the eval
surface), the matching kwarg on ``relative_body_pos_metric_factory``, and the
kernel-form / sigma pins for the reward itself.

WHY THIS FILE EXISTS AT ALL (2026-08-10). The v64 foot term shipped with a guard
file that called ``build_teacher_config()`` -- a function that does not exist in
this stack. Every assertion that touched the config therefore never ran, and the
"7/7 passing" reported at ship time was 7 pure-arithmetic tests plus two dead
ones. v64 trained for hours on that basis. The config-facing half now lives in
imprint ``test/test_foot_pos_term.py`` and goes through ``build_standard_configs``
-- the same entry point ``train_agent.py`` uses -- so it cannot pass against a
function nobody calls.

THE FORM PIN is the load-bearing test here. The kernel argument is
``e / sigma^2`` with ``e`` a MEAN SQUARED error over the selected bodies -- NOT
``(d / sigma)^2`` for a per-body distance ``d``, and NOT the mean of per-body
Gaussians. Those three readings AGREE on a single-body subset, so a test written
against one body passes under all of them; this confusion has already produced
one wrong analysis in this campaign. Every form assertion below therefore uses
FOUR UNEQUAL per-body errors, and the single-body degeneracy is asserted
explicitly so nobody "simplifies" the test back into ambiguity.
"""
import math

import numpy as np
import pytest
import torch

from protomotions.envs.component_factories import relative_body_pos_metric_factory
from protomotions.envs.rewards import tracking
from protomotions.envs.terminations import relative_body_pos_max_error

# H1-2 body order; the four v64 ankle bodies.
FOOT_BODY_INDICES = [5, 6, 11, 12]
FOOT_BODY_NAMES = [
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
]
NUM_BODIES = 27

V64_FOOT_SIGMA = 0.3           # coarse; shared with the wrist term by design
V64_FOOT_FINE_SIGMA = 0.15     # NOT the wrist term's 0.03 -- see the pins below
V64_FOOT_FINE_WEIGHT = 1.0
V64_FOOT_WEIGHT = 1.3
WRIST_FINE_SIGMA = 0.03
V64_REL_POS_FINE_SIGMA = 0.06  # v64 DELTA 1; v62 ran 0.03

UNDERFLOW_FLOOR = 0.05         # below this the channel supplies no usable gradient

# Measured on v62_ep7200 over canonical_eval_v1, 260 non-fallen clips.
# e_foot = mean over the four ankle bodies of squared local-frame error (m^2).
# Source: eval_harness_recovery/v62_ep7200_perbody_canonical_v1.npz
E_FOOT = {
    "median": 5.49e-3,
    "p75": 1.81e-2,
    "wbdyn_median": 2.83e-2,
    "crouch_median": 1.05e-2,
}
PERBODY_NPZ = (
    "/oscar/data/stellex/glvov/eval_harness_recovery/"
    "v62_ep7200_perbody_canonical_v1.npz"
)


# ------------------------------------------------------------------ helpers --
def _identity_frame(num_envs=1, num_bodies=NUM_BODIES):
    """Anchor at the origin with identity heading, so the anchor-relative
    heading-local frame is the identity map and a hand-computed displacement is
    exactly the error the kernel sees. The frame construction itself is guarded
    by test_wrist_relative_body_pos_stat.py; what is under test here is the
    SUBSET reduction, so the frame is deliberately made trivial."""
    ref_pos = torch.zeros(num_envs, num_bodies, 3)
    cur_pos = torch.zeros(num_envs, num_bodies, 3)
    # w-last identity quaternion.
    rot = torch.zeros(num_envs, num_bodies, 4)
    rot[..., 3] = 1.0
    anchor_rot = torch.zeros(num_envs, 4)
    anchor_rot[:, 3] = 1.0
    anchor_pos = torch.zeros(num_envs, 3)
    return ref_pos, cur_pos, rot, anchor_rot, anchor_pos


def _foot_reward(cur_pos, ref_pos, rot, anchor_rot, anchor_pos, **kwargs):
    return tracking.compute_relative_body_pos_rew(
        cur_pos,
        ref_pos,
        anchor_rot,
        rot,
        anchor_pos,
        anchor_idx=0,
        body_indices=torch.tensor(FOOT_BODY_INDICES),
        **kwargs,
    )


# ------------------------------------------------ the default (RESUME RULE) --
def test_metric_factory_default_is_byte_identical():
    """No body_indices => no new static param => every pre-v64
    ``evaluation_components`` entry builds exactly what it built before."""
    comp = relative_body_pos_metric_factory()
    assert comp.static_params == {}
    assert comp.compute_func is relative_body_pos_max_error


def test_metric_factory_threshold_only_is_unchanged():
    comp = relative_body_pos_metric_factory(threshold=0.25)
    assert comp.static_params == {"threshold": 0.25}


def test_max_error_kernel_default_is_the_whole_body_number():
    """Default (no body_indices) must still reduce over ALL bodies, so the
    pre-v64 ``relative_body_pos`` eval channel is untouched."""
    ref_pos, cur_pos, rot, anchor_rot, anchor_pos = _identity_frame()
    cur_pos[0, 5, 0] = 0.10       # an ankle
    cur_pos[0, 19, 0] = 0.40      # a wrist, worse
    full = relative_body_pos_max_error(cur_pos, ref_pos, anchor_pos, anchor_rot, rot, 0)
    assert torch.allclose(full, torch.tensor([0.40]), atol=1e-6)


# ------------------------------------------------------------ subset math --
def test_metric_factory_passes_body_indices_through():
    comp = relative_body_pos_metric_factory(body_indices=FOOT_BODY_INDICES)
    assert comp.compute_func is relative_body_pos_max_error
    assert comp.static_params["body_indices"] == FOOT_BODY_INDICES
    # The metric must read the SAME quantities the reward does, or the eval
    # surface and the objective can drift apart (READER/WRITER LAW).
    assert set(comp.dynamic_vars) == {
        "current_rigid_body_pos",
        "ref_rigid_body_pos",
        "current_anchor_pos",
        "current_anchor_rot",
        "ref_rigid_body_rot",
        "anchor_idx",
    }


def test_max_error_subset_matches_a_hand_computed_value():
    """Hand-computed, not re-derived from the implementation: the subset max
    must see the ankles only, and must be bounded by the whole-body max."""
    ref_pos, cur_pos, rot, anchor_rot, anchor_pos = _identity_frame(num_envs=2)
    # env 0: the worst body IS an ankle.
    cur_pos[0, 5, 0] = 0.20
    cur_pos[0, 11, 1] = 0.07
    cur_pos[0, 19, 0] = 0.05
    # env 1: the worst body is NOT an ankle -- the subset must ignore it.
    cur_pos[1, 6, 2] = 0.03
    cur_pos[1, 19, 0] = 0.90

    full = relative_body_pos_max_error(cur_pos, ref_pos, anchor_pos, anchor_rot, rot, 0)
    subset = relative_body_pos_max_error(
        cur_pos,
        ref_pos,
        anchor_pos,
        anchor_rot,
        rot,
        0,
        body_indices=torch.tensor(FOOT_BODY_INDICES),
    )
    assert torch.allclose(full, torch.tensor([0.20, 0.90]), atol=1e-6)
    assert torch.allclose(subset, torch.tensor([0.20, 0.03]), atol=1e-6)
    # The property that makes the subset directly comparable to the whole-body
    # number: only the final max() is restricted, the frame is not.
    assert bool((subset <= full + 1e-9).all())


def test_subset_reward_matches_a_hand_computed_value():
    """The reward's own subset math, against a number worked out by hand.

    e = mean over the FOUR selected bodies of squared distance:
        (0.16 + 0.04 + 0.01 + 0.01) / 4 = 0.22 / 4 = 0.055 m^2
    """
    ref_pos, cur_pos, rot, anchor_rot, anchor_pos = _identity_frame()
    for idx, d in zip(FOOT_BODY_INDICES, (0.4, 0.2, 0.1, 0.1)):
        cur_pos[0, idx, 0] = d
    # Non-selected bodies get large errors, to prove they genuinely do not enter.
    cur_pos[0, 0, 0] = 5.0
    cur_pos[0, 19, 1] = -5.0

    e_hand = 0.055
    rew = _foot_reward(cur_pos, ref_pos, rot, anchor_rot, anchor_pos,
                       sigma=V64_FOOT_SIGMA)
    assert rew.item() == pytest.approx(math.exp(-e_hand / V64_FOOT_SIGMA**2), rel=1e-6)

    # ... and it is genuinely restricted: the whole-body term collapses on the
    # same input while the foot subset does not.
    rew_all = tracking.compute_relative_body_pos_rew(
        cur_pos, ref_pos, anchor_rot, rot, anchor_pos, anchor_idx=0,
        sigma=V64_FOOT_SIGMA,
    )
    assert rew_all.item() < 1e-4 < rew.item()


# ------------------------------------------------------------- THE FORM PIN --
def test_kernel_argument_is_e_over_sigma_squared_with_four_unequal_errors():
    """THE FORM PIN. r = exp(-e/sigma^2) with e = mean over the subset of
    SQUARED distances -- not exp(-(d/sigma)^2) for some representative d, and
    not the mean of per-body Gaussians.

    All three readings coincide on a SINGLE-body subset (asserted below), which
    is exactly why this uses four UNEQUAL errors. The e/sigma^2 vs (d/sigma)^2
    confusion has already produced one wrong analysis in this campaign.
    """
    errors = (0.5, 0.3, 0.2, 0.1)
    assert len(set(errors)) == 4, "the pin is void unless the four errors differ"

    ref_pos, cur_pos, rot, anchor_rot, anchor_pos = _identity_frame()
    for idx, d in zip(FOOT_BODY_INDICES, errors):
        cur_pos[0, idx, 0] = d

    e = sum(d**2 for d in errors) / 4          # 0.0975 m^2
    rew = _foot_reward(cur_pos, ref_pos, rot, anchor_rot, anchor_pos,
                       sigma=V64_FOOT_SIGMA).item()
    assert rew == pytest.approx(math.exp(-e / V64_FOOT_SIGMA**2), rel=1e-6)

    # WRONG FORM 1 -- exp(-(mean|d| / sigma)^2), the "(d/sigma)^2" reading.
    mean_abs = sum(errors) / 4
    wrong_d_over_sigma = math.exp(-((mean_abs / V64_FOOT_SIGMA) ** 2))
    assert abs(rew - wrong_d_over_sigma) > 1e-3, (
        "the (d/sigma)^2 reading is indistinguishable here; the test has lost "
        "its power -- restore four unequal errors."
    )

    # WRONG FORM 2 -- the mean of per-body Gaussians (reduce AFTER exponentiating
    # instead of before). This is the reading that makes a body-restricted term
    # look pointless, because it removes the burying effect the subset exists to
    # undo.
    wrong_mean_of_exp = sum(
        math.exp(-(d**2) / V64_FOOT_SIGMA**2) for d in errors
    ) / 4
    assert abs(rew - wrong_mean_of_exp) > 1e-3

    # WRONG FORM 3 -- exp(-e^2/sigma^2), i.e. squaring an already-squared error.
    wrong_e_squared = math.exp(-(e**2) / V64_FOOT_SIGMA**2)
    assert abs(rew - wrong_e_squared) > 1e-3

    # POSITIVE pin on the exponent's LINEARITY in e, which only the correct form
    # satisfies: halving e must exactly square-root the reward.
    half = cur_pos.clone()
    for idx in FOOT_BODY_INDICES:
        half[0, idx, 0] = cur_pos[0, idx, 0] / math.sqrt(2.0)
    rew_half = _foot_reward(half, ref_pos, rot, anchor_rot, anchor_pos,
                            sigma=V64_FOOT_SIGMA).item()
    assert rew_half == pytest.approx(math.sqrt(rew), rel=1e-6)


def test_a_single_body_subset_cannot_distinguish_the_kernel_forms():
    """The degeneracy the test above exists to escape, asserted explicitly so a
    future edit cannot 'simplify' the form pin down to one body and still look
    like it is guarding something."""
    d = 0.37
    ref_pos, cur_pos, rot, anchor_rot, anchor_pos = _identity_frame()
    cur_pos[0, FOOT_BODY_INDICES[0], 0] = d
    rew = tracking.compute_relative_body_pos_rew(
        cur_pos, ref_pos, anchor_rot, rot, anchor_pos, anchor_idx=0,
        sigma=V64_FOOT_SIGMA,
        body_indices=torch.tensor([FOOT_BODY_INDICES[0]]),
    ).item()
    assert rew == pytest.approx(math.exp(-(d**2) / V64_FOOT_SIGMA**2), rel=1e-6)
    assert rew == pytest.approx(math.exp(-((d / V64_FOOT_SIGMA) ** 2)), rel=1e-6)


def test_dual_sigma_companion_adds_a_second_gaussian_of_the_same_argument():
    """v64 runs the term WITH a fine companion (fine_weight 1.0), so the pin has
    to cover the composed kernel, not just the coarse half."""
    errors = (0.5, 0.3, 0.2, 0.1)
    ref_pos, cur_pos, rot, anchor_rot, anchor_pos = _identity_frame()
    for idx, d in zip(FOOT_BODY_INDICES, errors):
        cur_pos[0, idx, 0] = d
    e = sum(d**2 for d in errors) / 4

    rew = _foot_reward(
        cur_pos, ref_pos, rot, anchor_rot, anchor_pos,
        sigma=V64_FOOT_SIGMA,
        fine_weight=V64_FOOT_FINE_WEIGHT,
        fine_sigma=V64_FOOT_FINE_SIGMA,
    ).item()
    expected = math.exp(-e / V64_FOOT_SIGMA**2) + V64_FOOT_FINE_WEIGHT * math.exp(
        -e / V64_FOOT_FINE_SIGMA**2
    )
    assert rew == pytest.approx(expected, rel=1e-6)
    # The companion raises the term's maximum from w to w*(1+fine_weight); the
    # coarse channel is untouched, which is what keeps capture range intact.
    coarse = _foot_reward(cur_pos, ref_pos, rot, anchor_rot, anchor_pos,
                          sigma=V64_FOOT_SIGMA).item()
    assert rew > coarse


# ------------------------------------------------------------- THE SIGMA PIN --
@pytest.mark.parametrize("where", sorted(E_FOOT))
def test_foot_fine_channel_is_live_at_the_measured_error(where):
    """THE anti-underflow guard. The fine channel must supply gradient at the
    error we actually observe, not at the error we wish we had. The campaign has
    been bitten by this twice: v57's ``hold_joint_quiet`` read exactly 0.0 for
    5,241 steps, and v62's ``relative_body_pos`` fine channel is dead on
    whole_body_dynamic (gradient 3 vs 940)."""
    e = E_FOOT[where]
    v = math.exp(-e / V64_FOOT_FINE_SIGMA**2)
    assert v > UNDERFLOW_FLOOR, (
        f"foot fine channel underflowed at measured e_foot[{where}]={e:.3e}: "
        f"value {v:.4f} <= {UNDERFLOW_FLOOR}. This is the hold_joint_quiet bug "
        f"again. Widen PM_FOOT_POS_FINE_SIGMA toward sqrt(e)={math.sqrt(e):.3f}."
    )


def test_the_wrist_fine_sigma_would_have_been_dead_on_the_feet():
    """Pins WHY the two terms do not share a fine width, so nobody 'simplifies'
    them back together. 0.03 is 22x below the usable-gradient floor at the
    MEDIAN foot error."""
    v = math.exp(-E_FOOT["median"] / WRIST_FINE_SIGMA**2)
    assert v < UNDERFLOW_FLOOR / 10, (
        f"the wrist fine width {WRIST_FINE_SIGMA} gives {v:.2e} at the median "
        f"foot error -- {UNDERFLOW_FLOOR / v:.0f}x below the usable-gradient "
        "floor, i.e. dead. Do not share a fine width between the two terms."
    )


def test_foot_fine_sigma_015_is_the_measured_optimum():
    """0.15 is not a round number someone liked: peak gradient of
    exp(-e/s^2)/s^2 is at s = sqrt(e), and the foot error population sits at
    sqrt(median)=0.074 .. sqrt(p75)=0.135. Anything materially tighter is dead;
    anything much wider throws away precision."""
    gradient = lambda e, s: math.exp(-e / s**2) / s**2
    assert math.sqrt(E_FOOT["p75"]) == pytest.approx(0.135, abs=0.002)
    for worse in (0.03, 0.05, 0.30, 0.50):
        assert gradient(E_FOOT["p75"], V64_FOOT_FINE_SIGMA) > gradient(
            E_FOOT["p75"], worse
        ), f"fine sigma {worse} beat {V64_FOOT_FINE_SIGMA} at the p75 foot error"


def test_rel_pos_fine_sigma_006_beats_the_alternatives():
    """v64 DELTA 1 is a maximisation, not a split-the-difference. Guards against
    a future edit drifting it to 0.10-0.15, which is strictly dominated, or back
    to v62's 0.03."""
    gradient = lambda e, s: math.exp(-e / s**2) / s**2
    crouch, wbdyn = 2.19e-3, 5.46e-3          # measured category medians of e
    for worse in (0.03, 0.10, 0.15):
        assert gradient(crouch, V64_REL_POS_FINE_SIGMA) + gradient(
            wbdyn, V64_REL_POS_FINE_SIGMA
        ) > gradient(crouch, worse) + gradient(wbdyn, worse), (
            f"rel_pos fine sigma {worse} beat {V64_REL_POS_FINE_SIGMA}"
        )


@pytest.mark.skipif(
    not __import__("os").path.exists(PERBODY_NPZ),
    reason=f"measured per-body eval dump not present at {PERBODY_NPZ}",
)
def test_sigma_does_not_retarget_bodies_only_clips():
    """The measurement that makes a body-restricted term NECESSARY rather than a
    nice-to-have: the per-body gradient SHARE inside ``relative_body_pos`` is
    invariant to fine_sigma (the v/s^2 factor is common to all bodies within a
    clip and cancels). Sigma re-weights across CLIPS; only a body subset
    re-weights across BODIES. If this ever fails, DELTA 1 and DELTA 2 are not
    independent and the v64 attribution rule is void."""
    z = np.load(PERBODY_NPZ, allow_pickle=True)
    per_body, ok, weights = z["per_body_m"], ~z["fell"].astype(bool), z["body_weights"]
    e = z["weighted_mse"]
    shares = []
    for s in (0.03, 0.06, 0.15):
        gj = (
            (np.exp(-e / s**2)[:, None] / s**2)
            * 2
            * weights[None, :]
            * per_body
            / weights.sum()
        )
        shares.append(
            np.median(gj[ok][:, FOOT_BODY_INDICES].sum(1) / gj[ok].sum(1))
        )
    assert max(shares) - min(shares) < 1e-9, f"expected invariance, got {shares}"
    assert 0.40 < shares[0] < 0.43, "ankles should hold ~41.3% of the gradient"
