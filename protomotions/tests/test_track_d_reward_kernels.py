# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit smokes for the dormant Track D reward kernels.

REWORKED 2026-07-10 (OmniH2O code-audit economics fix). Covers:
- Swing-apex SHORTFALL penalty (shuffle steps cost, target-height steps free,
  touchdown-only emission, reset handling).
- Ref-motion-GATED displacement-per-step reward (dead-zone, cap,
  touchdown-only, stationary-reference gate, anchor behavior across gating).
- Continuous in_the_air penalty.
- Root xy displacement / heading exp-kernel rewards (task-reward channel).
- Factory construction (dormant weight=0.0 defaults).
"""

import math

import torch

from protomotions.envs.rewards.big_step import (
    FeetApexHeightReward,
    StepDisplacementReward,
    compute_in_the_air_penalty,
)
from protomotions.envs.rewards.tracking import (
    compute_root_heading_rew,
    compute_root_xy_displacement_rew,
)

NUM_ENVS = 2
NUM_BODIES = 4
FOOT_IDS = torch.tensor([2, 3])
GROUND = torch.zeros(NUM_ENVS)

# Reference body velocities [NUM_ENVS, NUM_BODIES, 3]: moving / stationary.
REF_VEL_MOVING = torch.zeros(NUM_ENVS, NUM_BODIES, 3)
REF_VEL_MOVING[:, 0, 0] = 1.0  # root moving 1 m/s in x
REF_VEL_STATIC = torch.zeros(NUM_ENVS, NUM_BODIES, 3)


def _positions(foot0_xy=(0.0, 0.0), foot0_z=0.0, foot1_xy=(0.0, 0.1), foot1_z=0.0):
    """Body positions [NUM_ENVS, NUM_BODIES, 3]; only env 0's feet move."""
    pos = torch.zeros(NUM_ENVS, NUM_BODIES, 3)
    pos[0, 2] = torch.tensor([foot0_xy[0], foot0_xy[1], foot0_z])
    pos[0, 3] = torch.tensor([foot1_xy[0], foot1_xy[1], foot1_z])
    return pos


def _contacts(f0: bool, f1: bool = True):
    """Contact flags [NUM_ENVS, NUM_BODIES]; env 1 always fully planted."""
    contacts = torch.zeros(NUM_ENVS, NUM_BODIES, dtype=torch.bool)
    contacts[:, FOOT_IDS] = True
    contacts[0, 2] = f0
    contacts[0, 3] = f1
    return contacts


def _yaw_quat(yaw: float) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def test_apex_shortfall_shuffle_costs_and_emits_only_at_touchdown():
    rew = FeetApexHeightReward(apex_target_height=0.25)
    progress = torch.tensor([1, 1])

    # Step 1: both feet planted (state init) -> 0.
    r = rew(_contacts(True), _positions(), GROUND, FOOT_IDS, progress)
    assert torch.equal(r, torch.zeros(NUM_ENVS))

    # Fake SHUFFLE swing of foot 0: apex only 0.05 m. Nothing while airborne.
    for z in (0.02, 0.05, 0.03):
        progress = progress + 1
        r = rew(_contacts(False), _positions(foot0_z=z), GROUND, FOOT_IDS, progress)
        assert torch.equal(r, torch.zeros(NUM_ENVS)), "no emission during swing"

    # Touchdown: shortfall 0.25 - 0.05 = 0.20 paid exactly once, env 0 only.
    progress = progress + 1
    r = rew(_contacts(True), _positions(foot0_z=0.0), GROUND, FOOT_IDS, progress)
    torch.testing.assert_close(r, torch.tensor([0.20, 0.0]))

    # Next step, still planted -> nothing more (standing costs nothing).
    progress = progress + 1
    r = rew(_contacts(True), _positions(), GROUND, FOOT_IDS, progress)
    assert torch.equal(r, torch.zeros(NUM_ENVS))


def test_apex_shortfall_zero_at_or_above_target():
    rew = FeetApexHeightReward(apex_target_height=0.25)
    progress = torch.tensor([1, 1])
    rew(_contacts(True), _positions(), GROUND, FOOT_IDS, progress)

    # High step: apex 0.30 >= 0.25 target -> zero shortfall at touchdown.
    progress = progress + 1
    rew(_contacts(False), _positions(foot0_z=0.30), GROUND, FOOT_IDS, progress)
    progress = progress + 1
    r = rew(_contacts(True), _positions(foot0_z=0.0), GROUND, FOOT_IDS, progress)
    torch.testing.assert_close(r, torch.tensor([0.0, 0.0]))


def test_apex_shortfall_reset_clears_state_and_suppresses_touchdown():
    rew = FeetApexHeightReward(apex_target_height=0.25)
    progress = torch.tensor([5, 5])
    rew(_contacts(True), _positions(), GROUND, FOOT_IDS, progress)
    progress = progress + 1
    rew(_contacts(False), _positions(foot0_z=0.05), GROUND, FOOT_IDS, progress)

    # Env 0 resets mid-swing (progress does not advance): the landed-looking
    # transition must NOT charge the stale shortfall.
    progress = torch.tensor([1, 7])
    r = rew(_contacts(True), _positions(foot0_z=0.0), GROUND, FOOT_IDS, progress)
    assert torch.equal(r, torch.zeros(NUM_ENVS))

    # A fresh swing after the reset works normally (apex 0.07 -> 0.18 cost).
    progress = torch.tensor([2, 8])
    rew(_contacts(False), _positions(foot0_z=0.07), GROUND, FOOT_IDS, progress)
    progress = torch.tensor([3, 9])
    r = rew(_contacts(True), _positions(foot0_z=0.0), GROUND, FOOT_IDS, progress)
    torch.testing.assert_close(r, torch.tensor([0.18, 0.0]))


def test_apex_lift_pays_positive_capped_at_target_and_zero_for_stance():
    """v2 positive LIFT mode (imprint PR #119): a completed swing earns
    ``min(apex, target)/target`` at touchdown — a big high step pays the max
    (1.0), a shuffle pays proportionally small, a planted stance foot never
    lands and pays exactly 0. NEVER negative (no shortfall penalty)."""
    rew = FeetApexHeightReward(apex_target_height=0.18, reward_mode="lift")
    progress = torch.tensor([1, 1])

    # Step 1: both feet planted (state init) -> 0.
    r = rew(_contacts(True), _positions(), GROUND, FOOT_IDS, progress)
    assert torch.equal(r, torch.zeros(NUM_ENVS))

    # BIG HIGH step: apex 0.30 >= 0.18 target -> capped at max pay 1.0, once,
    # env 0 only; env 1 (planted stance) stays 0.
    for z in (0.10, 0.30, 0.20):
        progress = progress + 1
        r = rew(_contacts(False), _positions(foot0_z=z), GROUND, FOOT_IDS, progress)
        assert torch.equal(r, torch.zeros(NUM_ENVS)), "no emission during swing"
    progress = progress + 1
    r = rew(_contacts(True), _positions(foot0_z=0.0), GROUND, FOOT_IDS, progress)
    torch.testing.assert_close(r, torch.tensor([1.0, 0.0]))

    # Standing after the landing pays nothing (no touchdown).
    progress = progress + 1
    r = rew(_contacts(True), _positions(), GROUND, FOOT_IDS, progress)
    assert torch.equal(r, torch.zeros(NUM_ENVS))

    # SHUFFLE swing: apex only 0.045 -> small proportional pay 0.045/0.18=0.25,
    # never negative.
    progress = progress + 1
    r = rew(_contacts(False), _positions(foot0_z=0.045), GROUND, FOOT_IDS, progress)
    assert torch.equal(r, torch.zeros(NUM_ENVS))
    progress = progress + 1
    r = rew(_contacts(True), _positions(foot0_z=0.0), GROUND, FOOT_IDS, progress)
    torch.testing.assert_close(r, torch.tensor([0.25, 0.0]))


def test_apex_lift_reward_mode_validation():
    import pytest

    with pytest.raises(ValueError):
        FeetApexHeightReward(apex_target_height=0.18, reward_mode="bogus")


def _lift_big_high_step(rew, progress, ref_vel):
    """Drive one BIG HIGH swing of foot 0 (apex 0.30 >= 0.18 target -> pays the
    capped 1.0 when ungated) through to touchdown; return (reward, progress).
    ``ref_vel`` is threaded through so the LIFT gate can see the reference."""
    for z in (0.10, 0.30, 0.20):
        progress = progress + 1
        rew(_contacts(False), _positions(foot0_z=z), GROUND, FOOT_IDS, progress,
            ref_rigid_body_vel=ref_vel)
    progress = progress + 1
    r = rew(_contacts(True), _positions(foot0_z=0.0), GROUND, FOOT_IDS, progress,
            ref_rigid_body_vel=ref_vel)
    return r, progress


def test_apex_lift_gate_blocks_static_reference():
    """LIFT gate (min_ref_speed=0.05): a BIG HIGH step (would pay 1.0 ungated)
    against a STATIONARY reference earns exactly 0 — this is the march-in-place
    exploit close (imprint PR #119 step-in-place investigation)."""
    rew = FeetApexHeightReward(
        apex_target_height=0.18, reward_mode="lift", min_ref_speed=0.05
    )
    progress = torch.tensor([1, 1])
    rew(_contacts(True), _positions(), GROUND, FOOT_IDS, progress,
        ref_rigid_body_vel=REF_VEL_STATIC)
    r, _ = _lift_big_high_step(rew, progress, REF_VEL_STATIC)
    torch.testing.assert_close(r, torch.tensor([0.0, 0.0]))


def test_apex_lift_gate_open_when_reference_moving():
    """LIFT gate (min_ref_speed=0.05): the SAME big high step against a MOVING
    reference is unchanged — pays the full capped 1.0. The gate only removes
    stepping-in-place income; a genuine step is untouched."""
    rew = FeetApexHeightReward(
        apex_target_height=0.18, reward_mode="lift", min_ref_speed=0.05
    )
    progress = torch.tensor([1, 1])
    rew(_contacts(True), _positions(), GROUND, FOOT_IDS, progress,
        ref_rigid_body_vel=REF_VEL_MOVING)
    r, _ = _lift_big_high_step(rew, progress, REF_VEL_MOVING)
    torch.testing.assert_close(r, torch.tensor([1.0, 0.0]))


def test_apex_lift_gate_zero_min_ref_speed_is_byte_identical_to_ungated():
    """Regression guard for the LIVE gpu3202 run: min_ref_speed=0.0 (the default;
    launcher ships PM_STEP_LIFT_MIN_REF_SPEED=0) leaves the lift reward
    byte-identical to the pre-gate kernel — even under a STATIONARY reference and
    even with no ref-vel tensor supplied. Steps the SAME sequence through a
    gate-0.0 kernel (fed the static ref) and an ungated kernel (no ref) in
    lockstep and asserts exact equality at every emission."""
    gated0 = FeetApexHeightReward(
        apex_target_height=0.18, reward_mode="lift", min_ref_speed=0.0
    )
    ungated = FeetApexHeightReward(apex_target_height=0.18, reward_mode="lift")
    progress = torch.tensor([1, 1])

    def both(contacts, pos, p):
        rg = gated0(contacts, pos, GROUND, FOOT_IDS, p,
                    ref_rigid_body_vel=REF_VEL_STATIC)
        ru = ungated(contacts, pos, GROUND, FOOT_IDS, p)
        assert torch.equal(rg, ru), "min_ref_speed=0.0 must not alter emission"
        return rg

    both(_contacts(True), _positions(), progress)

    # Big high step under a STATIONARY ref: ungated pays 1.0, so gate-0.0 must
    # ALSO pay 1.0 (the gate is off) — not the 0.0 the 0.05 gate would give.
    for z in (0.10, 0.30, 0.20):
        progress = progress + 1
        both(_contacts(False), _positions(foot0_z=z), progress)
    progress = progress + 1
    r = both(_contacts(True), _positions(foot0_z=0.0), progress)
    torch.testing.assert_close(r, torch.tensor([1.0, 0.0]))

    # A following shuffle swing (apex 0.045 -> 0.045/0.18 = 0.25) also matches.
    progress = progress + 1
    both(_contacts(False), _positions(foot0_z=0.045), progress)
    progress = progress + 1
    r = both(_contacts(True), _positions(foot0_z=0.0), progress)
    torch.testing.assert_close(r, torch.tensor([0.25, 0.0]))


def test_step_displacement_reward_thresholds_and_caps_when_ref_moving():
    rew = StepDisplacementReward(min_step_length=0.1, reward_cap=0.5,
                                 min_ref_speed=0.1)
    progress = torch.tensor([1, 1])
    rew(_contacts(True), _positions(), FOOT_IDS, progress,
        ref_rigid_body_vel=REF_VEL_MOVING)

    def swing_and_land(start_progress, landing_xy, ref_vel):
        p = start_progress + 1
        rew(_contacts(False), _positions(foot0_xy=landing_xy, foot0_z=0.1),
            FOOT_IDS, p, ref_rigid_body_vel=ref_vel)
        p = p + 1
        r = rew(_contacts(True), _positions(foot0_xy=landing_xy), FOOT_IDS, p,
                ref_rigid_body_vel=ref_vel)
        return r, p

    # Nominal 0.4 m step with a MOVING reference: 0.4 - 0.1 = 0.3.
    r, progress = swing_and_land(progress, (0.4, 0.0), REF_VEL_MOVING)
    torch.testing.assert_close(r, torch.tensor([0.3, 0.0]))

    # Micro-step of 0.05 m: inside dead-zone -> 0.
    r, progress = swing_and_land(progress, (0.45, 0.0), REF_VEL_MOVING)
    torch.testing.assert_close(r, torch.tensor([0.0, 0.0]))

    # Lunge of 0.8 m: min(0.8 - 0.1, 0.5) = 0.5.
    r, progress = swing_and_land(progress, (1.25, 0.0), REF_VEL_MOVING)
    torch.testing.assert_close(r, torch.tensor([0.5, 0.0]))

    # Planted step after touchdown: no continuous emission.
    progress = progress + 1
    r = rew(_contacts(True), _positions(foot0_xy=(1.25, 0.0)), FOOT_IDS, progress,
            ref_rigid_body_vel=REF_VEL_MOVING)
    assert torch.equal(r, torch.zeros(NUM_ENVS))


def test_step_displacement_pays_nothing_on_stationary_reference():
    """The gate: a big step under a STATIONARY reference earns zero — no step
    income on frozen/stationary refs — and the touchdown anchor still moves
    so a later gated-open step is measured from its true previous touchdown."""
    rew = StepDisplacementReward(min_step_length=0.1, reward_cap=0.5,
                                 min_ref_speed=0.1)
    progress = torch.tensor([1, 1])
    rew(_contacts(True), _positions(), FOOT_IDS, progress,
        ref_rigid_body_vel=REF_VEL_STATIC)

    # 0.4 m step under a stationary ref -> 0.
    progress = progress + 1
    rew(_contacts(False), _positions(foot0_xy=(0.4, 0.0), foot0_z=0.1),
        FOOT_IDS, progress, ref_rigid_body_vel=REF_VEL_STATIC)
    progress = progress + 1
    r = rew(_contacts(True), _positions(foot0_xy=(0.4, 0.0)), FOOT_IDS, progress,
            ref_rigid_body_vel=REF_VEL_STATIC)
    torch.testing.assert_close(r, torch.tensor([0.0, 0.0]))

    # Reference starts moving; next 0.3 m step is measured from x=0.4 (the
    # anchored gated touchdown), not from x=0.
    progress = progress + 1
    rew(_contacts(False), _positions(foot0_xy=(0.7, 0.0), foot0_z=0.1),
        FOOT_IDS, progress, ref_rigid_body_vel=REF_VEL_MOVING)
    progress = progress + 1
    r = rew(_contacts(True), _positions(foot0_xy=(0.7, 0.0)), FOOT_IDS, progress,
            ref_rigid_body_vel=REF_VEL_MOVING)
    torch.testing.assert_close(r, torch.tensor([0.3 - 0.1, 0.0]))


def test_step_displacement_gate_disabled_with_zero_min_ref_speed():
    rew = StepDisplacementReward(min_step_length=0.1, reward_cap=0.5,
                                 min_ref_speed=0.0)
    progress = torch.tensor([1, 1])
    rew(_contacts(True), _positions(), FOOT_IDS, progress,
        ref_rigid_body_vel=REF_VEL_STATIC)
    progress = progress + 1
    rew(_contacts(False), _positions(foot0_xy=(0.4, 0.0), foot0_z=0.1),
        FOOT_IDS, progress, ref_rigid_body_vel=REF_VEL_STATIC)
    progress = progress + 1
    r = rew(_contacts(True), _positions(foot0_xy=(0.4, 0.0)), FOOT_IDS, progress,
            ref_rigid_body_vel=REF_VEL_STATIC)
    torch.testing.assert_close(r, torch.tensor([0.3, 0.0]))


def test_step_displacement_reset_reanchors_touchdown_position():
    rew = StepDisplacementReward(min_step_length=0.1, reward_cap=0.5)
    progress = torch.tensor([3, 3])
    rew(_contacts(True), _positions(), FOOT_IDS, progress,
        ref_rigid_body_vel=REF_VEL_MOVING)

    # Reset env 0: respawns with foot 0 at x=5.0. Stale-anchor distance not
    # paid; anchor moves to x=5.0.
    progress = torch.tensor([1, 4])
    r = rew(_contacts(True), _positions(foot0_xy=(5.0, 0.0)), FOOT_IDS, progress,
            ref_rigid_body_vel=REF_VEL_MOVING)
    assert torch.equal(r, torch.zeros(NUM_ENVS))

    progress = torch.tensor([2, 5])
    rew(_contacts(False), _positions(foot0_xy=(5.2, 0.0), foot0_z=0.1),
        FOOT_IDS, progress, ref_rigid_body_vel=REF_VEL_MOVING)
    progress = torch.tensor([3, 6])
    r = rew(_contacts(True), _positions(foot0_xy=(5.3, 0.0)), FOOT_IDS, progress,
            ref_rigid_body_vel=REF_VEL_MOVING)
    torch.testing.assert_close(r, torch.tensor([0.3 - 0.1, 0.0]))


def test_in_the_air_penalty_continuous_indicator():
    # Both feet airborne -> 1.0 (env 0); env 1 planted -> 0.
    r = compute_in_the_air_penalty(_contacts(False, False), FOOT_IDS)
    torch.testing.assert_close(r, torch.tensor([1.0, 0.0]))
    # One foot down -> 0.
    r = compute_in_the_air_penalty(_contacts(False, True), FOOT_IDS)
    torch.testing.assert_close(r, torch.tensor([0.0, 0.0]))
    r = compute_in_the_air_penalty(_contacts(True, True), FOOT_IDS)
    torch.testing.assert_close(r, torch.tensor([0.0, 0.0]))


def test_compute_root_xy_displacement_rew():
    current_root_pos = torch.tensor([[0.0, 0.0, 0.9], [1.0, 2.0, 0.9]])
    ref = torch.zeros(2, NUM_BODIES, 3)
    ref[1, 0, :2] = torch.tensor([1.0, 2.0])  # env 1 tracks perfectly
    ref[0, 0, :2] = torch.tensor([0.3, 0.4])  # env 0 off by 0.5 m

    r = compute_root_xy_displacement_rew(current_root_pos, ref, coefficient=-20.0)
    expected_err = (0.3**2 + 0.4**2) / 2  # mean over xy of squared error
    torch.testing.assert_close(
        r, torch.tensor([math.exp(-20.0 * expected_err), 1.0])
    )


def test_compute_root_heading_rew_wraps_angle():
    cur = torch.stack([_yaw_quat(0.0), _yaw_quat(3.0)])
    ref = torch.zeros(2, NUM_BODIES, 4)
    ref[0, 0] = _yaw_quat(math.pi / 2)
    ref[1, 0] = _yaw_quat(-3.0)  # 2*pi - 6.0 away unwrapped, ~0.2832 wrapped

    r = compute_root_heading_rew(cur, ref, coefficient=-2.0)
    wrapped = 2 * math.pi - 6.0
    torch.testing.assert_close(
        r,
        torch.tensor(
            [math.exp(-2.0 * (math.pi / 2) ** 2), math.exp(-2.0 * wrapped**2)]
        ),
        atol=1e-5,
        rtol=1e-5,
    )
    # Perfect heading -> reward 1.
    perfect = compute_root_heading_rew(cur, cur.unsqueeze(1))
    torch.testing.assert_close(perfect, torch.ones(2))


def test_track_d_factories_are_dormant_by_default():
    from protomotions.envs.component_factories import (
        in_the_air_penalty_factory,
        max_feet_height_rew_factory,
        root_heading_rew_factory,
        root_xy_displacement_rew_factory,
        step_displacement_rew_factory,
    )
    from protomotions.envs.mdp_component import MdpComponent

    for factory in (
        root_xy_displacement_rew_factory,
        root_heading_rew_factory,
        max_feet_height_rew_factory,
        step_displacement_rew_factory,
        in_the_air_penalty_factory,
    ):
        component = factory()
        assert isinstance(component, MdpComponent)
        assert component.static_params["weight"] == 0.0

    apex = max_feet_height_rew_factory(weight=-8.0, apex_target_height=0.25)
    assert apex.compute_func.apex_target_height == 0.25
    assert apex.compute_func.reward_mode == "shortfall"  # back-compat default
    # v2 positive LIFT mode passes through (imprint PR #119).
    apex_lift = max_feet_height_rew_factory(
        weight=1.5, apex_target_height=0.18, reward_mode="lift"
    )
    assert apex_lift.compute_func.reward_mode == "lift"
    assert apex_lift.compute_func.apex_target_height == 0.18
    assert apex_lift.static_params["weight"] == 1.5
    # LIFT ref-speed gate defaults OFF (0.0 = ungated) and passes through when set
    # (imprint PR #119 step-in-place guard).
    assert apex_lift.compute_func.min_ref_speed == 0.0
    apex_lift_gated = max_feet_height_rew_factory(
        weight=1.5, apex_target_height=0.18, reward_mode="lift", min_ref_speed=0.05
    )
    assert apex_lift_gated.compute_func.min_ref_speed == 0.05
    assert set(apex.dynamic_vars) == {
        "sim_contacts",
        "rigid_body_pos",
        "ground_heights",
        "contact_body_ids",
        "progress_buf",
        "ref_rigid_body_vel",
    }

    step = step_displacement_rew_factory(
        min_step_length=0.2, reward_cap=0.6, min_ref_speed=0.15
    )
    assert step.compute_func.min_step_length == 0.2
    assert step.compute_func.reward_cap == 0.6
    assert step.compute_func.min_ref_speed == 0.15
    assert "ref_rigid_body_vel" in step.dynamic_vars


def test_graced_action_smoothness_zeroes_masked_envs():
    from protomotions.envs.rewards.regularization import (
        compute_action_smoothness,
        compute_action_smoothness_graced,
    )

    cur = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    prev = torch.zeros(2, 2)
    base = compute_action_smoothness(cur, prev)
    # No mask -> identical to stock.
    torch.testing.assert_close(
        compute_action_smoothness_graced(cur, prev, None), base)
    # Masked env 0 pays nothing; env 1 unchanged.
    graced = compute_action_smoothness_graced(
        cur, prev, torch.tensor([True, False]))
    torch.testing.assert_close(graced[0], torch.tensor(0.0))
    torch.testing.assert_close(graced[1], base[1])


def test_graced_action_smoothness_factory_dormantable():
    from protomotions.envs.component_factories import (
        graced_action_smoothness_factory,
    )

    comp = graced_action_smoothness_factory(weight=-0.1)
    assert comp.static_params["weight"] == -0.1
    assert "perturbation_grace_mask" in comp.dynamic_vars
