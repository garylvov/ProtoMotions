# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit smokes for the dormant Track D reward kernels.

Covers:
- OmniH2O-style per-step max-feet-height reward (apex tracked across a fake
  swing, touchdown-only emission, cap).
- Displacement-per-step reward (micro-step dead-zone, cap, touchdown-only).
- Episode-reset state handling via progress_buf.
- Root xy displacement / heading exp-kernel rewards (Option-B fallback).
- Factory construction (dormant weight=0.0 defaults).
"""

import math

import torch

from protomotions.envs.rewards.big_step import (
    FeetApexHeightReward,
    StepDisplacementReward,
)
from protomotions.envs.rewards.tracking import (
    compute_root_heading_rew,
    compute_root_xy_displacement_rew,
)

NUM_ENVS = 2
NUM_BODIES = 4
FOOT_IDS = torch.tensor([2, 3])
GROUND = torch.zeros(NUM_ENVS)


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


def test_feet_apex_height_reward_tracks_apex_and_emits_only_at_touchdown():
    rew = FeetApexHeightReward(apex_height_cap=0.15)
    progress = torch.tensor([1, 1])

    # Step 1: both feet planted (state init) -> 0.
    r = rew(_contacts(True), _positions(), GROUND, FOOT_IDS, progress)
    assert torch.equal(r, torch.zeros(NUM_ENVS))

    # Fake swing of foot 0: rises to 0.12 then descends. No reward while airborne.
    for z in (0.05, 0.12, 0.08):
        progress = progress + 1
        r = rew(_contacts(False), _positions(foot0_z=z), GROUND, FOOT_IDS, progress)
        assert torch.equal(r, torch.zeros(NUM_ENVS)), "no reward during swing"

    # Touchdown: apex (0.12, below cap) paid exactly once, env 0 only.
    progress = progress + 1
    r = rew(_contacts(True), _positions(foot0_z=0.0), GROUND, FOOT_IDS, progress)
    torch.testing.assert_close(r, torch.tensor([0.12, 0.0]))

    # Next step, still planted -> nothing more.
    progress = progress + 1
    r = rew(_contacts(True), _positions(), GROUND, FOOT_IDS, progress)
    assert torch.equal(r, torch.zeros(NUM_ENVS))


def test_feet_apex_height_reward_caps_apex():
    rew = FeetApexHeightReward(apex_height_cap=0.15)
    progress = torch.tensor([1, 1])
    rew(_contacts(True), _positions(), GROUND, FOOT_IDS, progress)

    progress = progress + 1
    rew(_contacts(False), _positions(foot0_z=0.30), GROUND, FOOT_IDS, progress)
    progress = progress + 1
    r = rew(_contacts(True), _positions(foot0_z=0.0), GROUND, FOOT_IDS, progress)
    torch.testing.assert_close(r, torch.tensor([0.15, 0.0]))


def test_feet_apex_height_reward_reset_clears_state_and_suppresses_touchdown():
    rew = FeetApexHeightReward(apex_height_cap=0.15)
    progress = torch.tensor([5, 5])
    rew(_contacts(True), _positions(), GROUND, FOOT_IDS, progress)
    progress = progress + 1
    rew(_contacts(False), _positions(foot0_z=0.12), GROUND, FOOT_IDS, progress)

    # Env 0 resets mid-swing (progress does not advance): the landed-looking
    # transition must NOT pay the stale apex.
    progress = torch.tensor([1, 7])
    r = rew(_contacts(True), _positions(foot0_z=0.0), GROUND, FOOT_IDS, progress)
    assert torch.equal(r, torch.zeros(NUM_ENVS))

    # A fresh swing after the reset works normally.
    progress = torch.tensor([2, 8])
    rew(_contacts(False), _positions(foot0_z=0.07), GROUND, FOOT_IDS, progress)
    progress = torch.tensor([3, 9])
    r = rew(_contacts(True), _positions(foot0_z=0.0), GROUND, FOOT_IDS, progress)
    torch.testing.assert_close(r, torch.tensor([0.07, 0.0]))


def test_step_displacement_reward_thresholds_and_caps():
    rew = StepDisplacementReward(min_step_length=0.1, reward_cap=0.5)
    progress = torch.tensor([1, 1])
    rew(_contacts(True), _positions(), FOOT_IDS, progress)

    def swing_and_land(start_progress, landing_xy):
        p = start_progress + 1
        rew(_contacts(False), _positions(foot0_xy=landing_xy, foot0_z=0.1), FOOT_IDS, p)
        p = p + 1
        r = rew(_contacts(True), _positions(foot0_xy=landing_xy), FOOT_IDS, p)
        return r, p

    # Nominal 0.4 m step: reward 0.4 - 0.1 = 0.3, env 0 only, at touchdown only.
    r, progress = swing_and_land(progress, (0.4, 0.0))
    torch.testing.assert_close(r, torch.tensor([0.3, 0.0]))

    # Micro-step of 0.05 m (from x=0.4 to 0.45): inside dead-zone -> 0.
    r, progress = swing_and_land(progress, (0.45, 0.0))
    torch.testing.assert_close(r, torch.tensor([0.0, 0.0]))

    # Lunge of 0.8 m: reward min(0.8 - 0.1, 0.5) = 0.5.
    r, progress = swing_and_land(progress, (1.25, 0.0))
    torch.testing.assert_close(r, torch.tensor([0.5, 0.0]))

    # Planted step after touchdown: no continuous emission.
    progress = progress + 1
    r = rew(_contacts(True), _positions(foot0_xy=(1.25, 0.0)), FOOT_IDS, progress)
    assert torch.equal(r, torch.zeros(NUM_ENVS))


def test_step_displacement_reward_reset_reanchors_touchdown_position():
    rew = StepDisplacementReward(min_step_length=0.1, reward_cap=0.5)
    progress = torch.tensor([3, 3])
    rew(_contacts(True), _positions(), FOOT_IDS, progress)

    # Reset env 0: it respawns with foot 0 at x=5.0. Distance from the stale
    # anchor (x=0) must not be paid, and the anchor must move to x=5.0.
    progress = torch.tensor([1, 4])
    r = rew(_contacts(True), _positions(foot0_xy=(5.0, 0.0)), FOOT_IDS, progress)
    assert torch.equal(r, torch.zeros(NUM_ENVS))

    progress = torch.tensor([2, 5])
    rew(_contacts(False), _positions(foot0_xy=(5.2, 0.0), foot0_z=0.1), FOOT_IDS, progress)
    progress = torch.tensor([3, 6])
    r = rew(_contacts(True), _positions(foot0_xy=(5.3, 0.0)), FOOT_IDS, progress)
    torch.testing.assert_close(r, torch.tensor([0.3 - 0.1, 0.0]))


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
    ):
        component = factory()
        assert isinstance(component, MdpComponent)
        assert component.static_params["weight"] == 0.0

    apex = max_feet_height_rew_factory(weight=0.1, apex_height_cap=0.2)
    assert apex.compute_func.apex_height_cap == 0.2
    assert set(apex.dynamic_vars) == {
        "sim_contacts",
        "rigid_body_pos",
        "ground_heights",
        "contact_body_ids",
        "progress_buf",
    }

    step = step_displacement_rew_factory(min_step_length=0.2, reward_cap=0.6)
    assert step.compute_func.min_step_length == 0.2
    assert step.compute_func.reward_cap == 0.6
