# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-body weighting on relative_body_pos/ori tracking rewards."""

import torch

from protomotions.envs.rewards.tracking import (
    compute_relative_body_pos_rew,
    compute_relative_body_ori_rew,
)
from protomotions.envs.component_factories import (
    relative_body_pos_rew_factory,
    relative_body_ori_rew_factory,
    _resolve_body_indices_and_weights,
)
from protomotions.utils import rotations


BODY_NAMES = ["pelvis", "left_wrist_link", "right_wrist_link", "left_ankle_link"]


def _identity_quat(*shape):
    q = torch.zeros(*shape, 4)
    q[..., 3] = 1.0
    return q


def _base_scene(num_envs=2, num_bodies=4):
    """Anchor (body 0) tracked perfectly; other bodies have small position/rot error."""
    torch.manual_seed(0)
    current_anchor_pos = torch.zeros(num_envs, 3)
    current_anchor_rot = _identity_quat(num_envs)

    current_rigid_body_pos = torch.zeros(num_envs, num_bodies, 3)
    ref_rigid_body_pos = torch.zeros(num_envs, num_bodies, 3)
    current_rigid_body_rot = _identity_quat(num_envs, num_bodies)
    ref_rigid_body_rot = _identity_quat(num_envs, num_bodies)

    return (
        current_rigid_body_pos,
        ref_rigid_body_pos,
        current_anchor_rot,
        ref_rigid_body_rot,
        current_anchor_pos,
        current_rigid_body_rot,
    )


def test_position_reward_uniform_weights_matches_current_behavior():
    """Regression: body_weights=None (default) must be numerically identical
    to the pre-existing uniform-mean behavior, with or without body_indices."""
    (
        current_pos,
        ref_pos,
        current_anchor_rot,
        ref_rot,
        current_anchor_pos,
        _,
    ) = _base_scene()
    ref_pos = current_pos.clone()
    ref_pos[:, 1] += 0.1  # wrist (index 1) mistracked
    ref_pos[:, 3] += 0.2  # ankle (index 3) mistracked

    baseline = compute_relative_body_pos_rew(
        current_pos, ref_pos, current_anchor_rot, ref_rot, current_anchor_pos,
        anchor_idx=0, sigma=0.3,
    )
    uniform_via_weights = compute_relative_body_pos_rew(
        current_pos, ref_pos, current_anchor_rot, ref_rot, current_anchor_pos,
        anchor_idx=0, sigma=0.3, body_indices=None, body_weights=None,
    )
    torch.testing.assert_close(baseline, uniform_via_weights)

    # Explicit equal weights across the same subset as body_indices reproduces
    # the uniform-mean-over-subset result.
    subset_uniform = compute_relative_body_pos_rew(
        current_pos, ref_pos, current_anchor_rot, ref_rot, current_anchor_pos,
        anchor_idx=0, sigma=0.3, body_indices=[1, 2, 3],
    )
    subset_equal_weights = compute_relative_body_pos_rew(
        current_pos, ref_pos, current_anchor_rot, ref_rot, current_anchor_pos,
        anchor_idx=0, sigma=0.3, body_indices=[1, 2, 3], body_weights=[1.0, 1.0, 1.0],
    )
    torch.testing.assert_close(subset_uniform, subset_equal_weights)


def test_position_reward_upweighted_wrist_moves_reward_toward_wrist_error():
    """Upweighting a mistracked body should pull the (unweighted) reward down
    further than uniform weighting, and pull it up when the upweighted body
    is well-tracked while others are not."""
    (
        current_pos,
        _,
        current_anchor_rot,
        ref_rot,
        current_anchor_pos,
        _,
    ) = _base_scene()

    ref_pos = current_pos.clone()
    ref_pos[:, 1] += 0.5  # left wrist badly mistracked
    # bodies 2, 3 tracked perfectly

    uniform = compute_relative_body_pos_rew(
        current_pos, ref_pos, current_anchor_rot, ref_rot, current_anchor_pos,
        anchor_idx=0, sigma=0.3, body_indices=[1, 2, 3],
    )
    wrist_upweighted = compute_relative_body_pos_rew(
        current_pos, ref_pos, current_anchor_rot, ref_rot, current_anchor_pos,
        anchor_idx=0, sigma=0.3, body_indices=[1, 2, 3], body_weights=[10.0, 1.0, 1.0],
    )
    # Upweighting the mistracked wrist should make the reward worse (lower),
    # since its large error now dominates the weighted mean.
    assert torch.all(wrist_upweighted < uniform)

    # Conversely, downweighting the mistracked wrist should improve the reward.
    wrist_downweighted = compute_relative_body_pos_rew(
        current_pos, ref_pos, current_anchor_rot, ref_rot, current_anchor_pos,
        anchor_idx=0, sigma=0.3, body_indices=[1, 2, 3], body_weights=[0.01, 1.0, 1.0],
    )
    assert torch.all(wrist_downweighted > uniform)


def test_orientation_reward_uniform_and_upweighted():
    num_envs, num_bodies = 2, 4
    current_rot = _identity_quat(num_envs, num_bodies)
    ref_rot = _identity_quat(num_envs, num_bodies)
    current_anchor_rot = _identity_quat(num_envs)

    # Rotate wrist (idx 1) by a noticeable angle in the reference.
    angle = torch.full((num_envs,), 0.6)
    axis = torch.zeros(num_envs, 3)
    axis[:, 2] = 1.0
    ref_rot = ref_rot.clone()
    ref_rot[:, 1] = rotations.quat_from_angle_axis(angle, axis, w_last=True)

    uniform = compute_relative_body_ori_rew(
        current_rot, ref_rot, current_anchor_rot, anchor_idx=0, sigma=0.4,
        body_indices=[1, 2, 3],
    )
    upweighted = compute_relative_body_ori_rew(
        current_rot, ref_rot, current_anchor_rot, anchor_idx=0, sigma=0.4,
        body_indices=[1, 2, 3], body_weights=[10.0, 1.0, 1.0],
    )
    assert torch.all(upweighted < uniform)

    default_none = compute_relative_body_ori_rew(
        current_rot, ref_rot, current_anchor_rot, anchor_idx=0, sigma=0.4,
    )
    explicit_none = compute_relative_body_ori_rew(
        current_rot, ref_rot, current_anchor_rot, anchor_idx=0, sigma=0.4,
        body_indices=None, body_weights=None,
    )
    torch.testing.assert_close(default_none, explicit_none)


def test_resolve_body_indices_and_weights_name_lookup():
    resolved = _resolve_body_indices_and_weights(
        body_indices=None,
        body_weights={"left_wrist_link": 3.0, "right_wrist_link": 3.0},
        body_names=BODY_NAMES,
    )
    assert resolved["body_indices"] == [1, 2]
    assert resolved["body_weights"] == [3.0, 3.0]

    # No body_weights -> passthrough of body_indices (or nothing at all).
    assert _resolve_body_indices_and_weights(None, None, None) == {}
    assert _resolve_body_indices_and_weights([0, 1], None, None) == {
        "body_indices": [0, 1]
    }


def test_resolve_body_indices_and_weights_errors():
    import pytest

    with pytest.raises(ValueError):
        _resolve_body_indices_and_weights(
            body_indices=[0], body_weights={"pelvis": 1.0}, body_names=BODY_NAMES
        )
    with pytest.raises(ValueError):
        _resolve_body_indices_and_weights(
            body_indices=None, body_weights={"pelvis": 1.0}, body_names=None
        )


def test_factories_plumb_body_weights_into_static_params():
    pos_component = relative_body_pos_rew_factory(
        body_weights={"left_wrist_link": 5.0}, body_names=BODY_NAMES
    )
    assert pos_component.static_params["body_indices"] == [1]
    assert pos_component.static_params["body_weights"] == [5.0]

    ori_component = relative_body_ori_rew_factory(
        body_weights={"right_wrist_link": 2.5}, body_names=BODY_NAMES
    )
    assert ori_component.static_params["body_indices"] == [2]
    assert ori_component.static_params["body_weights"] == [2.5]

    # Backward compatible default: no body_weights/body_indices keys at all.
    default_component = relative_body_pos_rew_factory()
    assert "body_weights" not in default_component.static_params
    assert "body_indices" not in default_component.static_params
