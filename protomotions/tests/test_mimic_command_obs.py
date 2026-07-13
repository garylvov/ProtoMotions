# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the heading-frame future-displacement command observation kernel."""

import torch

from protomotions.envs.obs.mimic_command import build_mimic_future_displacement_cmd
from protomotions.envs.obs.utils import heading_local_xyz_delta
from protomotions.envs.rewards.tracking import compute_heading_local_anchor_drift_rew
from protomotions.envs.component_factories import (
    mimic_future_displacement_cmd_factory,
    heading_local_anchor_drift_rew_factory,
)
from protomotions.utils import rotations


def _random_yaw_quat(num: int, generator: torch.Generator) -> torch.Tensor:
    angle = torch.rand(num, generator=generator) * 2 * torch.pi
    axis = torch.zeros(num, 3)
    axis[:, 2] = 1.0
    return rotations.quat_from_angle_axis(angle, axis, w_last=True)


def _random_quat(num: int, generator: torch.Generator) -> torch.Tensor:
    """Arbitrary (not just yaw) random unit quaternions, for origin/target rot inputs."""
    q = torch.randn(num, 4, generator=generator)
    return rotations.normalize(q)


def _make_inputs(num_envs: int, total_future_steps: int, generator: torch.Generator):
    current_pos = torch.randn(num_envs, 3, generator=generator)
    current_rot = _random_quat(num_envs, generator)
    ref_pos = torch.randn(num_envs, total_future_steps, 3, generator=generator)
    ref_rot = _random_quat(num_envs * total_future_steps, generator).reshape(
        num_envs, total_future_steps, 4
    )
    return current_pos, current_rot, ref_pos, ref_rot


def test_output_dims_with_and_without_heading_delta():
    generator = torch.Generator().manual_seed(0)
    num_envs, total_steps = 4, 3
    current_pos, current_rot, ref_pos, ref_rot = _make_inputs(
        num_envs, total_steps, generator
    )

    out_with_heading = build_mimic_future_displacement_cmd(
        current_pos, current_rot, ref_pos, ref_rot,
        future_steps=[1, 2], w_last=True, include_heading_delta=True,
    )
    assert out_with_heading.shape == (num_envs, 2 * 5)

    out_without_heading = build_mimic_future_displacement_cmd(
        current_pos, current_rot, ref_pos, ref_rot,
        future_steps=[1, 2], w_last=True, include_heading_delta=False,
    )
    assert out_without_heading.shape == (num_envs, 2 * 3)

    # xyz portion identical regardless of include_heading_delta
    xyz_with = out_with_heading.reshape(num_envs, 2, 5)[..., :3]
    xyz_without = out_without_heading.reshape(num_envs, 2, 3)
    torch.testing.assert_close(xyz_with, xyz_without)


def test_single_step_int_selector_matches_first_n():
    generator = torch.Generator().manual_seed(1)
    num_envs, total_steps = 3, 5
    current_pos, current_rot, ref_pos, ref_rot = _make_inputs(
        num_envs, total_steps, generator
    )

    out_int = build_mimic_future_displacement_cmd(
        current_pos, current_rot, ref_pos, ref_rot, future_steps=3, w_last=True,
    )
    out_list = build_mimic_future_displacement_cmd(
        current_pos, current_rot, ref_pos, ref_rot, future_steps=[1, 2, 3], w_last=True,
    )
    torch.testing.assert_close(out_int, out_list)


def test_z_delta_is_raw_and_preserved():
    """Z delta must be target_z - origin_z, unaffected by heading rotation."""
    num_envs = 6
    current_pos = torch.zeros(num_envs, 3)
    current_pos[:, 2] = torch.linspace(0.0, 1.0, num_envs)
    # Arbitrary heading for the origin: should not affect the z output.
    yaws = torch.linspace(0, 2 * torch.pi, num_envs)
    axis = torch.zeros(num_envs, 3)
    axis[:, 2] = 1.0
    current_rot = rotations.quat_from_angle_axis(yaws, axis, w_last=True)

    ref_pos = torch.zeros(num_envs, 1, 3)
    target_z = torch.linspace(-1.0, 2.0, num_envs)
    ref_pos[:, 0, 2] = target_z
    ref_rot = rotations.quat_identity([num_envs, 1], w_last=True)

    out = build_mimic_future_displacement_cmd(
        current_pos, current_rot, ref_pos, ref_rot,
        future_steps=1, w_last=True, include_heading_delta=False,
    )
    expected_z = target_z - current_pos[:, 2]
    torch.testing.assert_close(out[:, 2], expected_z)


def test_heading_invariance_under_global_yaw_rotation():
    """Rotating the whole world (all inputs) by a shared random yaw must not
    change the command output: it is expressed purely in the origin's own
    heading-local frame, so it is invariant to the world frame's heading."""
    generator = torch.Generator().manual_seed(42)
    num_envs, total_steps = 5, 4
    current_pos, current_rot, ref_pos, ref_rot = _make_inputs(
        num_envs, total_steps, generator
    )

    baseline = build_mimic_future_displacement_cmd(
        current_pos, current_rot, ref_pos, ref_rot,
        future_steps=[1, 3, 4], w_last=True, include_heading_delta=True,
    )

    world_yaw = _random_yaw_quat(1, generator).expand(num_envs, 4)

    rotated_current_pos = rotations.quat_rotate(world_yaw, current_pos, w_last=True)
    rotated_current_rot = rotations.quat_mul(world_yaw, current_rot, w_last=True)

    world_yaw_steps = world_yaw.unsqueeze(1).expand(-1, total_steps, -1).reshape(-1, 4)
    rotated_ref_pos = rotations.quat_rotate(
        world_yaw_steps, ref_pos.reshape(-1, 3), w_last=True
    ).reshape(num_envs, total_steps, 3)
    rotated_ref_rot = rotations.quat_mul(
        world_yaw_steps, ref_rot.reshape(-1, 4), w_last=True
    ).reshape(num_envs, total_steps, 4)

    rotated = build_mimic_future_displacement_cmd(
        rotated_current_pos, rotated_current_rot, rotated_ref_pos, rotated_ref_rot,
        future_steps=[1, 3, 4], w_last=True, include_heading_delta=True,
    )

    torch.testing.assert_close(baseline, rotated, atol=1e-4, rtol=1e-4)


def test_pure_delta_no_translation_leak():
    """Command must depend only on the relative offset, not on absolute world
    position (translating both origin and targets by the same vector leaves
    the command unchanged)."""
    generator = torch.Generator().manual_seed(7)
    num_envs, total_steps = 3, 2
    current_pos, current_rot, ref_pos, ref_rot = _make_inputs(
        num_envs, total_steps, generator
    )

    baseline = build_mimic_future_displacement_cmd(
        current_pos, current_rot, ref_pos, ref_rot, future_steps=2, w_last=True,
    )

    shift = torch.randn(num_envs, 3, generator=generator)
    shifted = build_mimic_future_displacement_cmd(
        current_pos + shift,
        current_rot,
        ref_pos + shift.unsqueeze(1),
        ref_rot,
        future_steps=2,
        w_last=True,
    )

    torch.testing.assert_close(baseline, shifted, atol=1e-5, rtol=1e-5)


def test_heading_local_xyz_delta_matches_kernel_xyz_component():
    generator = torch.Generator().manual_seed(3)
    num_envs = 4
    current_pos, current_rot, ref_pos, ref_rot = _make_inputs(num_envs, 1, generator)

    expected = heading_local_xyz_delta(
        current_pos, current_rot, ref_pos[:, 0], w_last=True
    )
    out = build_mimic_future_displacement_cmd(
        current_pos, current_rot, ref_pos, ref_rot,
        future_steps=1, w_last=True, include_heading_delta=False,
    )
    torch.testing.assert_close(out, expected)


def test_factory_constructs_and_binds_expected_paths():
    from protomotions.envs.context_views import EnvContext

    component = mimic_future_displacement_cmd_factory(future_steps=[8])
    assert component.compute_func is build_mimic_future_displacement_cmd
    assert component.static_params["future_steps"] == [8]
    assert component.static_params["include_heading_delta"] is True

    dynamic_vars = component.dynamic_vars
    assert dynamic_vars["current_state_anchor_pos"].path == "current.anchor_pos"
    assert dynamic_vars["current_state_anchor_rot"].path == "current.anchor_rot"
    assert dynamic_vars["mimic_ref_anchor_pos"].path == "mimic.future_anchor_pos"
    assert dynamic_vars["mimic_ref_anchor_rot"].path == "mimic.future_anchor_rot"

    noisy_component = mimic_future_displacement_cmd_factory(use_noisy=True)
    assert (
        noisy_component.dynamic_vars["current_state_anchor_pos"].path
        == "noisy.anchor_pos"
    )


# =============================================================================
# D3: heading_local_anchor_drift_rew (reward twin of the command observation)
# =============================================================================


def test_heading_local_anchor_drift_rew_zero_drift_gives_reward_one():
    generator = torch.Generator().manual_seed(11)
    num_envs = 5
    current_anchor_pos = torch.randn(num_envs, 3, generator=generator)
    current_anchor_rot = _random_quat(num_envs, generator)

    reward = compute_heading_local_anchor_drift_rew(
        current_anchor_pos, current_anchor_rot, current_anchor_pos.clone(), sigma=0.3,
    )
    torch.testing.assert_close(reward, torch.ones(num_envs))


def test_heading_local_anchor_drift_rew_falls_off_with_drift_and_sigma():
    num_envs = 4
    current_anchor_pos = torch.zeros(num_envs, 3)
    current_anchor_rot = rotations.quat_identity([num_envs], w_last=True)

    drift_mags = torch.tensor([0.0, 0.1, 0.3, 1.0])
    ref_anchor_pos = torch.zeros(num_envs, 3)
    ref_anchor_pos[:, 0] = drift_mags

    reward = compute_heading_local_anchor_drift_rew(
        current_anchor_pos, current_anchor_rot, ref_anchor_pos, sigma=0.3,
    )
    # Monotonically decreasing as drift grows.
    assert torch.all(reward[1:] < reward[:-1])
    assert reward[0].item() == 1.0

    # Larger sigma (wider kernel) gives a strictly higher reward for the same
    # nonzero drift.
    tight = compute_heading_local_anchor_drift_rew(
        current_anchor_pos, current_anchor_rot, ref_anchor_pos, sigma=0.1,
    )
    wide = compute_heading_local_anchor_drift_rew(
        current_anchor_pos, current_anchor_rot, ref_anchor_pos, sigma=1.0,
    )
    assert torch.all(wide[1:] > tight[1:])


def test_heading_local_anchor_drift_rew_matches_command_kernel_xyz():
    """The reward's drift term should equal ||heading_local_xyz_delta(...)||^2,
    i.e. it is the current-step special case of the future-displacement
    command's per-step xyz component."""
    generator = torch.Generator().manual_seed(21)
    num_envs = 4
    current_anchor_pos = torch.randn(num_envs, 3, generator=generator)
    current_anchor_rot = _random_quat(num_envs, generator)
    ref_anchor_pos = torch.randn(num_envs, 3, generator=generator)

    reward = compute_heading_local_anchor_drift_rew(
        current_anchor_pos, current_anchor_rot, ref_anchor_pos, sigma=0.5,
    )
    delta = heading_local_xyz_delta(
        current_anchor_pos, current_anchor_rot, ref_anchor_pos, w_last=True
    )
    expected = torch.exp(-delta.pow(2).sum(dim=-1) / (0.5 ** 2))
    torch.testing.assert_close(reward, expected)


def test_heading_local_anchor_drift_rew_factory_binds_expected_paths():
    from protomotions.envs.context_views import EnvContext

    component = heading_local_anchor_drift_rew_factory(weight=0.7, sigma=0.25)
    assert component.compute_func is compute_heading_local_anchor_drift_rew
    assert component.static_params["weight"] == 0.7
    assert component.static_params["sigma"] == 0.25

    dynamic_vars = component.dynamic_vars
    assert dynamic_vars["current_anchor_pos"].path == "current.anchor_pos"
    assert dynamic_vars["current_anchor_rot"].path == "current.anchor_rot"
    assert dynamic_vars["ref_anchor_pos"].path == "mimic.ref_anchor_pos"
