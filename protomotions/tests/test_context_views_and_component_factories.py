# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for typed context views and MDP component factories."""

from types import SimpleNamespace

import torch

from protomotions.envs import component_factories as factories
from protomotions.envs import context_views
from protomotions.envs.context_views import (
    CurrentStateView,
    EnvContext,
    HistoricalView,
    MaskedMimicContext,
    MimicContext,
    PathContext,
    SteeringContext,
    TargetContext,
)


def _identity_quat(*shape: int) -> torch.Tensor:
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


def _robot_state(num_envs: int = 2, num_bodies: int = 3):
    rigid_body_pos = torch.arange(num_envs * num_bodies * 3, dtype=torch.float).reshape(
        num_envs, num_bodies, 3
    )
    rigid_body_rot = _identity_quat(num_envs, num_bodies)
    rigid_body_vel = torch.ones(num_envs, num_bodies, 3)
    rigid_body_ang_vel = torch.ones(num_envs, num_bodies, 3) * 2.0
    return SimpleNamespace(
        rigid_body_pos=rigid_body_pos,
        rigid_body_rot=rigid_body_rot,
        rigid_body_vel=rigid_body_vel,
        rigid_body_ang_vel=rigid_body_ang_vel,
        rigid_body_contacts=torch.tensor(
            [[True, False, True], [False, True, False]]
        ),
        rigid_body_contact_forces=torch.ones(num_envs, num_bodies, 3) * 3.0,
        dof_pos=torch.ones(num_envs, 4),
        dof_vel=torch.ones(num_envs, 4) * 2.0,
        dof_forces=torch.ones(num_envs, 4) * 3.0,
        local_rigid_body_rot=rigid_body_rot.clone(),
        root_pos=rigid_body_pos[:, 0, :],
        root_rot=rigid_body_rot[:, 0, :],
        root_vel=rigid_body_vel[:, 0, :],
        root_ang_vel=rigid_body_ang_vel[:, 0, :],
    )


def test_current_and_historical_views_precompute_accessors(monkeypatch):
    monkeypatch.setattr(
        context_views,
        "compute_local_ang_vel",
        lambda rot, ang_vel: ang_vel + 10.0,
    )
    state = _robot_state()

    current = CurrentStateView(state, anchor_idx=2)

    assert torch.equal(current.root_pos, state.root_pos)
    assert torch.equal(current.root_height, state.rigid_body_pos[:, 0, 2])
    assert torch.equal(current.anchor_pos, state.rigid_body_pos[:, 2, :])
    assert torch.equal(current.anchor_local_ang_vel, state.rigid_body_ang_vel[:, 2, :] + 10.0)
    assert EnvContext.current.anchor_pos.path == "current.anchor_pos"
    assert EnvContext.mimic.ref_state.rigid_body_pos.path == "mimic.ref_state.rigid_body_pos"

    buffer = SimpleNamespace(
        historical_rigid_body_pos=torch.ones(2, 2, 3, 3),
        historical_rigid_body_rot=_identity_quat(2, 2, 3),
        historical_rigid_body_vel=torch.ones(2, 2, 3, 3) * 2.0,
        historical_rigid_body_ang_vel=torch.ones(2, 2, 3, 3) * 3.0,
        historical_dof_pos=torch.ones(2, 2, 4),
        historical_dof_vel=torch.ones(2, 2, 4) * 4.0,
        historical_actions=torch.ones(2, 2, 5),
        historical_processed_actions=torch.ones(2, 2, 5) * 2.0,
        historical_ground_heights=torch.ones(2, 2) * 0.5,
        historical_body_contacts=torch.ones(2, 2, 3, dtype=torch.bool),
        historical_root_pos=torch.ones(2, 2, 3) * 5.0,
        historical_root_rot=_identity_quat(2, 2),
        historical_root_ang_vel=torch.ones(2, 2, 3) * 6.0,
        historical_anchor_pos=torch.ones(2, 2, 3) * 7.0,
        historical_anchor_rot=_identity_quat(2, 2),
        historical_anchor_vel=torch.ones(2, 2, 3) * 8.0,
        historical_anchor_ang_vel=torch.ones(2, 2, 3) * 9.0,
        noisy_historical_rigid_body_pos=torch.ones(2, 2, 3, 3) * -1.0,
        noisy_historical_rigid_body_rot=_identity_quat(2, 2, 3),
        noisy_historical_rigid_body_vel=torch.ones(2, 2, 3, 3) * -2.0,
        noisy_historical_rigid_body_ang_vel=torch.ones(2, 2, 3, 3) * -3.0,
        noisy_historical_dof_pos=torch.ones(2, 2, 4) * -4.0,
        noisy_historical_dof_vel=torch.ones(2, 2, 4) * -5.0,
        noisy_historical_ground_heights=torch.ones(2, 2) * -0.5,
        noisy_historical_root_pos=torch.ones(2, 2, 3) * -6.0,
        noisy_historical_root_rot=_identity_quat(2, 2),
        noisy_historical_root_ang_vel=torch.ones(2, 2, 3) * -7.0,
        noisy_historical_anchor_pos=torch.ones(2, 2, 3) * -8.0,
        noisy_historical_anchor_rot=_identity_quat(2, 2),
    )
    clean_history = HistoricalView(buffer, use_noisy=False)
    noisy_history = HistoricalView(buffer, use_noisy=True)

    assert torch.equal(clean_history.actions, buffer.historical_actions)
    assert torch.equal(clean_history.body_contacts, buffer.historical_body_contacts)
    assert torch.equal(clean_history.root_local_ang_vel, buffer.historical_root_ang_vel + 10.0)
    assert torch.equal(clean_history.anchor_ang_vel, buffer.historical_anchor_ang_vel)
    assert torch.equal(noisy_history.rigid_body_pos, buffer.noisy_historical_rigid_body_pos)
    assert noisy_history.actions is None
    assert noisy_history.anchor_vel is None


def test_control_contexts_and_env_context_store_optional_views():
    ref_state = _robot_state()
    future_pos = torch.ones(2, 4, 3, 3)
    future_rot = _identity_quat(2, 4, 3)
    future_vel = torch.ones(2, 4, 3, 3) * 2.0
    future_ang_vel = torch.ones(2, 4, 3, 3) * 3.0
    future_dof_pos = torch.ones(2, 4, 5)
    future_dof_vel = torch.ones(2, 4, 5) * 4.0
    mimic = MimicContext(
        ref_state=ref_state,
        future_pos=future_pos,
        future_rot=future_rot,
        future_vel=future_vel,
        future_ang_vel=future_ang_vel,
        future_dof_pos=future_dof_pos,
        future_dof_vel=future_dof_vel,
        anchor_idx=1,
        ref_lr=torch.ones(2, 5),
    )
    masked = MaskedMimicContext(
        mimic=mimic,
        ref_pos=future_pos,
        ref_rot=future_rot,
        target_times=torch.ones(2, 4),
        time_offsets=torch.arange(8, dtype=torch.float).reshape(2, 4),
        target_poses_masks=torch.ones(2, 4),
        target_bodies_masks=torch.ones(2, 4 * 3 * 2),
    )
    steering = SteeringContext(
        tar_dir=torch.ones(2, 2),
        tar_dir_theta=torch.ones(2),
        tar_speed=torch.ones(2) * 3.0,
        tar_face_dir=torch.ones(2, 2) * -1.0,
        prev_root_pos=torch.zeros(2, 3),
    )
    path = PathContext(
        tar_pos=torch.ones(2, 3),
        head_pos=torch.ones(2, 3) * 2.0,
        traj_samples=torch.ones(2, 5, 3),
        height_conditioned=True,
        head_body_id=2,
        progress_buf=torch.tensor([1, 2]),
    )
    target = TargetContext(torch.ones(2, 3), tar_proximity_threshold=0.5)
    current = CurrentStateView(ref_state, anchor_idx=1)
    env_context = EnvContext(
        current=current,
        noisy=current,
        dt=1.0 / 60.0,
        previous_action=torch.ones(2, 5),
        current_processed_action=torch.ones(2, 5) * 2.0,
        previous_processed_action=torch.ones(2, 5) * 3.0,
        ground_heights=torch.zeros(2),
        noisy_ground_heights=torch.ones(2),
        body_contacts=torch.ones(2, 3, dtype=torch.bool),
        current_contact_force_magnitudes=torch.ones(2, 3),
        prev_contact_force_magnitudes=torch.ones(2, 3) * 2.0,
        progress_buf=torch.tensor([4, 5]),
        contact_body_ids=torch.tensor([0, 2]),
        non_termination_contact_body_ids=torch.tensor([1]),
        odom_scale=torch.ones(2),
        odom_yaw_cos_sin=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        mimic=mimic,
        masked_mimic=masked,
        steering=steering,
        path=path,
        target=target,
    )

    assert torch.equal(mimic.future_root_pos, future_pos[:, :, 0, :])
    assert torch.equal(mimic.ref_anchor_pos, ref_state.rigid_body_pos[:, 1, :])
    assert torch.equal(masked.time_offsets, torch.arange(8, dtype=torch.float).reshape(2, 4))
    assert env_context.masked_mimic.mimic is mimic
    assert env_context.steering.tar_speed.tolist() == [3.0, 3.0]
    assert env_context.path.height_conditioned is True
    assert env_context.target.tar_proximity_threshold == 0.5


def _bindings(component):
    return component.get_bindings_dict()


def _params(component):
    return component.get_params()


def test_observation_factories_bind_expected_context_paths():
    max_coords = factories.max_coords_obs_factory(
        use_noisy=True,
        local_obs=False,
        root_height_obs=False,
        observe_contacts=True,
    )
    assert _bindings(max_coords)["body_pos"] == "noisy.rigid_body_pos"
    assert _bindings(max_coords)["ground_height"] == "noisy_ground_heights"
    assert _params(max_coords)["observe_contacts"] is True
    assert _params(max_coords)["local_obs"] is False

    reduced = factories.reduced_coords_obs_factory(
        root_height_obs=True,
        root_vel_obs=True,
    )
    assert _bindings(reduced)["root_pos"] == "current.root_pos"
    assert _bindings(reduced)["root_vel"] == "current.root_vel"

    hist_max = factories.historical_max_coords_obs_factory(
        use_noisy=True,
        observe_contacts=True,
        history_steps=[1, 3],
    )
    assert _bindings(hist_max)["historical_rigid_body_pos"] == "noisy_historical.rigid_body_pos"
    assert _params(hist_max)["history_steps"] == [1, 3]

    hist_reduced = factories.historical_reduced_coords_obs_factory(use_noisy=False)
    assert _bindings(hist_reduced)["historical_dof_pos"] == "historical.dof_pos"

    previous = factories.previous_actions_factory(history_steps=3, processed=True)
    assert _bindings(previous) == {"historical_actions": "historical.processed_actions"}
    assert _params(previous)["history_steps"] == 3

    max_target = factories.mimic_target_poses_max_coords_factory(
        use_noisy=True,
        with_velocities=False,
        with_relative=False,
        future_steps=2,
    )
    assert _bindings(max_target)["current_state_body_pos"] == "noisy.rigid_body_pos"
    assert _params(max_target)["future_steps"] == 2
    assert _params(max_target)["with_velocities"] is False

    future_rel = factories.mimic_target_poses_future_rel_factory(
        use_noisy=True,
        future_steps=4,
    )
    assert _bindings(future_rel)["current_state_body_rot"] == "noisy.rigid_body_rot"
    assert _params(future_rel)["future_steps"] == 4

    reduced_target = factories.mimic_target_poses_reduced_coords_factory(
        include_xy_offset=True,
        include_height=True,
        include_anchor_vel=True,
        include_anchor_ang_vel=True,
        zero_xy_offset=True,
    )
    assert _bindings(reduced_target)["mimic_ref_anchor_pos"] == "mimic.future_anchor_pos"
    assert _params(reduced_target)["zero_xy_offset"] is True

    deploy = factories.mimic_deploy_target_poses_factory(
        use_noisy=True,
        include_dof_vel=False,
        future_steps=[1, 4],
    )
    assert _bindings(deploy)["current_anchor_rot"] == "noisy.anchor_rot"
    assert _params(deploy)["include_dof_vel"] is False

    assert _bindings(factories.target_obs_factory())["tar_pos"] == "target.tar_pos"
    assert _bindings(factories.steering_obs_factory())["tar_speed"] == "steering.tar_speed"
    assert _bindings(factories.path_obs_factory())["traj_samples"] == "path.traj_samples"


def test_reward_factories_and_bundles_bind_expected_context_paths():
    smooth = factories.action_smoothness_factory(weight=-0.5)
    assert _bindings(smooth)["current_processed_action"] == "current_processed_action"
    assert _params(smooth)["weight"] == -0.5

    bundle = factories.mimic_tracking_rewards_factory(
        gt_weight=1.0,
        gr_weight=2.0,
        gv_weight=3.0,
        gav_weight=4.0,
        rh_weight=5.0,
    )
    assert set(bundle) == {"gt_rew", "gr_rew", "gv_rew", "gav_rew", "rh_rew"}
    assert _params(bundle["gt_rew"])["weight"] == 1.0
    assert _params(bundle["rh_rew"])["weight"] == 5.0

    rel = factories.gt_rel_rew_factory(body_indices=[0, 2])
    assert _bindings(rel)["anchor_idx"] == "mimic.anchor_idx"
    assert _params(rel)["body_indices"] == [0, 2]

    anchor_xy = factories.anchor_xy_rew_factory(weight=0.7, coefficient=-3.0)
    assert _bindings(anchor_xy)["current_anchor_pos"] == "current.anchor_pos"
    assert _params(anchor_xy) == {"weight": 0.7, "coefficient": -3.0}

    corrupted = factories.corrupted_xy_offset_factory(
        log_noise_std=0.2,
        soft_threshold=0.4,
    )
    assert _bindings(corrupted)["odom_yaw_cos_sin"] == "odom_yaw_cos_sin"
    assert _params(corrupted)["log_noise_std"] == 0.2

    power = factories.pow_rew_factory(
        weight=-2.0,
        min_value=None,
        use_torque_squared=True,
    )
    assert _bindings(power)["dof_forces"] == "current.dof_forces"
    assert "min_value" not in _params(power)
    assert _params(power)["use_torque_squared"] is True
    default_power = factories.pow_rew_factory(min_value=-0.25)
    assert _params(default_power)["min_value"] == -0.25

    contact_match = factories.contact_match_rew_factory(
        weight=-0.4,
        zero_during_grace_period=False,
    )
    assert _bindings(contact_match)["ref_contacts"] == "mimic.ref_state.rigid_body_contacts"
    assert _params(contact_match)["zero_during_grace_period"] is False

    force_change = factories.contact_force_change_rew_factory(
        min_value=None,
        threshold=5.0,
    )
    assert _bindings(force_change)["prev_contact_force_magnitudes"] == "prev_contact_force_magnitudes"
    assert "min_value" not in _params(force_change)
    default_force_change = factories.contact_force_change_rew_factory(min_value=-0.75)
    assert _params(default_force_change)["min_value"] == -0.75

    assert _bindings(factories.target_reward_factory())["tar_proximity_threshold"] == "target.tar_proximity_threshold"
    assert _bindings(factories.steering_reward_factory())["dt"] == "dt"
    assert _bindings(factories.path_following_reward_factory())["height_conditioned"] == "path.height_conditioned"

    for factory_fn in [
        factories.global_anchor_pos_rew_factory,
        factories.global_anchor_ori_rew_factory,
        factories.relative_body_pos_rew_factory,
        factories.relative_body_ori_rew_factory,
        factories.global_body_lin_vel_rew_factory,
        factories.global_body_ang_vel_rew_factory,
    ]:
        component = factory_fn(weight=0.9, sigma=1.3)
        assert _params(component) == {"weight": 0.9, "sigma": 1.3}


def test_termination_and_metric_factories_bind_metadata_and_wrappers():
    tracking_term = factories.tracking_error_term_factory(threshold=0.6)
    assert _bindings(tracking_term)["current_rigid_body_pos"] == "current.rigid_body_pos"
    assert _params(tracking_term)["threshold"] == 0.6
    assert "settle_steps" not in _params(tracking_term)
    assert (
        _params(factories.tracking_error_term_factory(settle_steps=25))["settle_steps"]
        == 25
    )

    fall = factories.fall_termination_factory(termination_height=0.2)
    assert _bindings(fall)["progress_buf"] == "progress_buf"
    assert _params(fall)["termination_height"] == 0.2

    assert _params(factories.anchor_pos_error_term_factory(threshold=0.1))["threshold"] == 0.1
    assert _params(factories.anchor_ori_error_term_factory(threshold=0.2))["threshold"] == 0.2
    assert _params(factories.relative_body_pos_error_term_factory(threshold=0.3))["threshold"] == 0.3
    assert _params(factories.anchor_height_error_term_factory(threshold=0.4))["threshold"] == 0.4

    no_threshold_metric = factories.gt_error_factory()
    assert _params(no_threshold_metric) == {}
    for factory_fn in [
        factories.gt_error_factory,
        factories.max_joint_error_factory,
        factories.gr_error_factory,
        factories.anchor_pos_metric_factory,
        factories.anchor_ori_metric_factory,
        factories.relative_body_pos_metric_factory,
        factories.anchor_height_error_metric_factory,
    ]:
        component = factory_fn(threshold=0.75)
        assert _params(component)["threshold"] == 0.75

    path_metric = factories.path_distance_error_factory(
        threshold=1.5,
        min_progress=3,
    )
    assert _bindings(path_metric)["target_pos"] == "path.tar_pos"
    assert _params(path_metric)["threshold"] == 0.5
    assert torch.equal(
        path_metric.get_compute_func()(
            head_pos=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            target_pos=torch.zeros(2, 3),
            fail_dist=1.5,
            progress_buf=torch.tensor([10, 10]),
            min_progress=3,
        ),
        torch.tensor([False, True]),
    )

    steering_metric = factories.steering_velocity_error_factory(
        speed_tolerance=0.5,
        direction_tolerance=0.7,
    )
    assert _bindings(steering_metric)["prev_root_pos"] == "steering.prev_root_pos"
    assert _params(steering_metric)["threshold"] == 0.5
    assert torch.equal(
        steering_metric.get_compute_func()(
            root_pos=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            prev_root_pos=torch.zeros(2, 3),
            tar_dir=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            tar_speed=torch.tensor([1.0, 1.0]),
            dt=1.0,
            speed_tolerance=0.5,
            direction_tolerance=0.7,
        ),
        torch.tensor([False, True]),
    )


def test_resume_inject_reward_components_v54_dormant_activation():
    """v5.4 resume-time COMPONENT INJECTION: contact channel + arm-flail tax."""
    # --- env unset => frozen dict byte-identical, no proof lines.
    frozen = {}
    lines = []
    changed = factories.resume_inject_reward_components(
        frozen, env={}, log_fn=lines.append
    )
    assert changed is False and frozen == {} and lines == []

    # --- all three weight vars set, components absent => injected with the
    # right factories, weights, thresholds, and dynamic_vars.
    frozen = {"some_existing": factories.pow_rew_factory(weight=-1e-4)}
    lines = []
    env = {
        "PM_CONTACT_MATCH_WEIGHT": "0.1",
        "PM_CONTACT_MATCH_REF_THRESHOLD": "0.6",
        "PM_LIFTOFF_PENALTY_WEIGHT": "-0.5",
        "PM_ACTION_SMOOTH_LME_WEIGHT": "-0.1",
    }
    changed = factories.resume_inject_reward_components(
        frozen, env=env, log_fn=lines.append
    )
    assert changed is True
    assert set(frozen) == {
        "some_existing", "contact_match", "liftoff_penalty", "action_smooth_lme"
    }

    cm = frozen["contact_match"]
    assert _params(cm)["weight"] == 0.1
    assert _params(cm)["ref_contact_threshold"] == 0.6
    assert _params(cm)["match_reward"] is True
    assert _bindings(cm)["sim_contacts"] == "current.rigid_body_contacts"
    assert _bindings(cm)["ref_contacts"] == "mimic.ref_state.rigid_body_contacts"
    assert _bindings(cm)["contact_body_ids"] == "contact_body_ids"

    lo = frozen["liftoff_penalty"]
    assert _params(lo)["weight"] == -0.5
    assert _params(lo)["ref_contact_threshold"] == 0.5  # default when var unset
    assert _params(lo)["min_value"] == -0.2  # factory income clamp intact
    assert _bindings(lo)["historical_body_contacts"] == "historical.body_contacts"

    lme = frozen["action_smooth_lme"]
    assert _params(lme)["weight"] == -0.1
    assert _bindings(lme)["perturbation_grace_mask"] == "perturbation_grace_mask"
    assert _bindings(lme)["current_processed_action"] == "current_processed_action"

    inject_lines = [l for l in lines if l.startswith("RESUME INJECT component ")]
    assert len(inject_lines) == 3
    assert any(
        l.startswith("RESUME INJECT component contact_match weight=0.1")
        for l in inject_lines
    )

    # --- second resume with the components now frozen in: patch, don't
    # re-inject; unchanged values are silent.
    lines2 = []
    changed = factories.resume_inject_reward_components(
        frozen, env=dict(env, PM_CONTACT_MATCH_WEIGHT="0.2"), log_fn=lines2.append
    )
    assert changed is True
    assert _params(frozen["contact_match"])["weight"] == 0.2
    assert [l for l in lines2 if "RESUME INJECT" in l] == []
    assert any("RESUME override contact_match.weight = 0.2" in l for l in lines2)

    # --- empty-string weight var counts as unset.
    frozen2 = {}
    changed = factories.resume_inject_reward_components(
        frozen2, env={"PM_CONTACT_MATCH_WEIGHT": ""}, log_fn=lambda _l: None
    )
    assert changed is False and frozen2 == {}


# =============================================================================
# DUAL-SIGMA companion + static-hold velocity penalty (2026-08-04)
# =============================================================================

_DUAL_SIGMA_FACTORIES = {
    "global_anchor_pos": factories.global_anchor_pos_rew_factory,
    "global_wrist_pos": factories.global_body_pos_rew_factory,
    "relative_body_pos": factories.relative_body_pos_rew_factory,
    "dof_pos_track": factories.dof_pos_track_rew_factory,
    "heading_local_anchor_drift": factories.heading_local_anchor_drift_rew_factory,
}


def test_dual_sigma_factories_add_no_static_params_when_off():
    """RULE 10: unset companion => static_params byte-identical to before.

    Not merely "fine_weight == 0": the KEYS must be ABSENT, so a config pickled
    by this build is indistinguishable from one pickled before the option
    existed and no frozen-config comparison can drift.
    """
    for name, factory in _DUAL_SIGMA_FACTORIES.items():
        default = _params(factory())
        explicit_off = _params(factory(fine_weight=0.0))
        assert "fine_weight" not in default, name
        assert "fine_sigma" not in default, name
        assert default == explicit_off, name
        # A sigma supplied without a weight is still fully off.
        assert _params(factory(fine_weight=0.0, fine_sigma=0.05)) == default, name


def test_dual_sigma_factories_record_companion_when_enabled():
    for name, factory in _DUAL_SIGMA_FACTORIES.items():
        params = _params(factory(fine_weight=0.5, fine_sigma=0.05))
        assert params["fine_weight"] == 0.5, name
        assert params["fine_sigma"] == 0.05, name
        # The coarse sigma is untouched by enabling the companion.
        assert params["sigma"] == _params(factory())["sigma"], name


def test_dual_sigma_factories_reject_half_set_and_invalid_pairs():
    for name, factory in _DUAL_SIGMA_FACTORIES.items():
        for bad in (
            {"fine_weight": 0.5},                        # sigma missing
            {"fine_weight": 0.5, "fine_sigma": 0.0},     # sigma not positive
            {"fine_weight": 0.5, "fine_sigma": -0.05},
            {"fine_weight": -0.5, "fine_sigma": 0.05},   # negative weight
        ):
            try:
                factory(**bad)
            except ValueError:
                continue
            raise AssertionError(f"{name}: expected ValueError for {bad}")


def test_wrist_position_term_is_the_hand_governing_dual_sigma_binding():
    """The wrist term is relative_body_pos restricted to the wrist bodies.

    Guards the Task-1 finding: hand placement is scored anchor-relative in the
    heading-local frame, NOT in world coordinates, and NOT by global_anchor_pos
    (which is the ROOT). If this binding ever changes, the dual-sigma knob
    aimed at the hand is aimed at the wrong thing.
    """
    wrist = factories.relative_body_pos_rew_factory(
        weight=1.3, sigma=0.3, body_indices=[21, 28],
        fine_weight=0.5, fine_sigma=0.05,
    )
    params, bindings = _params(wrist), _bindings(wrist)
    assert params["body_indices"] == [21, 28]
    assert params["fine_weight"] == 0.5 and params["fine_sigma"] == 0.05
    # Anchor-relative + heading-local: needs the anchor pose, not just world pos.
    assert bindings["current_rigid_body_pos"] == "current.rigid_body_pos"
    assert bindings["current_anchor_pos"] == "current.anchor_pos"
    assert bindings["current_anchor_rot"] == "current.anchor_rot"
    assert bindings["anchor_idx"] == "mimic.anchor_idx"

    # global_anchor_pos, by contrast, never sees a per-body position at all.
    root = _bindings(factories.global_anchor_pos_rew_factory(weight=0.5))
    assert "current_rigid_body_pos" not in root
    assert root["current_anchor_pos"] == "current.anchor_pos"


def test_static_hold_vel_penalty_factory_binds_and_resolves_bodies():
    comp = factories.static_hold_body_vel_penalty_factory(
        weight=-0.5, ref_speed_gate=0.05
    )
    params, bindings = _params(comp), _bindings(comp)
    assert params["weight"] == -0.5
    assert params["ref_speed_gate"] == 0.05
    assert params["zero_during_grace_period"] is True
    assert "body_indices" not in params  # None = all bodies
    assert bindings["rigid_body_vel"] == "current.rigid_body_vel"
    assert bindings["ref_rigid_body_vel"] == "mimic.ref_state.rigid_body_vel"
    # Self-contained by default: no HOLD-FIX still-mask dependency.
    assert "reference_still_mask" not in bindings

    masked = factories.static_hold_body_vel_penalty_factory(
        weight=-0.5, use_reference_still_mask=True
    )
    assert _bindings(masked)["reference_still_mask"] == "reference_still_mask"

    named = factories.static_hold_body_vel_penalty_factory(
        weight=-0.5,
        body_names=["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"],
        body_name_filter=["left_wrist_yaw_link", "right_wrist_yaw_link"],
    )
    assert _params(named)["body_indices"] == [1, 2]

    for bad in (
        {"body_indices": [1], "body_name_filter": ["x"]},  # mutually exclusive
        {"body_name_filter": ["x"]},                       # body_names missing
    ):
        try:
            factories.static_hold_body_vel_penalty_factory(weight=-0.5, **bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_train_agent_resume_rows_cover_every_dual_sigma_and_static_hold_knob():
    """READER/WRITER LAW: a fresh-build gate without its resume row is a trap.

    The resume path freezes the reward config from the pickle, so a knob that
    only exists in teacher.py silently no-ops on every warm resume. Assert the
    re-apply table carries BOTH halves of every dual-sigma pair plus the
    static-hold velocity knobs.
    """
    import pathlib

    source = pathlib.Path(
        __file__
    ).resolve().parents[1].joinpath("train_agent.py").read_text()

    expected = [
        ("global_wrist_pos", "PM_GLOBAL_WRIST_POS_WEIGHT", "weight"),
        ("global_wrist_pos", "PM_GLOBAL_WRIST_POS_SIGMA", "sigma"),
        ("global_wrist_pos", "PM_GLOBAL_WRIST_POS_FINE_WEIGHT", "fine_weight"),
        ("global_wrist_pos", "PM_GLOBAL_WRIST_POS_FINE_SIGMA", "fine_sigma"),
        ("relative_body_pos", "PM_REL_POS_FINE_WEIGHT", "fine_weight"),
        ("relative_body_pos", "PM_REL_POS_FINE_SIGMA", "fine_sigma"),
        ("wrist_relative_body_pos", "PM_WRIST_POS_FINE_WEIGHT", "fine_weight"),
        ("wrist_relative_body_pos", "PM_WRIST_POS_FINE_SIGMA", "fine_sigma"),
        ("global_anchor_pos", "PM_GLOBAL_POS_FINE_WEIGHT", "fine_weight"),
        ("global_anchor_pos", "PM_GLOBAL_POS_FINE_SIGMA", "fine_sigma"),
        ("dof_pos_track", "PM_DOF_POS_TRACK_FINE_WEIGHT", "fine_weight"),
        ("dof_pos_track", "PM_DOF_POS_TRACK_FINE_SIGMA", "fine_sigma"),
        ("heading_local_anchor_drift", "PM_HEADING_DRIFT_FINE_WEIGHT", "fine_weight"),
        ("heading_local_anchor_drift", "PM_HEADING_DRIFT_FINE_SIGMA", "fine_sigma"),
        ("static_hold_vel", "PM_STATIC_HOLD_VEL_WEIGHT", "weight"),
        ("static_hold_vel", "PM_STATIC_HOLD_VEL_REF_GATE", "ref_speed_gate"),
    ]
    for comp, var, key in expected:
        assert f'"{var}"' in source, f"missing resume row for {var}"
        assert f'"{comp}"' in source, f"missing component {comp}"
        assert f'"{key}"' in source, f"missing static_params key {key}"


def test_validate_dual_sigma_components_proves_and_rejects():
    """The resume-time pair validator: loud proof lines, hard errors."""
    # Nothing enabled => silent, nothing active.
    frozen = {
        "relative_body_pos": factories.relative_body_pos_rew_factory(),
        "global_anchor_pos": factories.global_anchor_pos_rew_factory(),
    }
    lines = []
    assert factories.validate_dual_sigma_components(frozen, lines.append) == []
    assert lines == []

    # Enabled => proof line naming the coarse sigma, fine width, and new max.
    frozen["global_anchor_pos"].static_params["fine_weight"] = 0.5
    frozen["global_anchor_pos"].static_params["fine_sigma"] = 0.05
    lines = []
    active = factories.validate_dual_sigma_components(frozen, lines.append)
    assert active == ["global_anchor_pos"]
    proof = [l for l in lines if l.startswith("DUAL-SIGMA ACTIVE global_anchor_pos")]
    assert len(proof) == 1
    assert "sigma=0.3" in proof[0] and "0.05" in proof[0] and "1.500" in proof[0]

    # A fine sigma that is NOT narrower is a loud SUSPECT warning, not silence.
    frozen["global_anchor_pos"].static_params["fine_sigma"] = 0.4
    lines = []
    factories.validate_dual_sigma_components(frozen, lines.append)
    assert any(l.startswith("DUAL-SIGMA SUSPECT") for l in lines)

    # Half-set pair (weight without sigma) is a HARD error, not a rollout NaN.
    frozen["global_anchor_pos"].static_params.pop("fine_sigma")
    try:
        factories.validate_dual_sigma_components(frozen, lambda _l: None)
    except ValueError as exc:
        assert "fine_sigma" in str(exc)
    else:
        raise AssertionError("expected ValueError for half-set dual-sigma pair")

    # Negative fine weight is rejected.
    frozen["global_anchor_pos"].static_params["fine_weight"] = -0.5
    frozen["global_anchor_pos"].static_params["fine_sigma"] = 0.05
    try:
        factories.validate_dual_sigma_components(frozen, lambda _l: None)
    except ValueError as exc:
        assert "fine_weight" in str(exc)
    else:
        raise AssertionError("expected ValueError for negative fine_weight")

    # A component absent from the frozen config is skipped, never invented.
    assert factories.validate_dual_sigma_components({}, lambda _l: None) == []


def test_global_body_pos_factory_binds_world_frame_and_resolves_body_names():
    """The world-frame wrist term: bindings, body-set resolution, gating."""
    comp = factories.global_body_pos_rew_factory(
        weight=0.6, sigma=0.3, body_indices=[21, 28]
    )
    params, bindings = _params(comp), _bindings(comp)
    assert params["weight"] == 0.6 and params["sigma"] == 0.3
    assert params["body_indices"] == [21, 28]
    # WORLD frame: raw body positions vs raw reference positions. Crucially it
    # binds NEITHER anchor_pos NOR anchor_rot -- that is the whole difference
    # from relative_body_pos, which subtracts the anchor from both sides and so
    # cancels base sway exactly.
    assert bindings["current_rigid_body_pos"] == "current.rigid_body_pos"
    assert bindings["ref_rigid_body_pos"] == "mimic.ref_state.rigid_body_pos"
    assert "current_anchor_pos" not in bindings
    assert "current_anchor_rot" not in bindings
    assert "anchor_idx" not in bindings
    # Default is UNGATED (holds + motion).
    assert "reference_still_mask" not in bindings

    gated = factories.global_body_pos_rew_factory(
        weight=0.6, use_reference_still_mask=True
    )
    assert _bindings(gated)["reference_still_mask"] == "reference_still_mask"

    # Body NAMES resolve through body_weights (name -> multiplier).
    named = factories.global_body_pos_rew_factory(
        weight=0.6,
        body_names=["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"],
        body_weights={"left_wrist_yaw_link": 1.0, "right_wrist_yaw_link": 1.0},
    )
    assert _params(named)["body_indices"] == [1, 2]

    # An unknown body name must FAIL LOUDLY, never resolve to a silent default.
    try:
        factories.global_body_pos_rew_factory(
            weight=0.6,
            body_names=["pelvis", "left_wrist_yaw_link"],
            body_weights={"no_such_link": 1.0},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected a hard failure on an unknown body name")


def test_global_wrist_pos_stat_writer_emits_metres_only_when_registered():
    """READER/WRITER: the go/no-go stat must exist, be a Tensor, and be silent
    when the reward component is absent.

    The reward itself is a Gaussian that already reads ~0.95 at the measured
    error, so the METRE-valued error stat is what makes "is it learning"
    answerable. It must also survive the agent's extras aggregator, which
    silently drops non-Tensor values and skips anything under ``raw/``.
    """
    from protomotions.envs.base_env.env import BaseEnv

    cur = torch.zeros(2, 4, 3)
    ref = torch.zeros(2, 4, 3)
    # env0: wrist bodies (2, 3) off by 3 cm and 7 cm. env1: exact.
    cur[0, 2, 0] = 0.03
    cur[0, 3, 1] = 0.07

    ctx = SimpleNamespace(
        current=SimpleNamespace(rigid_body_pos=cur),
        mimic=SimpleNamespace(ref_state=SimpleNamespace(rigid_body_pos=ref)),
    )

    # --- component ABSENT => no keys written at all (Rule 10).
    quiet = SimpleNamespace(
        config=SimpleNamespace(reward_components={}), extras={}
    )
    BaseEnv._log_global_wrist_pos_extras(quiet, ctx)
    assert quiet.extras == {}

    # --- component REGISTERED => metre-valued per-env error over its body set.
    comp = factories.global_body_pos_rew_factory(weight=0.6, body_indices=[2, 3])
    env = SimpleNamespace(
        config=SimpleNamespace(reward_components={"global_wrist_pos": comp}),
        extras={},
    )
    BaseEnv._log_global_wrist_pos_extras(env, ctx)

    assert set(env.extras) == {
        "global_wrist_pos/err_m", "global_wrist_pos/err_max_m"
    }
    for key, value in env.extras.items():
        assert isinstance(value, torch.Tensor), f"{key} must be a Tensor"
        assert not key.startswith("raw/"), f"{key} would be skipped by the agent"
        assert value.shape == (2,)

    # env0 mean of (0.03, 0.07) = 0.05 ; max = 0.07 ; env1 is exact.
    assert torch.allclose(
        env.extras["global_wrist_pos/err_m"], torch.tensor([0.05, 0.0]), atol=1e-6
    )
    assert torch.allclose(
        env.extras["global_wrist_pos/err_max_m"],
        torch.tensor([0.07, 0.0]),
        atol=1e-6,
    )

    # --- no mimic reference (e.g. a non-mimic env) => silent, no partial keys.
    env2 = SimpleNamespace(
        config=SimpleNamespace(reward_components={"global_wrist_pos": comp}),
        extras={},
    )
    BaseEnv._log_global_wrist_pos_extras(
        env2, SimpleNamespace(current=ctx.current, mimic=None)
    )
    assert env2.extras == {}
