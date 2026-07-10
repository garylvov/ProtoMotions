# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MimicADD observation additions."""

import math
from types import SimpleNamespace

import torch

from protomotions.agents.mimic import agent_add
from protomotions.agents.mimic.agent_add import (
    MimicADD,
    NUM_ROOT_DISPLACEMENT_FEATURES,
    compute_root_displacement_features,
)


def _state(body_pos, body_rot=None):
    if body_rot is None:
        body_rot = torch.zeros(body_pos.shape[0], body_pos.shape[1], 4)
    return SimpleNamespace(
        rigid_body_pos=body_pos,
        rigid_body_rot=body_rot,
        rigid_body_vel=torch.zeros_like(body_pos),
        rigid_body_ang_vel=torch.zeros_like(body_pos),
    )


def _yaw_quat(yaw: float) -> torch.Tensor:
    """w-last (xyzw) quaternion for a rotation of ``yaw`` about z."""
    return torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def test_mimic_add_adds_tracking_difference_observation(monkeypatch):
    num_envs = 2
    ref_pos = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
        ]
    )
    cur_pos = ref_pos - 0.5
    env = SimpleNamespace(
        motion_manager=SimpleNamespace(
            motion_times=torch.tensor([0.1, 0.2]),
            motion_ids=torch.tensor([1, 2]),
        ),
        motion_lib=SimpleNamespace(
            get_motion_state=lambda motion_ids, motion_times: _state(ref_pos.clone())
        ),
        get_spawn_to_ref_pose_offset_with_terrain_height_correction=lambda pos: torch.zeros_like(
            pos
        ),
        terrain=SimpleNamespace(
            get_ground_heights=lambda roots: roots[:, 2] * 0.0,
        ),
        simulator=SimpleNamespace(get_bodies_state=lambda: _state(cur_pos.clone())),
    )
    agent = MimicADD.__new__(MimicADD)
    agent.env = env
    agent.num_envs = num_envs

    monkeypatch.setattr(
        agent_add.AMP,
        "add_agent_info_to_obs",
        lambda self, obs: {**obs, "base_added": torch.ones(num_envs, 1)},
    )

    def fake_max_coords(body_pos, ground_height, **kwargs):
        assert kwargs["local_obs"] is False
        assert kwargs["root_height_obs"] is True
        assert kwargs["observe_contacts"] is False
        assert kwargs["body_contacts"].shape == (num_envs, 0)
        return body_pos.reshape(num_envs, -1) + ground_height[:, None]

    monkeypatch.setattr(
        agent_add,
        "compute_humanoid_max_coords_observations",
        fake_max_coords,
    )

    obs = agent.add_agent_info_to_obs({"obs": torch.zeros(num_envs, 1)})

    assert torch.equal(obs["base_added"], torch.ones(num_envs, 1))
    assert torch.equal(
        obs["mimic_target_poses_diff"],
        (ref_pos - cur_pos).reshape(num_envs, -1),
    )


def test_root_displacement_features_values_and_wrapping():
    # Env 0: current heading 0 — heading-frame xy error equals world error.
    # Env 1: current heading pi/2 — world error (1, 0) maps to local (0, -1);
    #        heading error wraps: ref -3.0 rad vs cur 3.0 rad -> +0.2832 rad.
    ref_root_pos = torch.tensor([[1.0, 2.0, 0.9], [2.0, 0.0, 0.9]])
    cur_root_pos = torch.tensor([[0.0, 0.0, 0.9], [1.0, 0.0, 0.9]])
    ref_root_rot = torch.stack([_yaw_quat(math.pi / 2), _yaw_quat(-3.0)])
    cur_root_rot = torch.stack([_yaw_quat(0.0), _yaw_quat(math.pi / 2)])

    feats = compute_root_displacement_features(
        ref_root_pos, ref_root_rot, cur_root_pos, cur_root_rot, w_last=True
    )

    assert feats.shape == (2, NUM_ROOT_DISPLACEMENT_FEATURES)
    torch.testing.assert_close(
        feats[0], torch.tensor([1.0, 2.0, math.pi / 2]), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        feats[1, :2], torch.tensor([0.0, -1.0]), atol=1e-5, rtol=0
    )
    expected_wrap = math.remainder(-3.0 - math.pi / 2, 2 * math.pi)
    torch.testing.assert_close(
        feats[1, 2], torch.tensor(expected_wrap), atol=1e-5, rtol=0
    )
    # Perfect tracking -> exact zero features (matches ADD's zero positives).
    zero = compute_root_displacement_features(
        ref_root_pos, ref_root_rot, ref_root_pos.clone(), ref_root_rot.clone()
    )
    torch.testing.assert_close(zero, torch.zeros(2, 3), atol=1e-6, rtol=0)


def test_mimic_add_root_displacement_features_appended_when_enabled(monkeypatch):
    num_envs = 2
    ref_pos = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
        ]
    )
    cur_pos = ref_pos - 0.5
    ref_rot = torch.stack(
        [_yaw_quat(math.pi / 2).expand(2, 4), _yaw_quat(0.0).expand(2, 4)]
    ).clone()
    cur_rot = torch.stack(
        [_yaw_quat(0.0).expand(2, 4), _yaw_quat(0.0).expand(2, 4)]
    ).clone()
    env = SimpleNamespace(
        motion_manager=SimpleNamespace(
            motion_times=torch.tensor([0.1, 0.2]),
            motion_ids=torch.tensor([1, 2]),
        ),
        motion_lib=SimpleNamespace(
            get_motion_state=lambda motion_ids, motion_times: _state(
                ref_pos.clone(), ref_rot.clone()
            )
        ),
        get_spawn_to_ref_pose_offset_with_terrain_height_correction=lambda pos: torch.zeros_like(
            pos
        ),
        terrain=SimpleNamespace(
            get_ground_heights=lambda roots: roots[:, 2] * 0.0,
        ),
        simulator=SimpleNamespace(
            get_bodies_state=lambda: _state(cur_pos.clone(), cur_rot.clone())
        ),
    )
    agent = MimicADD.__new__(MimicADD)
    agent.env = env
    agent.num_envs = num_envs
    agent.config = SimpleNamespace(add_root_displacement_features=True)

    monkeypatch.setattr(
        agent_add.AMP,
        "add_agent_info_to_obs",
        lambda self, obs: obs,
    )
    monkeypatch.setattr(
        agent_add,
        "compute_humanoid_max_coords_observations",
        lambda body_pos, ground_height, **kwargs: body_pos.reshape(num_envs, -1),
    )

    obs = agent.add_agent_info_to_obs({})
    diff = obs["mimic_target_poses_diff"]

    pose_diff = (ref_pos - cur_pos).reshape(num_envs, -1)
    expected_feats = compute_root_displacement_features(
        ref_root_pos=ref_pos[:, 0],
        ref_root_rot=ref_rot[:, 0],
        current_root_pos=cur_pos[:, 0],
        current_root_rot=cur_rot[:, 0],
    )
    # Default form (2026-07-10): HEADING channel only is appended.
    assert diff.shape == (num_envs, pose_diff.shape[1] + 1)
    torch.testing.assert_close(diff[:, : pose_diff.shape[1]], pose_diff)
    torch.testing.assert_close(diff[:, -1:], expected_feats[:, 2:3])
    torch.testing.assert_close(
        diff[0, -1], torch.tensor(math.pi / 2), atol=1e-5, rtol=0
    )

    # A/B path: add_root_xy_features restores the legacy 3-channel form.
    agent2 = MimicADD.__new__(MimicADD)
    agent2.env = env
    agent2.num_envs = num_envs
    agent2.config = SimpleNamespace(
        add_root_displacement_features=True, add_root_xy_features=True
    )
    diff = agent2.add_agent_info_to_obs({})["mimic_target_poses_diff"]
    assert diff.shape == (
        num_envs, pose_diff.shape[1] + NUM_ROOT_DISPLACEMENT_FEATURES
    )
    torch.testing.assert_close(diff[:, -3:], expected_feats)
    # Env 0: xy error 0.5, 0.5 in heading frame (cur heading 0), heading err pi/2.
    torch.testing.assert_close(
        diff[0, -3:], torch.tensor([0.5, 0.5, math.pi / 2]), atol=1e-5, rtol=0
    )


def test_mimic_add_root_displacement_features_off_by_default(monkeypatch):
    """Without the flag (or without config at all) behavior is unchanged."""
    num_envs = 1
    ref_pos = torch.ones(num_envs, 2, 3)
    cur_pos = torch.zeros(num_envs, 2, 3)
    env = SimpleNamespace(
        motion_manager=SimpleNamespace(
            motion_times=torch.tensor([0.1]),
            motion_ids=torch.tensor([0]),
        ),
        motion_lib=SimpleNamespace(
            get_motion_state=lambda motion_ids, motion_times: _state(ref_pos.clone())
        ),
        get_spawn_to_ref_pose_offset_with_terrain_height_correction=lambda pos: torch.zeros_like(
            pos
        ),
        terrain=SimpleNamespace(get_ground_heights=lambda roots: roots[:, 2] * 0.0),
        simulator=SimpleNamespace(get_bodies_state=lambda: _state(cur_pos.clone())),
    )
    agent = MimicADD.__new__(MimicADD)
    agent.env = env
    agent.num_envs = num_envs

    monkeypatch.setattr(agent_add.AMP, "add_agent_info_to_obs", lambda self, obs: obs)
    monkeypatch.setattr(
        agent_add,
        "compute_humanoid_max_coords_observations",
        lambda body_pos, ground_height, **kwargs: body_pos.reshape(num_envs, -1),
    )

    obs = agent.add_agent_info_to_obs({})
    assert obs["mimic_target_poses_diff"].shape == (num_envs, 6)


def test_mimic_add_expert_disc_obs_extends_dim_when_features_enabled(monkeypatch):
    agent = MimicADD.__new__(MimicADD)
    agent.device = torch.device("cpu")
    agent.config = SimpleNamespace(add_root_displacement_features=True)
    agent.env = SimpleNamespace(
        observation_manager=SimpleNamespace(
            observation_history_buffers={
                "max_coords_obs": SimpleNamespace(data=torch.zeros(4, 5))
            }
        )
    )
    monkeypatch.setattr(
        agent_add.AMP,
        "get_expert_disc_obs",
        lambda self, num_samples: {"max_coords_obs": torch.ones(num_samples, 40)},
    )

    obs = agent.get_expert_disc_obs(3)

    # Expert positives stay all-zero, extended by the HEADING channel
    # (default form; xy channels only with add_root_xy_features).
    assert torch.equal(
        obs["mimic_target_poses_diff"],
        torch.zeros(3, 5 + 1),
    )

    agent.config = SimpleNamespace(
        add_root_displacement_features=True, add_root_xy_features=True
    )
    obs = agent.get_expert_disc_obs(3)
    assert torch.equal(
        obs["mimic_target_poses_diff"],
        torch.zeros(3, 5 + NUM_ROOT_DISPLACEMENT_FEATURES),
    )


def test_mimic_add_expert_disc_obs_adds_zero_tracking_diff_from_history(monkeypatch):
    agent = MimicADD.__new__(MimicADD)
    agent.device = torch.device("cpu")
    agent.env = SimpleNamespace(
        observation_manager=SimpleNamespace(
            observation_history_buffers={
                "max_coords_obs": SimpleNamespace(data=torch.zeros(4, 5))
            }
        )
    )
    monkeypatch.setattr(
        agent_add.AMP,
        "get_expert_disc_obs",
        lambda self, num_samples: {"max_coords_obs": torch.ones(num_samples, 40)},
    )

    obs = agent.get_expert_disc_obs(3)

    assert torch.equal(obs["mimic_target_poses_diff"], torch.zeros(3, 5))


def test_mimic_add_expert_disc_obs_infers_dim_from_expert_obs(monkeypatch):
    agent = MimicADD.__new__(MimicADD)
    agent.device = torch.device("cpu")
    agent.env = SimpleNamespace(
        observation_manager=SimpleNamespace(observation_history_buffers={})
    )
    monkeypatch.setattr(
        agent_add.AMP,
        "get_expert_disc_obs",
        lambda self, num_samples: {
            "historical_max_coords_obs": torch.ones(num_samples, 24)
        },
    )

    obs = agent.get_expert_disc_obs(2)

    assert torch.equal(obs["mimic_target_poses_diff"], torch.zeros(2, 3))


# ---------------------------------------------------------------------------
# Track D 2026-07-10: per-body diff feature scales + global-diff-bodies A/B
# ---------------------------------------------------------------------------

def test_build_diff_feature_scale_vector_layout():
    from protomotions.agents.mimic.agent_add import build_diff_feature_scale_vector

    body_names = ["pelvis", "left_wrist_yaw_link", "head_aux", "left_foot"]
    vec = build_diff_feature_scale_vector(
        body_names,
        {"_wrist_yaw_link": 3.0, "head": 1.5, "root_displacement": 2.0},
        num_appended_root_channels=1,
        device=torch.device("cpu"),
    )
    b = 4
    assert vec.shape[0] == 1 + 3 * (b - 1) + 6 * b + 3 * b + 3 * b + 1
    # root_h channel takes body 0's (pelvis) scale = 1.0
    assert vec[0] == 1.0
    # pos block: bodies 1..3 (wrist, head, foot), 3 channels each
    torch.testing.assert_close(vec[1:4], torch.full((3,), 3.0))
    torch.testing.assert_close(vec[4:7], torch.full((3,), 1.5))
    torch.testing.assert_close(vec[7:10], torch.full((3,), 1.0))
    # rot block: bodies 0..3, 6 channels each
    rot0 = 10
    torch.testing.assert_close(vec[rot0:rot0 + 6], torch.full((6,), 1.0))
    torch.testing.assert_close(vec[rot0 + 6:rot0 + 12], torch.full((6,), 3.0))
    # appended root channel takes the root_displacement scale
    assert vec[-1] == 2.0
    # dormant spec: all-ones
    ones = build_diff_feature_scale_vector(
        body_names, {}, 0, torch.device("cpu"))
    assert torch.equal(ones, torch.ones(1 + 9 + 24 + 12 + 12))


def _mini_env(ref_pos, cur_pos, body_names, num_envs):
    return SimpleNamespace(
        motion_manager=SimpleNamespace(
            motion_times=torch.zeros(num_envs),
            motion_ids=torch.zeros(num_envs, dtype=torch.long),
        ),
        motion_lib=SimpleNamespace(
            get_motion_state=lambda motion_ids, motion_times: _state(ref_pos.clone())
        ),
        get_spawn_to_ref_pose_offset_with_terrain_height_correction=lambda pos: torch.zeros_like(pos),
        terrain=SimpleNamespace(get_ground_heights=lambda roots: roots[:, 2] * 0.0),
        simulator=SimpleNamespace(get_bodies_state=lambda: _state(cur_pos.clone())),
        robot_config=SimpleNamespace(
            kinematic_info=SimpleNamespace(body_names=list(body_names))
        ),
    )


def test_global_diff_bodies_pure_translation_shows_in_listed_bodies(monkeypatch):
    """A pure root translation offset appears ONLY in the global bodies'
    position-diff channels; root-stripped bodies read zero (A/B option)."""
    num_envs = 1
    body_names = ["pelvis", "left_wrist_yaw_link", "left_foot"]
    b = len(body_names)
    cur_pos = torch.tensor([[[0.0, 0.0, 1.0], [0.3, 0.2, 1.2], [0.1, 0.0, 0.05]]])
    ref_pos = cur_pos + torch.tensor([0.5, 0.0, 0.0])  # whole-body translation

    agent = MimicADD.__new__(MimicADD)
    agent.env = _mini_env(ref_pos, cur_pos, body_names, num_envs)
    agent.num_envs = num_envs
    agent.config = SimpleNamespace(global_diff_bodies=["_wrist_yaw_link"])

    monkeypatch.setattr(agent_add.AMP, "add_agent_info_to_obs", lambda self, obs: obs)
    # Fake obs fn with a layout-compatible pos block: [root_h | body pos 1..B-1]
    monkeypatch.setattr(
        agent_add,
        "compute_humanoid_max_coords_observations",
        lambda body_pos, ground_height, **kwargs: torch.cat(
            [body_pos[:, 0, 2:3], body_pos[:, 1:].reshape(num_envs, -1)], dim=-1
        ),
    )

    diff = agent.add_agent_info_to_obs({})["mimic_target_poses_diff"]
    assert diff.shape == (num_envs, 1 + 3 * (b - 1))
    # Wrist (body 1) keeps the GLOBAL error incl. translation.
    torch.testing.assert_close(diff[0, 1:4], torch.tensor([0.5, 0.0, 0.0]))
    # Foot (body 2) is root-stripped: pure translation cancels to zero.
    torch.testing.assert_close(diff[0, 4:7], torch.zeros(3))


def test_diff_feature_scales_applied_and_dormant_by_default(monkeypatch):
    num_envs = 1
    body_names = ["pelvis", "left_wrist_yaw_link", "left_foot"]
    b = len(body_names)
    total = 1 + 3 * (b - 1) + 6 * b + 3 * b + 3 * b
    cur_pos = torch.zeros(num_envs, b, 3)
    ref_pos = torch.ones(num_envs, b, 3)

    def fake_obs(body_pos, ground_height, **kwargs):
        # Constant-per-input obs with the exact expected layout width.
        return torch.full((num_envs, total), float(body_pos.sum()))

    def build(config):
        agent = MimicADD.__new__(MimicADD)
        agent.env = _mini_env(ref_pos, cur_pos, body_names, num_envs)
        agent.num_envs = num_envs
        if config is not None:
            agent.config = config
        return agent

    monkeypatch.setattr(agent_add.AMP, "add_agent_info_to_obs", lambda self, obs: obs)
    monkeypatch.setattr(
        agent_add, "compute_humanoid_max_coords_observations", fake_obs
    )

    base = build(None).add_agent_info_to_obs({})["mimic_target_poses_diff"]
    scaled = build(
        SimpleNamespace(add_diff_feature_scales={"_wrist_yaw_link": 3.0})
    ).add_agent_info_to_obs({})["mimic_target_poses_diff"]

    assert base.shape == scaled.shape == (num_envs, total)
    # Wrist pos channels x3, foot pos channels unscaled.
    torch.testing.assert_close(scaled[0, 1:4], base[0, 1:4] * 3.0)
    torch.testing.assert_close(scaled[0, 4:7], base[0, 4:7])
    # Expert zero positives stay zero under scaling: dim cache carries width.
    agent = build(SimpleNamespace(add_diff_feature_scales={"_wrist_yaw_link": 3.0},
                                  reference_obs_components=None))
    agent.device = torch.device("cpu")
    agent.add_agent_info_to_obs({})
    expert = agent.get_expert_disc_obs(2)
    assert torch.equal(expert["mimic_target_poses_diff"], torch.zeros(2, total))
