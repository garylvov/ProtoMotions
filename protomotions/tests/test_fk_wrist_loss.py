# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard tests for the FK Cartesian losses (supervised student).

Two terms, one shared FK pass and one shared reference:
  * ``fk_wrist_*``    -- wrist position + orientation
  * ``fk_global_pos`` -- all-body position, targeting the eval's
                         ``mean_body_pos_error``

Both read ``mimic_target_poses`` (``build_max_coords_target_poses``) and
RE-ANCHOR it to its own root before comparing.

THE TWO BUGS THIS FILE EXISTS TO PREVENT
----------------------------------------
1. (2026-08-04) ``max_coords_obs`` width was validated as 1+3*(NB-1)+15*NB when
   the kernel emits 1+3*(NB-1)+12*NB. The suite passed anyway because its
   helper built the tensor from the SAME wrong formula. Fix: every observation
   the readers are tested against now comes out of the real kernel via the real
   factory.
2. (2026-08-05) The reference was used WITHOUT re-anchoring it to its own root.
   The raw block is ``R_hinv_cur . (ref_body(t+d) - root(t))``: the reference
   body at t+d measured from the CURRENT root, so the residual carried the
   root's displacement over the lookahead -- metres for locomotion. It drove
   the live ``fk_wrist_pos_loss`` to ~7.7 (2.8 m RMS) on a robot whose wrists
   sit ~0.5 m from the pelvis. The decisive guard is
   ``test_pure_root_translation_scores_exactly_zero``: translate the reference
   bodily while holding the joint angles, and a correct term must score 0.
"""

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from protomotions.agents.common.supervision import (
    SupervisionLossConfig,
    SupervisionLossType,
)
from protomotions.agents.supervised.agent import SupervisedAgent
from protomotions.agents.supervised.config import SupervisedAgentConfig
from protomotions.components import pose_lib
from protomotions.envs.action.action_functions import make_pd_action_config
from protomotions.robot_configs.h1_2 import H1_2RobotConfig
from protomotions.utils import rotations

MJCF_PATH = "protomotions/data/assets/mjcf/h1_2_box_feet.xml"
WRISTS = ["left_wrist_yaw_link", "right_wrist_yaw_link"]


# ---------------------------------------------------------------------------
# helpers -- every observation comes from the REAL kernel
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def robot_config():
    return H1_2RobotConfig()


def _fk(kinematic_info, joint_targets, root_pos=None):
    if root_pos is None:
        root_pos = joint_targets.new_zeros(joint_targets.shape[0], 3)
    mats = pose_lib.extract_transforms_from_qpos_non_root(
        kinematic_info, joint_targets
    )
    return pose_lib.compute_forward_kinematics_from_transforms(
        kinematic_info, root_pos, mats
    )


def _experiment_max_coords_component(use_noisy=False, local_obs=True):
    from protomotions.envs.component_factories import max_coords_obs_factory

    return max_coords_obs_factory(
        use_noisy=use_noisy,
        local_obs=local_obs,
        root_height_obs=True,
        observe_contacts=False,
    )


def _experiment_mimic_target_poses_component(
    with_velocities=True, with_relative=True, future_steps=1, use_noisy=False
):
    """The exact component masked_mimic_trackc_v1.py builds."""
    from protomotions.envs.component_factories import (
        mimic_target_poses_max_coords_factory,
    )

    return mimic_target_poses_max_coords_factory(
        use_noisy=use_noisy,
        with_velocities=with_velocities,
        with_relative=with_relative,
        future_steps=future_steps,
    )


def _experiment_components():
    return {
        "max_coords_obs": _experiment_max_coords_component(),
        "mimic_target_poses": _experiment_mimic_target_poses_component(),
    }


def _real_max_coords_obs(robot_config, cur_pos, cur_rot, observe_contacts=False):
    """max_coords_obs from the real kernel + the heading-local root rotation."""
    from protomotions.envs.obs.humanoid import (
        compute_humanoid_max_coords_observations,
    )

    batch, num_bodies = cur_pos.shape[0], cur_pos.shape[1]
    obs = compute_humanoid_max_coords_observations(
        body_pos=cur_pos,
        body_rot=cur_rot,
        body_vel=torch.zeros(batch, num_bodies, 3),
        body_ang_vel=torch.zeros(batch, num_bodies, 3),
        ground_height=torch.zeros(batch),
        body_contacts=torch.zeros(batch, 2),
        local_obs=True,
        root_height_obs=True,
        observe_contacts=observe_contacts,
        w_last=True,
    )
    heading_local = rotations.quat_mul(
        rotations.calc_heading_quat_inv(cur_rot[:, 0], w_last=True),
        cur_rot[:, 0],
        w_last=True,
    )
    return obs, pose_lib.quaternion_to_matrix(heading_local, w_last=True)


def _real_mimic_target_poses(
    cur_pos, cur_rot, ref_pos, ref_rot, with_velocities=True, with_relative=True
):
    """mimic_target_poses from the real kernel. All inputs are WORLD frame."""
    from protomotions.envs.obs.target_poses import build_max_coords_target_poses

    batch, num_bodies = cur_pos.shape[0], cur_pos.shape[1]
    return build_max_coords_target_poses(
        current_state_body_pos=cur_pos,
        current_state_body_rot=cur_rot,
        current_state_body_vel=torch.zeros(batch, num_bodies, 3),
        current_state_body_ang_vel=torch.zeros(batch, num_bodies, 3),
        mimic_ref_pos=ref_pos.unsqueeze(1),
        mimic_ref_rot=ref_rot.unsqueeze(1),
        mimic_ref_vel=torch.zeros(batch, 1, num_bodies, 3),
        mimic_ref_ang_vel=torch.zeros(batch, 1, num_bodies, 3),
        with_velocities=with_velocities,
        with_relative=with_relative,
        w_last=True,
    )


def _random_state(robot_config, batch, seed=None):
    """A random but well-formed (body_pos, body_rot) world state."""
    if seed is not None:
        torch.manual_seed(seed)
    num_bodies = robot_config.kinematic_info.num_bodies
    pos = torch.randn(batch, num_bodies, 3)
    rot = torch.nn.functional.normalize(torch.randn(batch, num_bodies, 4), dim=-1)
    return pos, rot


def _pose_from_action(robot_config, actions, root_world_rot):
    """World body poses of the COMMANDED configuration, root at the origin."""
    action_config = make_pd_action_config(robot_config)
    params = {k: v for k, v in action_config.items() if k != "fn"}
    joint_targets = action_config["fn"](actions, **params)["processed_action"]
    pos_pelvis, rot_pelvis = _fk(robot_config.kinematic_info, joint_targets)
    root_mat = pose_lib.quaternion_to_matrix(root_world_rot, w_last=True)
    pos_world = torch.einsum("bij,bnj->bni", root_mat, pos_pelvis)
    rot_world = torch.einsum("bij,bnjk->bnik", root_mat, rot_pelvis)
    quat_world = pose_lib.matrix_to_quaternion(rot_world, w_last=True)
    return pos_world, quat_world


def _make_agent(robot_config, config_overrides=None, has_fk_fields=True):
    agent = object.__new__(SupervisedAgent)
    fields = dict(
        model=SimpleNamespace(),
        loss=SupervisionLossConfig(
            loss_type=SupervisionLossType.MSE,
            prediction_key="privileged_action",
            target_key="expert_actions",
            log_prefix="supervision",
        ),
        l2c2_weight=0.0,
        l2c2_obs_pairs={},
    )
    if has_fk_fields:
        fields.update(
            fk_wrist_pos_weight=0.0,
            fk_wrist_ori_weight=0.0,
            fk_wrist_body_names=list(WRISTS),
            fk_wrist_ref_key="mimic_target_poses",
            fk_wrist_root_rot_obs_key="max_coords_obs",
            fk_wrist_future_step=0,
            fk_global_pos_weight=0.0,
            fk_global_body_names=None,
            fk_global_ref_key="mimic_target_poses",
            fk_global_future_step=0,
        )
    fields.update(config_overrides or {})
    agent.config = SimpleNamespace(**fields)
    agent.device = torch.device("cpu")
    agent.current_epoch = 0
    agent.env = SimpleNamespace(
        robot_config=robot_config,
        config=SimpleNamespace(
            observation_components=_experiment_components(),
            action_config=make_pd_action_config(robot_config),
        ),
    )
    return agent


def _batch(robot_config, actions, cur_pos, cur_rot, ref_pos, ref_rot):
    max_coords_obs, _ = _real_max_coords_obs(robot_config, cur_pos, cur_rot)
    return {
        "privileged_action": actions,
        "max_coords_obs": max_coords_obs,
        "mimic_target_poses": _real_mimic_target_poses(
            cur_pos, cur_rot, ref_pos, ref_rot
        ),
    }


# ---------------------------------------------------------------------------
# 1. FK correctness against an independent implementation (MuJoCo)
# ---------------------------------------------------------------------------


def test_pose_lib_fk_matches_mujoco_forward(robot_config):
    mujoco = pytest.importorskip("mujoco")

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    ki = robot_config.kinematic_info

    mj_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    mj_body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]
    assert mj_joint_names[1:] == list(ki.dof_names)
    assert mj_body_names[1:] == list(ki.body_names)

    rng = np.random.RandomState(0)
    for trial in range(3):
        joints = rng.uniform(-0.4, 0.4, size=ki.num_dofs)
        data.qpos[:3] = 0.0
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:] = joints
        mujoco.mj_forward(model, data)

        pos, rot = _fk(ki, torch.tensor(joints, dtype=torch.float32)[None])
        for name in ki.body_names:
            i = ki.body_names.index(name)
            j = mj_body_names.index(name)
            assert np.allclose(
                pos[0, i].numpy(), data.xpos[j], atol=1e-5
            ), f"trial {trial}: position mismatch on {name}"
            assert np.allclose(
                rot[0, i].numpy(), data.xmat[j].reshape(3, 3), atol=1e-5
            ), f"trial {trial}: rotation mismatch on {name}"


def test_pose_lib_fk_is_differentiable(robot_config):
    ki = robot_config.kinematic_info
    joints = (torch.randn(4, ki.num_dofs) * 0.2).requires_grad_(True)
    pos, _ = _fk(ki, joints)
    pos[:, ki.body_names.index("left_wrist_yaw_link")].pow(2).sum().backward()
    assert joints.grad is not None
    assert torch.isfinite(joints.grad).all()
    assert joints.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 2. THE ANCHOR GUARD -- the test that would have caught the ~7.7 bug
# ---------------------------------------------------------------------------


def test_pure_root_translation_scores_exactly_zero(robot_config):
    """The robot walked d metres holding the SAME joint angles.

    The commanded joint configuration is already perfect, so BOTH terms must
    score exactly zero. Without the root re-anchoring the reference is measured
    from the CURRENT root and this scores ~||d||^2 instead -- which is exactly
    how the wrist term reached 7.7 m^2 (2.8 m RMS) on the live run.
    """
    torch.manual_seed(30)
    ki = robot_config.kinematic_info
    batch = 6
    actions = (torch.randn(batch, ki.num_dofs) * 0.25).requires_grad_(True)

    root_quat = torch.nn.functional.normalize(torch.randn(batch, 4), dim=-1)
    cur_pos, cur_rot = _pose_from_action(robot_config, actions, root_quat)
    cur_pos, cur_rot = cur_pos.detach(), cur_rot.detach()

    # Same pose, translated bodily by a large displacement (the live scale).
    disp = torch.tensor([[2.4, -1.3, 0.35]]).repeat(batch, 1)
    ref_pos = cur_pos + disp.unsqueeze(1)
    ref_rot = cur_rot.clone()

    agent = _make_agent(
        robot_config,
        {
            "fk_wrist_pos_weight": 1.0,
            "fk_wrist_ori_weight": 1.0,
            "fk_global_pos_weight": 1.0,
        },
    )
    batch_td = _batch(robot_config, actions, cur_pos, cur_rot, ref_pos, ref_rot)

    pos_loss, ori_loss = agent._calculate_fk_wrist_loss(batch_td, actions)
    global_loss = agent._calculate_fk_global_loss(batch_td, actions)

    assert pos_loss.item() < 1e-8, (
        f"wrist position loss {pos_loss.item():.4f} != 0 under a pure root "
        f"translation of {disp[0].tolist()} m -- reference is not root-anchored"
    )
    assert ori_loss.item() < 1e-8
    assert global_loss.item() < 1e-8, (
        f"global loss {global_loss.item():.4f} != 0 under a pure root "
        "translation -- reference is not root-anchored"
    )


def test_anchor_correction_removes_exactly_the_root_offset(robot_config):
    """Pin the failure mode: un-anchored, the block carries the root offset."""
    torch.manual_seed(31)
    ki = robot_config.kinematic_info
    batch = 4
    actions = torch.randn(batch, ki.num_dofs) * 0.2
    root_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(batch, 1)
    cur_pos, cur_rot = _pose_from_action(robot_config, actions, root_quat)

    disp_len = 2.8
    ref_pos = cur_pos + torch.tensor([[disp_len, 0.0, 0.0]]).repeat(
        batch, 1
    ).unsqueeze(1)

    agent = _make_agent(robot_config, {"fk_global_pos_weight": 1.0})
    ctx = agent._fk_wrist_context(torch.device("cpu"))
    obs = _real_mimic_target_poses(cur_pos, cur_rot, ref_pos, cur_rot)

    raw = obs[..., : 3 * ki.num_bodies].view(batch, ki.num_bodies, 3)
    anchored, _ = agent._fk_reference_blocks(
        {"mimic_target_poses": obs}, ctx, "mimic_target_poses", 0
    )
    # Every body in the raw block sits ~disp_len away from the current root.
    assert raw.norm(dim=-1).min() > disp_len - 1.0
    # The anchor correction removes exactly that.
    assert torch.allclose(anchored, raw - raw[:, 0:1, :], atol=1e-6)
    assert anchored[:, 0].abs().max() < 1e-6


# ---------------------------------------------------------------------------
# 3. physically-interpretable units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset_cm", [5.0, 13.1, 25.0])
def test_wrist_loss_equals_squared_offset_in_metres(robot_config, offset_cm):
    """A known N-cm wrist offset must score exactly (N/100)^2."""
    torch.manual_seed(32)
    ki = robot_config.kinematic_info
    batch = 4
    actions = torch.randn(batch, ki.num_dofs) * 0.2
    root_quat = torch.nn.functional.normalize(torch.randn(batch, 4), dim=-1)
    cur_pos, cur_rot = _pose_from_action(robot_config, actions, root_quat)

    d = offset_cm / 100.0
    ref_pos = cur_pos.clone()
    for name in WRISTS:
        b = ki.body_names.index(name)
        ref_pos[:, b] = ref_pos[:, b] + torch.tensor([0.0, 0.0, d])

    agent = _make_agent(robot_config, {"fk_wrist_pos_weight": 1.0})
    batch_td = _batch(robot_config, actions, cur_pos, cur_rot, ref_pos, cur_rot)
    pos_loss, _ = agent._calculate_fk_wrist_loss(batch_td, actions)
    assert pos_loss.item() == pytest.approx(d * d, rel=1e-4)


@pytest.mark.parametrize("offset_cm", [10.0, 20.0])
def test_global_loss_equals_squared_offset_in_metres(robot_config, offset_cm):
    """The same units guard for the all-body term."""
    torch.manual_seed(33)
    ki = robot_config.kinematic_info
    batch = 3
    actions = torch.randn(batch, ki.num_dofs) * 0.2
    root_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(batch, 1)
    cur_pos, cur_rot = _pose_from_action(robot_config, actions, root_quat)

    d = offset_cm / 100.0
    ref_pos = cur_pos.clone()
    # every body EXCEPT the root (displacing the root is a pure translation and
    # is correctly invisible after anchoring)
    ref_pos[:, 1:] = ref_pos[:, 1:] + torch.tensor([0.0, d, 0.0])

    agent = _make_agent(robot_config, {"fk_global_pos_weight": 1.0})
    batch_td = _batch(robot_config, actions, cur_pos, cur_rot, ref_pos, cur_rot)
    loss = agent._calculate_fk_global_loss(batch_td, actions)
    expected = d * d * (ki.num_bodies - 1) / ki.num_bodies
    assert loss.item() == pytest.approx(expected, rel=1e-4)


def test_losses_are_zero_when_commanded_equals_reference(robot_config):
    torch.manual_seed(34)
    ki = robot_config.kinematic_info
    batch = 4
    actions = (torch.randn(batch, ki.num_dofs) * 0.25).requires_grad_(True)
    root_quat = torch.nn.functional.normalize(torch.randn(batch, 4), dim=-1)
    cur_pos, cur_rot = _pose_from_action(robot_config, actions, root_quat)
    cur_pos, cur_rot = cur_pos.detach(), cur_rot.detach()

    agent = _make_agent(
        robot_config,
        {
            "fk_wrist_pos_weight": 1.0,
            "fk_wrist_ori_weight": 1.0,
            "fk_global_pos_weight": 1.0,
        },
    )
    batch_td = _batch(robot_config, actions, cur_pos, cur_rot, cur_pos, cur_rot)
    pos_loss, ori_loss = agent._calculate_fk_wrist_loss(batch_td, actions)
    assert pos_loss.item() < 1e-8
    assert ori_loss.item() < 1e-8
    assert agent._calculate_fk_global_loss(batch_td, actions).item() < 1e-8


def test_root_body_contributes_exactly_zero(robot_config):
    """After the anchor correction the root residual is identically zero."""
    torch.manual_seed(35)
    ki = robot_config.kinematic_info
    batch = 3
    actions = torch.randn(batch, ki.num_dofs) * 0.2
    root_quat = torch.nn.functional.normalize(torch.randn(batch, 4), dim=-1)
    cur_pos, cur_rot = _pose_from_action(robot_config, actions, root_quat)
    ref_pos, ref_rot = _random_state(robot_config, batch, seed=36)

    agent = _make_agent(
        robot_config,
        {
            "fk_global_pos_weight": 1.0,
            "fk_global_body_names": [ki.body_names[0]],
        },
    )
    batch_td = _batch(robot_config, actions, cur_pos, cur_rot, ref_pos, ref_rot)
    assert agent._calculate_fk_global_loss(batch_td, actions).item() == 0.0


# ---------------------------------------------------------------------------
# 4. the sparse masked-mimic reference is rejected (it has no root)
# ---------------------------------------------------------------------------


def test_sparse_masked_mimic_reference_is_rejected(robot_config):
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.masked_mimic import compute_target_poses_only

    ki = robot_config.kinematic_info
    cond = torch.tensor(
        [ki.body_names.index(n) for n in robot_config.trackable_bodies_subset],
        dtype=torch.long,
    )
    assert 0 not in cond.tolist(), "premise: the pelvis is not conditionable"

    agent = _make_agent(
        robot_config,
        {
            "fk_wrist_pos_weight": 1.0,
            "fk_wrist_ref_key": "masked_mimic_target_poses",
        },
    )
    agent.env.config.observation_components = dict(
        _experiment_components(),
        masked_mimic_target_poses=MdpComponent(
            compute_func=compute_target_poses_only,
            dynamic_vars={},
            static_params={
                "conditionable_body_ids": cond,
                "include_root_relative": True,
            },
        ),
    )
    with pytest.raises(ValueError, match="ROOT is not among them"):
        agent._fk_wrist_context(torch.device("cpu"))


# ---------------------------------------------------------------------------
# 5. reference layout resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "with_relative,with_velocities,expected_per_body",
    [(True, True, 24), (True, False, 18), (False, True, 15), (False, False, 9)],
)
def test_reference_stride_is_derived_not_hardcoded(
    robot_config, with_relative, with_velocities, expected_per_body
):
    ki = robot_config.kinematic_info
    batch = 3
    cur_pos, cur_rot = _random_state(robot_config, batch, seed=38)
    ref_pos, ref_rot = _random_state(robot_config, batch, seed=39)

    obs = _real_mimic_target_poses(
        cur_pos,
        cur_rot,
        ref_pos,
        ref_rot,
        with_velocities=with_velocities,
        with_relative=with_relative,
    )
    assert obs.shape[-1] == expected_per_body * ki.num_bodies

    agent = _make_agent(robot_config, {"fk_global_pos_weight": 1.0})
    agent.env.config.observation_components = {
        "mimic_target_poses": _experiment_mimic_target_poses_component(
            with_velocities=with_velocities, with_relative=with_relative
        ),
        "max_coords_obs": _experiment_max_coords_component(),
    }
    ctx = agent._fk_wrist_context(torch.device("cpu"))
    assert ctx["global_features_per_body"] == expected_per_body

    pos, rot = agent._fk_reference_blocks(
        {"mimic_target_poses": obs}, ctx, "mimic_target_poses", 0
    )
    # Independent recomputation of the root-anchored reference.
    hinv = rotations.calc_heading_quat_inv(cur_rot[:, 0], w_last=True)
    for b in range(ki.num_bodies):
        want = rotations.quat_rotate(
            hinv, ref_pos[:, b] - ref_pos[:, 0], w_last=True
        )
        assert torch.allclose(pos[:, b], want, atol=1e-5)
        want_rot = rotations.quat_to_tan_norm(
            rotations.quat_mul(hinv, ref_rot[:, b], w_last=True), w_last=True
        )
        assert torch.allclose(rot[:, b], want_rot, atol=1e-5)


def test_multi_step_reference_selects_the_right_step(robot_config):
    """The live config resolves future_steps=4; indexing must be step-major."""
    ki = robot_config.kinematic_info
    batch = 3
    cur_pos, cur_rot = _random_state(robot_config, batch, seed=41)
    per_step, obs_steps = [], []
    for s in range(3):
        ref_pos, ref_rot = _random_state(robot_config, batch, seed=50 + s)
        per_step.append(ref_pos)
        obs_steps.append(
            _real_mimic_target_poses(cur_pos, cur_rot, ref_pos, ref_rot)
        )
    obs = torch.cat(obs_steps, dim=-1)

    hinv = rotations.calc_heading_quat_inv(cur_rot[:, 0], w_last=True)
    for s in range(3):
        agent = _make_agent(
            robot_config,
            {"fk_global_pos_weight": 1.0, "fk_global_future_step": s},
        )
        ctx = agent._fk_wrist_context(torch.device("cpu"))
        got = agent._fk_global_reference({"mimic_target_poses": obs}, ctx)
        ref_pos = per_step[s]
        for b in range(ki.num_bodies):
            want = rotations.quat_rotate(
                hinv, ref_pos[:, b] - ref_pos[:, 0], w_last=True
            )
            assert torch.allclose(got[:, b], want, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. no-op at zero
# ---------------------------------------------------------------------------


def _extra_loss_with_grad(agent, batch, actions):
    extra_loss, log_dict = agent.calculate_extra_loss(batch, actions)
    if not extra_loss.requires_grad:
        return extra_loss, log_dict, None
    grad = torch.autograd.grad(
        extra_loss, actions, allow_unused=True, retain_graph=True
    )[0]
    return extra_loss, log_dict, grad


def test_all_weights_zero_is_byte_identical_and_builds_no_context(robot_config):
    torch.manual_seed(42)
    ki = robot_config.kinematic_info
    batch = 5
    actions_a = (torch.randn(batch, ki.num_dofs) * 0.1).requires_grad_(True)
    actions_b = actions_a.detach().clone().requires_grad_(True)
    cur_pos, cur_rot = _random_state(robot_config, batch, seed=43)
    ref_pos, ref_rot = _random_state(robot_config, batch, seed=44)

    legacy = _make_agent(robot_config, has_fk_fields=False)
    current = _make_agent(robot_config)

    la, loga, ga = _extra_loss_with_grad(
        legacy,
        _batch(robot_config, actions_a, cur_pos, cur_rot, ref_pos, ref_rot),
        actions_a,
    )
    lb, logb, gb = _extra_loss_with_grad(
        current,
        _batch(robot_config, actions_b, cur_pos, cur_rot, ref_pos, ref_rot),
        actions_b,
    )

    assert la.item() == lb.item() == 0.0
    assert loga == logb == {}
    assert ga is None and gb is None
    assert current._fk_wrist_ctx is None


def test_zero_weight_leaves_an_existing_extra_term_untouched(robot_config):
    torch.manual_seed(45)
    ki = robot_config.kinematic_info
    batch = 5
    actions_a = (torch.randn(batch, ki.num_dofs) * 0.1).requires_grad_(True)
    actions_b = actions_a.detach().clone().requires_grad_(True)
    previous = torch.randn(batch, ki.num_dofs) * 0.1

    legacy = _make_agent(
        robot_config, {"action_rate_weight": 0.5}, has_fk_fields=False
    )
    current = _make_agent(robot_config, {"action_rate_weight": 0.5})

    la, loga, ga = _extra_loss_with_grad(
        legacy,
        {"privileged_action": actions_a, "previous_actions": previous},
        actions_a,
    )
    lb, logb, gb = _extra_loss_with_grad(
        current,
        {"privileged_action": actions_b, "previous_actions": previous},
        actions_b,
    )
    assert la.item() == lb.item()
    assert set(loga) == set(logb) == {"supervised/action_rate_loss"}
    assert torch.equal(ga, gb)


def test_global_weight_zero_leaves_the_wrist_term_untouched(robot_config):
    torch.manual_seed(46)
    ki = robot_config.kinematic_info
    batch = 4
    actions_a = (torch.randn(batch, ki.num_dofs) * 0.2).requires_grad_(True)
    actions_b = actions_a.detach().clone().requires_grad_(True)
    cur_pos, cur_rot = _random_state(robot_config, batch, seed=47)
    ref_pos, ref_rot = _random_state(robot_config, batch, seed=48)

    legacy = _make_agent(robot_config, {"fk_wrist_pos_weight": 1.0})
    for f in (
        "fk_global_pos_weight",
        "fk_global_body_names",
        "fk_global_ref_key",
        "fk_global_future_step",
    ):
        delattr(legacy.config, f)
    current = _make_agent(
        robot_config, {"fk_wrist_pos_weight": 1.0, "fk_global_pos_weight": 0.0}
    )

    la, loga, ga = _extra_loss_with_grad(
        legacy,
        _batch(robot_config, actions_a, cur_pos, cur_rot, ref_pos, ref_rot),
        actions_a,
    )
    lb, logb, gb = _extra_loss_with_grad(
        current,
        _batch(robot_config, actions_b, cur_pos, cur_rot, ref_pos, ref_rot),
        actions_b,
    )
    assert la.item() == lb.item()
    assert set(loga) == set(logb) == {"supervised/fk_wrist_pos_loss"}
    assert torch.equal(ga, gb)


# ---------------------------------------------------------------------------
# 7. gradients and the shared FK pass
# ---------------------------------------------------------------------------


def test_gradient_reaches_the_action_tensor(robot_config):
    torch.manual_seed(49)
    ki = robot_config.kinematic_info
    batch = 4
    actions = (torch.randn(batch, ki.num_dofs) * 0.2).requires_grad_(True)
    cur_pos, cur_rot = _random_state(robot_config, batch, seed=52)
    ref_pos, ref_rot = _random_state(robot_config, batch, seed=53)

    agent = _make_agent(
        robot_config,
        {
            "fk_wrist_pos_weight": 1.0,
            "fk_wrist_ori_weight": 0.5,
            "fk_global_pos_weight": 1.0,
        },
    )
    batch_td = _batch(robot_config, actions, cur_pos, cur_rot, ref_pos, ref_rot)
    extra_loss, log_dict = agent.calculate_extra_loss(batch_td, actions)
    assert {
        "supervised/fk_wrist_pos_loss",
        "supervised/fk_wrist_ori_loss",
        "supervised/fk_global_pos_loss",
    } <= set(log_dict)

    grad = torch.autograd.grad(extra_loss, actions)[0]
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0

    dof_names = list(ki.dof_names)
    leg = [
        i
        for i, n in enumerate(dof_names)
        if "hip" in n or "knee" in n or "ankle" in n
    ]
    arm = [
        i
        for i, n in enumerate(dof_names)
        if "shoulder" in n or "elbow" in n or "wrist" in n
    ]
    assert grad[:, arm].abs().sum() > 0
    # the all-body term is what puts gradient on the leg chain
    assert grad[:, leg].abs().sum() > 0


def test_wrist_only_puts_no_gradient_on_the_leg_chain(robot_config):
    torch.manual_seed(54)
    ki = robot_config.kinematic_info
    batch = 4
    actions = (torch.randn(batch, ki.num_dofs) * 0.2).requires_grad_(True)
    cur_pos, cur_rot = _random_state(robot_config, batch, seed=55)
    ref_pos, ref_rot = _random_state(robot_config, batch, seed=56)

    agent = _make_agent(robot_config, {"fk_wrist_pos_weight": 1.0})
    batch_td = _batch(robot_config, actions, cur_pos, cur_rot, ref_pos, ref_rot)
    extra_loss, _ = agent.calculate_extra_loss(batch_td, actions)
    grad = torch.autograd.grad(extra_loss, actions)[0]

    leg = [
        i
        for i, n in enumerate(ki.dof_names)
        if "hip" in n or "knee" in n or "ankle" in n
    ]
    assert grad[:, leg].abs().sum() == 0


def test_both_terms_share_one_fk_pass(robot_config):
    torch.manual_seed(57)
    ki = robot_config.kinematic_info
    batch = 3
    actions = (torch.randn(batch, ki.num_dofs) * 0.2).requires_grad_(True)
    cur_pos, cur_rot = _random_state(robot_config, batch, seed=58)
    ref_pos, ref_rot = _random_state(robot_config, batch, seed=59)

    agent = _make_agent(
        robot_config,
        {
            "fk_wrist_pos_weight": 1.0,
            "fk_wrist_ori_weight": 1.0,
            "fk_global_pos_weight": 1.0,
        },
    )
    calls = {"n": 0}
    original = SupervisedAgent._fk_commanded_body_poses

    def counting(self, *a, **k):
        calls["n"] += 1
        return original(self, *a, **k)

    agent._fk_commanded_body_poses = counting.__get__(agent, SupervisedAgent)
    agent.calculate_extra_loss(
        _batch(robot_config, actions, cur_pos, cur_rot, ref_pos, ref_rot), actions
    )
    assert calls["n"] == 1


def test_context_is_built_once_and_reused(robot_config):
    torch.manual_seed(60)
    ki = robot_config.kinematic_info
    batch = 3
    actions = torch.randn(batch, ki.num_dofs) * 0.2
    cur_pos, cur_rot = _random_state(robot_config, batch, seed=61)
    ref_pos, ref_rot = _random_state(robot_config, batch, seed=62)

    agent = _make_agent(robot_config, {"fk_global_pos_weight": 1.0})
    batch_td = _batch(robot_config, actions, cur_pos, cur_rot, ref_pos, ref_rot)
    assert agent._fk_wrist_ctx is None
    agent._calculate_fk_global_loss(batch_td, actions)
    first = agent._fk_wrist_ctx
    assert first is not None
    agent._calculate_fk_global_loss(batch_td, actions)
    assert agent._fk_wrist_ctx is first


# ---------------------------------------------------------------------------
# 8. the root-rotation reader (2026-08-04 regression: 433 vs 520)
# ---------------------------------------------------------------------------


def test_real_max_coords_obs_width_matches_the_readers_expectation(robot_config):
    batch = 4
    cur_pos, cur_rot = _random_state(robot_config, batch, seed=63)
    obs, _ = _real_max_coords_obs(robot_config, cur_pos, cur_rot)

    agent = _make_agent(robot_config, {"fk_wrist_pos_weight": 1.0})
    ctx = agent._fk_wrist_context(torch.device("cpu"))
    nb = robot_config.kinematic_info.num_bodies
    assert obs.shape[-1] == 1 + 3 * (nb - 1) + 12 * nb
    assert obs.shape[-1] == ctx["root_rot_min_width"]
    assert ctx["root_rot_offset"] == 1 + 3 * (nb - 1)


def test_reader_recovers_root_rotation_from_the_real_observation(robot_config):
    batch = 5
    cur_pos, cur_rot = _random_state(robot_config, batch, seed=64)
    obs, expected = _real_max_coords_obs(robot_config, cur_pos, cur_rot)

    agent = _make_agent(robot_config, {"fk_wrist_pos_weight": 1.0})
    ctx = agent._fk_wrist_context(torch.device("cpu"))
    got = agent._fk_wrist_root_heading_local_rot(
        {"max_coords_obs": obs}, ctx, torch.zeros(batch, 2, 3)
    )
    assert torch.allclose(got, expected, atol=1e-5)
    assert (expected - torch.eye(3)).abs().max() > 0.02


def test_contact_tail_does_not_disturb_the_reader(robot_config):
    batch = 3
    cur_pos, cur_rot = _random_state(robot_config, batch, seed=65)
    obs, expected = _real_max_coords_obs(
        robot_config, cur_pos, cur_rot, observe_contacts=True
    )
    agent = _make_agent(robot_config, {"fk_wrist_pos_weight": 1.0})
    ctx = agent._fk_wrist_context(torch.device("cpu"))
    assert obs.shape[-1] > ctx["root_rot_min_width"]
    got = agent._fk_wrist_root_heading_local_rot(
        {"max_coords_obs": obs}, ctx, torch.zeros(batch, 2, 3)
    )
    assert torch.allclose(got, expected, atol=1e-5)


def test_world_frame_observation_is_rejected_up_front(robot_config):
    agent = _make_agent(robot_config, {"fk_wrist_pos_weight": 1.0})
    agent.env.config.observation_components = dict(
        _experiment_components(),
        max_coords_obs=_experiment_max_coords_component(local_obs=False),
    )
    with pytest.raises(ValueError, match="local_obs=False"):
        agent._fk_wrist_context(torch.device("cpu"))


def test_frame_mismatch_between_reference_and_root_rotation_is_flagged(
    robot_config, caplog
):
    """The live recipe's trap: clean targets + noisy proprioception."""
    import logging

    components = {
        "max_coords_obs": _experiment_max_coords_component(use_noisy=True),
        "clean_max_coords_obs": _experiment_max_coords_component(use_noisy=False),
        "mimic_target_poses": _experiment_mimic_target_poses_component(
            use_noisy=False
        ),
    }
    agent = _make_agent(robot_config, {"fk_global_pos_weight": 1.0})
    agent.env.config.observation_components = components
    with caplog.at_level(logging.WARNING):
        agent._fk_wrist_context(torch.device("cpu"))
    assert "FRAME MISMATCH" in caplog.text

    caplog.clear()
    fixed = _make_agent(
        robot_config,
        {
            "fk_global_pos_weight": 1.0,
            "fk_wrist_root_rot_obs_key": "clean_max_coords_obs",
        },
    )
    fixed.env.config.observation_components = components
    with caplog.at_level(logging.WARNING):
        fixed._fk_wrist_context(torch.device("cpu"))
    assert "FRAME MISMATCH" not in caplog.text


# ---------------------------------------------------------------------------
# 9. config surface
# ---------------------------------------------------------------------------


def test_config_defaults_are_off():
    config = SupervisedAgentConfig(batch_size=1, training_max_steps=1)
    assert config.fk_wrist_pos_weight == 0.0
    assert config.fk_wrist_ori_weight == 0.0
    assert config.fk_wrist_body_names is None
    assert config.fk_wrist_ref_key == "mimic_target_poses"
    assert config.fk_wrist_root_rot_obs_key == "max_coords_obs"
    assert config.fk_wrist_future_step == 0
    assert config.fk_global_pos_weight == 0.0
    assert config.fk_global_body_names is None
    assert config.fk_global_ref_key == "mimic_target_poses"
    assert config.fk_global_future_step == 0
    # the masked-reference mask key is gone: that path is structurally invalid
    assert not hasattr(config, "fk_wrist_ref_mask_key")


def test_missing_reference_key_raises_a_named_error(robot_config):
    ki = robot_config.kinematic_info
    actions = torch.zeros(2, ki.num_dofs)
    agent = _make_agent(robot_config, {"fk_wrist_pos_weight": 1.0})
    with pytest.raises(KeyError, match="mimic_target_poses"):
        agent._calculate_fk_wrist_loss({"privileged_action": actions}, actions)
