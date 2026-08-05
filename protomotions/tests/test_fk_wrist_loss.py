# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard tests for the FK Cartesian wrist loss (supervised student).

Covers, in order:
  * ``pose_lib``'s batched torch FK matches MuJoCo ``mj_forward`` on the very
    same MJCF (position AND rotation), and is differentiable;
  * ``fk_wrist_*_weight = 0`` leaves total extra loss, log dict and gradients
    byte-identical to a config that has never heard of the fields;
  * the loss is (numerically) zero when the commanded wrist pose equals the
    reference, in the pelvis frame AND under a tilted root;
  * the visibility mask gates the term;
  * the gradient reaches the action tensor.
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
# helpers
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


def _make_agent(robot_config, config_overrides=None, has_fk_fields=True):
    """A SupervisedAgent shell with just enough env/config for extra losses."""
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
            fk_wrist_ref_key="masked_mimic_target_poses",
            fk_wrist_ref_mask_key="masked_mimic_target_masks",
            fk_wrist_root_rot_obs_key="max_coords_obs",
            fk_wrist_future_step=0,
        )
    fields.update(config_overrides or {})
    agent.config = SimpleNamespace(**fields)
    agent.device = torch.device("cpu")
    agent.current_epoch = 0
    agent.env = SimpleNamespace(
        robot_config=robot_config,
        config=SimpleNamespace(
            observation_components={},
            action_config=make_pd_action_config(robot_config),
        ),
    )
    return agent


def _pack_reference(robot_config, ref_pos, ref_tan_norm, num_steps=3, step=0):
    """Build a synthetic ``masked_mimic_target_poses`` / mask pair.

    Layout mirrors ``build_sparse_target_poses(include_root_relative=True)``:
    ``[envs, steps, conditionable_bodies, 2, 12]`` with the root-relative
    halves at ``[..., 0, 6:9]`` (position) and ``[..., 1, 6:12]`` (rotation).
    """
    body_names = robot_config.kinematic_info.body_names
    cond = [body_names.index(n) for n in robot_config.trackable_bodies_subset]
    slots = [cond.index(body_names.index(n)) for n in WRISTS]
    batch = ref_pos.shape[0]

    obs = torch.zeros(batch, num_steps, len(cond), 2, 12)
    masks = torch.zeros(batch, num_steps, len(cond), 2)
    for w, slot in enumerate(slots):
        obs[:, step, slot, 0, 6:9] = ref_pos[:, w]
        obs[:, step, slot, 1, 6:12] = ref_tan_norm[:, w]
        masks[:, step, slot, :] = 1.0
    return obs.reshape(batch, -1), masks.reshape(batch, -1)


def _real_max_coords_obs(robot_config, root_world_quat, observe_contacts=False):
    """Build ``max_coords_obs`` with the REAL kernel, not a hand-rolled tensor.

    The original version of this helper synthesized a tensor from the same
    (wrong) width formula the reader used, so the suite happily agreed with a
    bug that killed the live run at its first optimize step. Everything the
    reader is tested against now comes out of
    ``compute_humanoid_max_coords_observations`` itself.

    Args:
        root_world_quat: the root's WORLD rotation [B, 4], w-last.

    Returns:
        ``(obs, root_heading_local_mat)`` -- the observation, and the
        heading-local root rotation the reader is expected to recover. The
        expectation is derived from ``calc_heading_quat_inv``, the kernel's own
        definition, NOT assumed: a tilt about a non-vertical axis still carries
        some heading, so ``heading_inv * root_world`` is not simply "the tilt".
    """
    from protomotions.envs.obs.humanoid import (
        compute_humanoid_max_coords_observations,
    )

    num_bodies = robot_config.kinematic_info.num_bodies
    batch = root_world_quat.shape[0]

    body_rot = torch.nn.functional.normalize(
        torch.randn(batch, num_bodies, 4), dim=-1
    )
    body_rot[:, 0] = root_world_quat
    obs = compute_humanoid_max_coords_observations(
        body_pos=torch.randn(batch, num_bodies, 3),
        body_rot=body_rot,
        body_vel=torch.randn(batch, num_bodies, 3),
        body_ang_vel=torch.randn(batch, num_bodies, 3),
        ground_height=torch.zeros(batch),
        body_contacts=torch.zeros(batch, 2),
        local_obs=True,
        root_height_obs=True,
        observe_contacts=observe_contacts,
        w_last=True,
    )
    heading_local_quat = rotations.quat_mul(
        rotations.calc_heading_quat_inv(root_world_quat, w_last=True),
        root_world_quat,
        w_last=True,
    )
    return obs, pose_lib.quaternion_to_matrix(heading_local_quat, w_last=True)


def _wrist_indices(robot_config):
    body_names = robot_config.kinematic_info.body_names
    return [body_names.index(n) for n in WRISTS]


# ---------------------------------------------------------------------------
# 1. FK correctness against an independent implementation (MuJoCo)
# ---------------------------------------------------------------------------


def test_pose_lib_fk_matches_mujoco_forward(robot_config):
    """pose_lib FK == MuJoCo mj_forward on the same MJCF, pos and rot."""
    mujoco = pytest.importorskip("mujoco")

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    kinematic_info = robot_config.kinematic_info

    mj_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    mj_body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]
    # Free joint first, worldbody first: both orderings must line up with the
    # KinematicInfo the robot config parsed out of the same file.
    assert mj_joint_names[1:] == list(kinematic_info.dof_names)
    assert mj_body_names[1:] == list(kinematic_info.body_names)

    rng = np.random.RandomState(0)
    for trial in range(3):
        joints = rng.uniform(-0.4, 0.4, size=kinematic_info.num_dofs)
        data.qpos[:3] = 0.0
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:] = joints
        mujoco.mj_forward(model, data)

        torch_joints = torch.tensor(joints, dtype=torch.float32)[None]
        pos, rot = _fk(kinematic_info, torch_joints)

        for name in kinematic_info.body_names:
            i = kinematic_info.body_names.index(name)
            j = mj_body_names.index(name)
            assert np.allclose(
                pos[0, i].numpy(), data.xpos[j], atol=1e-5
            ), f"trial {trial}: position mismatch on {name}"
            assert np.allclose(
                rot[0, i].numpy(), data.xmat[j].reshape(3, 3), atol=1e-5
            ), f"trial {trial}: rotation mismatch on {name}"


def test_pose_lib_fk_is_differentiable(robot_config):
    """The FK used by the loss must propagate gradients to the joint targets."""
    kinematic_info = robot_config.kinematic_info
    joints = (torch.randn(4, kinematic_info.num_dofs) * 0.2).requires_grad_(True)
    pos, rot = _fk(kinematic_info, joints)
    left = kinematic_info.body_names.index("left_wrist_yaw_link")
    pos[:, left].pow(2).sum().backward()
    assert joints.grad is not None
    assert torch.isfinite(joints.grad).all()
    assert joints.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 2. weight = 0 is byte-identical (and pays no FK cost)
# ---------------------------------------------------------------------------


def _extra_loss_with_grad(agent, batch, actions):
    extra_loss, log_dict = agent.calculate_extra_loss(batch, actions)
    if not extra_loss.requires_grad:
        # No enabled term touches the action at all -> no gradient path.
        return extra_loss, log_dict, None
    grad = torch.autograd.grad(
        extra_loss, actions, allow_unused=True, retain_graph=True
    )[0]
    return extra_loss, log_dict, grad


def test_zero_weight_is_byte_identical_to_config_without_the_fields(robot_config):
    torch.manual_seed(0)
    num_dofs = robot_config.kinematic_info.num_dofs
    actions_a = (torch.randn(6, num_dofs) * 0.1).requires_grad_(True)
    actions_b = actions_a.detach().clone().requires_grad_(True)

    ref_pos = torch.randn(6, 2, 3)
    ref_tan = torch.randn(6, 2, 6)
    ref, masks = _pack_reference(robot_config, ref_pos, ref_tan)
    root_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(6, 1)

    def make_batch(actions):
        return {
            "privileged_action": actions,
            "expert_actions": torch.zeros_like(actions),
            "previous_actions": torch.zeros_like(actions),
            "masked_mimic_target_poses": ref,
            "masked_mimic_target_masks": masks,
            "max_coords_obs": _real_max_coords_obs(robot_config, root_quat)[0],
        }

    # Old pickled config: the fields do not exist at all.
    legacy = _make_agent(robot_config, has_fk_fields=False)
    # New config, fields present at their 0.0 defaults.
    current = _make_agent(robot_config)

    loss_legacy, log_legacy, grad_legacy = _extra_loss_with_grad(
        legacy, make_batch(actions_a), actions_a
    )
    loss_current, log_current, grad_current = _extra_loss_with_grad(
        current, make_batch(actions_b), actions_b
    )

    assert loss_legacy.item() == loss_current.item() == 0.0
    assert log_legacy == log_current == {}
    # No FK term means no path from extra_loss to the action at all.
    assert grad_legacy is None and grad_current is None
    # And the FK context is never built -> zero FK cost paid.
    assert current._fk_wrist_ctx is None
    assert "supervised/fk_wrist_pos_loss" not in log_current
    assert "supervised/fk_wrist_ori_loss" not in log_current


def test_zero_weight_leaves_an_existing_extra_term_untouched(robot_config):
    """A live extra term (action_rate) must be bit-identical with FK at 0."""
    torch.manual_seed(1)
    num_dofs = robot_config.kinematic_info.num_dofs
    actions_a = (torch.randn(5, num_dofs) * 0.1).requires_grad_(True)
    actions_b = actions_a.detach().clone().requires_grad_(True)
    previous = torch.randn(5, num_dofs) * 0.1

    def make_batch(actions):
        return {
            "privileged_action": actions,
            "previous_actions": previous,
        }

    legacy = _make_agent(
        robot_config, {"action_rate_weight": 0.5}, has_fk_fields=False
    )
    current = _make_agent(robot_config, {"action_rate_weight": 0.5})

    loss_a, log_a, grad_a = _extra_loss_with_grad(
        legacy, make_batch(actions_a), actions_a
    )
    loss_b, log_b, grad_b = _extra_loss_with_grad(
        current, make_batch(actions_b), actions_b
    )

    assert loss_a.item() == loss_b.item()
    assert set(log_a) == set(log_b) == {"supervised/action_rate_loss"}
    assert torch.equal(grad_a, grad_b)


# ---------------------------------------------------------------------------
# 3. loss == 0 when the commanded wrist pose equals the reference
# ---------------------------------------------------------------------------


def _commanded_wrist_pose(robot_config, actions, root_rot_mat=None):
    """FK(action) wrist pose, optionally rotated into the heading-local frame."""
    action_config = make_pd_action_config(robot_config)
    params = {k: v for k, v in action_config.items() if k != "fn"}
    joint_targets = action_config["fn"](actions, **params)["processed_action"]
    pos, rot = _fk(robot_config.kinematic_info, joint_targets)
    idx = _wrist_indices(robot_config)
    pos = pos[:, idx]
    rot = rot[:, idx]
    if root_rot_mat is not None:
        pos = torch.einsum("bij,bwj->bwi", root_rot_mat, pos)
        rot = torch.einsum("bij,bwjk->bwik", root_rot_mat, rot)
    tan_norm = torch.cat((rot[..., :, 0], rot[..., :, 2]), dim=-1)
    return pos, tan_norm


def test_loss_is_zero_when_reference_equals_commanded_pose_pelvis_frame(
    robot_config,
):
    """fk_wrist_root_rot_obs_key=None => pure pelvis frame, no obs needed."""
    torch.manual_seed(2)
    num_dofs = robot_config.kinematic_info.num_dofs
    actions = (torch.randn(4, num_dofs) * 0.3).requires_grad_(True)

    with torch.no_grad():
        ref_pos, ref_tan = _commanded_wrist_pose(robot_config, actions)
    ref, masks = _pack_reference(robot_config, ref_pos, ref_tan)

    agent = _make_agent(
        robot_config,
        {
            "fk_wrist_pos_weight": 1.0,
            "fk_wrist_ori_weight": 1.0,
            "fk_wrist_root_rot_obs_key": None,
        },
    )
    batch = {
        "privileged_action": actions,
        "masked_mimic_target_poses": ref,
        "masked_mimic_target_masks": masks,
    }
    pos_loss, ori_loss = agent._calculate_fk_wrist_loss(batch, actions)
    assert pos_loss.item() < 1e-10
    assert ori_loss.item() < 1e-10


def test_loss_is_zero_under_a_tilted_root(robot_config):
    """The heading-local correction must be applied, not assumed identity."""
    torch.manual_seed(3)
    num_dofs = robot_config.kinematic_info.num_dofs
    batch_size = 4
    actions = (torch.randn(batch_size, num_dofs) * 0.3).requires_grad_(True)

    # A genuine pelvis tilt (roll+pitch), the part a heading-only frame keeps.
    angle = torch.full((batch_size,), 0.25)
    axis = torch.nn.functional.normalize(
        torch.tensor([[0.4, 0.9, 0.0]]).repeat(batch_size, 1), dim=-1
    )
    root_quat = rotations.quat_from_angle_axis(angle, axis, w_last=True)
    # Observation first: it defines what the heading-local rotation actually is.
    max_coords_obs, root_mat = _real_max_coords_obs(robot_config, root_quat)

    with torch.no_grad():
        ref_pos, ref_tan = _commanded_wrist_pose(robot_config, actions, root_mat)
    ref, masks = _pack_reference(robot_config, ref_pos, ref_tan)

    agent = _make_agent(
        robot_config,
        {"fk_wrist_pos_weight": 1.0, "fk_wrist_ori_weight": 1.0},
    )
    batch = {
        "privileged_action": actions,
        "masked_mimic_target_poses": ref,
        "masked_mimic_target_masks": masks,
        "max_coords_obs": max_coords_obs,
    }
    pos_loss, ori_loss = agent._calculate_fk_wrist_loss(batch, actions)
    assert pos_loss.item() < 1e-8
    assert ori_loss.item() < 1e-8

    # Ignoring the tilt is NOT free: the same batch scored in the pelvis frame
    # must be materially wrong (the wrists sit ~0.5 m from the pelvis).
    naive = _make_agent(
        robot_config,
        {
            "fk_wrist_pos_weight": 1.0,
            "fk_wrist_ori_weight": 1.0,
            "fk_wrist_root_rot_obs_key": None,
        },
    )
    naive_pos, naive_ori = naive._calculate_fk_wrist_loss(batch, actions)
    assert naive_pos.item() > 1e-3
    assert naive_ori.item() > 1e-3


def test_loss_grows_with_reference_offset(robot_config):
    """Position loss is the squared metric error in the comparison frame."""
    torch.manual_seed(4)
    num_dofs = robot_config.kinematic_info.num_dofs
    actions = torch.randn(3, num_dofs) * 0.2

    with torch.no_grad():
        ref_pos, ref_tan = _commanded_wrist_pose(robot_config, actions)
    offset = torch.zeros_like(ref_pos)
    offset[..., 0] = 0.10  # 10 cm along x, on every scored wrist
    ref, masks = _pack_reference(robot_config, ref_pos + offset, ref_tan)

    agent = _make_agent(
        robot_config,
        {"fk_wrist_pos_weight": 1.0, "fk_wrist_root_rot_obs_key": None},
    )
    batch = {
        "privileged_action": actions,
        "masked_mimic_target_poses": ref,
        "masked_mimic_target_masks": masks,
    }
    pos_loss, _ = agent._calculate_fk_wrist_loss(batch, actions)
    assert pos_loss.item() == pytest.approx(0.10**2, rel=1e-4)


# ---------------------------------------------------------------------------
# 4. masking and gradient flow
# ---------------------------------------------------------------------------


def test_masked_out_targets_do_not_contribute(robot_config):
    torch.manual_seed(5)
    num_dofs = robot_config.kinematic_info.num_dofs
    actions = torch.randn(4, num_dofs) * 0.2

    with torch.no_grad():
        ref_pos, ref_tan = _commanded_wrist_pose(robot_config, actions)
    bad_ref_pos = ref_pos + 1.0
    ref, masks = _pack_reference(robot_config, bad_ref_pos, ref_tan)

    agent = _make_agent(
        robot_config,
        {
            "fk_wrist_pos_weight": 1.0,
            "fk_wrist_ori_weight": 1.0,
            "fk_wrist_root_rot_obs_key": None,
        },
    )
    batch = {
        "privileged_action": actions,
        "masked_mimic_target_poses": ref,
        "masked_mimic_target_masks": masks,
    }
    visible_pos, _ = agent._calculate_fk_wrist_loss(batch, actions)
    assert visible_pos.item() > 1.0

    batch["masked_mimic_target_masks"] = torch.zeros_like(masks)
    masked_pos, masked_ori = agent._calculate_fk_wrist_loss(batch, actions)
    assert masked_pos.item() == 0.0
    assert masked_ori.item() == 0.0


def test_gradient_reaches_the_action_tensor(robot_config):
    torch.manual_seed(6)
    num_dofs = robot_config.kinematic_info.num_dofs
    actions = (torch.randn(4, num_dofs) * 0.2).requires_grad_(True)

    ref_pos = torch.randn(4, 2, 3) * 0.2
    ref_tan = torch.randn(4, 2, 6) * 0.2
    ref, masks = _pack_reference(robot_config, ref_pos, ref_tan)
    root_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(4, 1)

    agent = _make_agent(
        robot_config,
        {"fk_wrist_pos_weight": 1.0, "fk_wrist_ori_weight": 0.5},
    )
    batch = {
        "privileged_action": actions,
        "masked_mimic_target_poses": ref,
        "masked_mimic_target_masks": masks,
        "max_coords_obs": _real_max_coords_obs(robot_config, root_quat)[0],
    }

    extra_loss, log_dict = agent.calculate_extra_loss(batch, actions)
    assert "supervised/fk_wrist_pos_loss" in log_dict
    assert "supervised/fk_wrist_ori_loss" in log_dict

    grad = torch.autograd.grad(extra_loss, actions)[0]
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0

    # Arm DOFs must carry (most of) the signal: the leg chain has no path to a
    # wrist, so its gradient is exactly zero.
    dof_names = list(robot_config.kinematic_info.dof_names)
    arm = [i for i, n in enumerate(dof_names) if "shoulder" in n or "elbow" in n or "wrist" in n]
    leg = [i for i, n in enumerate(dof_names) if "hip" in n or "knee" in n or "ankle" in n]
    assert grad[:, arm].abs().sum() > 0
    assert grad[:, leg].abs().sum() == 0


def test_context_is_built_once_and_reused(robot_config):
    """Kinematic info must be precomputed, never rebuilt per step."""
    torch.manual_seed(7)
    num_dofs = robot_config.kinematic_info.num_dofs
    actions = torch.randn(3, num_dofs) * 0.2
    ref, masks = _pack_reference(
        robot_config, torch.zeros(3, 2, 3), torch.zeros(3, 2, 6)
    )

    agent = _make_agent(
        robot_config,
        {"fk_wrist_pos_weight": 1.0, "fk_wrist_root_rot_obs_key": None},
    )
    batch = {
        "privileged_action": actions,
        "masked_mimic_target_poses": ref,
        "masked_mimic_target_masks": masks,
    }
    assert agent._fk_wrist_ctx is None
    agent._calculate_fk_wrist_loss(batch, actions)
    first = agent._fk_wrist_ctx
    assert first is not None
    agent._calculate_fk_wrist_loss(batch, actions)
    assert agent._fk_wrist_ctx is first


# ---------------------------------------------------------------------------
# 5. the reference slicing matches the REAL observation kernel
# ---------------------------------------------------------------------------


def test_reference_slice_matches_the_real_masked_mimic_obs_kernel(robot_config):
    """Guard the layout assumption against compute_target_poses_only itself.

    If build_sparse_target_poses ever reorders its blocks, this fails loudly
    instead of silently training against the wrong six numbers.
    """
    from protomotions.envs.obs.masked_mimic import compute_target_poses_only

    torch.manual_seed(8)
    body_names = robot_config.kinematic_info.body_names
    num_bodies = len(body_names)
    num_envs, num_steps = 3, 4
    cond = torch.tensor(
        [body_names.index(n) for n in robot_config.trackable_bodies_subset],
        dtype=torch.long,
    )

    cur_pos = torch.randn(num_envs, num_bodies, 3)
    cur_rot = torch.nn.functional.normalize(
        torch.randn(num_envs, num_bodies, 4), dim=-1
    )
    ref_pos = torch.randn(num_envs, num_steps, num_bodies, 3)
    ref_rot = torch.nn.functional.normalize(
        torch.randn(num_envs, num_steps, num_bodies, 4), dim=-1
    )
    masks = torch.ones(num_envs, num_steps * len(cond) * 2)

    obs = compute_target_poses_only(
        current_state_body_pos=cur_pos,
        current_state_body_rot=cur_rot,
        masked_mimic_ref_pos=ref_pos,
        masked_mimic_ref_rot=ref_rot,
        masked_mimic_target_bodies_masks=masks,
        conditionable_body_ids=cond,
        include_root_relative=True,
    )

    step = 2
    agent = _make_agent(robot_config, {"fk_wrist_future_step": step})
    ctx = agent._fk_wrist_context(torch.device("cpu"))
    got_pos, got_tan, pos_mask, rot_mask = agent._fk_wrist_reference(
        {"masked_mimic_target_poses": obs, "masked_mimic_target_masks": masks},
        ctx,
    )

    # Expected: (ref_wrist - current_root) rotated by the CURRENT root's
    # inverse heading -- root-relative, heading-local.
    heading_inv = rotations.calc_heading_quat_inv(cur_rot[:, 0], w_last=True)
    for w, name in enumerate(WRISTS):
        b = body_names.index(name)
        delta = ref_pos[:, step, b] - cur_pos[:, 0]
        expected_pos = rotations.quat_rotate(heading_inv, delta, w_last=True)
        expected_tan = rotations.quat_to_tan_norm(
            rotations.quat_mul(heading_inv, ref_rot[:, step, b], w_last=True),
            w_last=True,
        )
        assert torch.allclose(got_pos[:, w], expected_pos, atol=1e-5)
        assert torch.allclose(got_tan[:, w], expected_tan, atol=1e-5)

    assert torch.equal(pos_mask, torch.ones_like(pos_mask))
    assert torch.equal(rot_mask, torch.ones_like(rot_mask))


# ---------------------------------------------------------------------------
# 6. the root-rotation reader against the REAL experiment observation spec
#
# REGRESSION (2026-08-05): the live 8-rank run died at its first optimize step
# with "'max_coords_obs' width 433 is too small ... expected >= 520". The reader
# validated against 1 + 3*(NB-1) + 15*NB; the kernel emits
# 1 + 3*(NB-1) + 12*NB (6*NB rotation + 3*NB vel + 3*NB ang_vel). The suite
# passed anyway because its helper BUILT the tensor from the same wrong formula.
# These tests take the observation from the real factory + real kernel.
# ---------------------------------------------------------------------------


def _experiment_max_coords_component(use_noisy=False, local_obs=True):
    """The exact component this experiment's config builds."""
    from protomotions.envs.component_factories import max_coords_obs_factory

    return max_coords_obs_factory(
        use_noisy=use_noisy,
        local_obs=local_obs,
        root_height_obs=True,
        observe_contacts=False,
    )


def test_real_max_coords_obs_width_matches_the_readers_expectation(robot_config):
    """The width the kernel emits must be the width the reader demands."""
    root_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(4, 1)
    obs, _ = _real_max_coords_obs(robot_config, root_quat)

    agent = _make_agent(robot_config, {"fk_wrist_pos_weight": 1.0})
    ctx = agent._fk_wrist_context(torch.device("cpu"))

    num_bodies = robot_config.kinematic_info.num_bodies
    assert obs.shape[-1] == 1 + 3 * (num_bodies - 1) + 12 * num_bodies
    assert obs.shape[-1] == ctx["root_rot_min_width"]
    assert ctx["root_rot_offset"] == 1 + 3 * (num_bodies - 1)


def test_reader_recovers_root_rotation_from_the_real_observation(robot_config):
    """End-to-end: real factory + real kernel -> reader returns the tilt."""
    torch.manual_seed(9)
    batch = 5
    angle = torch.linspace(0.05, 0.6, batch)
    axis = torch.nn.functional.normalize(
        torch.tensor([[0.3, 0.8, 0.0]]).repeat(batch, 1), dim=-1
    )
    root_quat = rotations.quat_from_angle_axis(angle, axis, w_last=True)

    agent = _make_agent(robot_config, {"fk_wrist_pos_weight": 1.0})
    agent.env.config.observation_components = {
        "max_coords_obs": _experiment_max_coords_component()
    }
    ctx = agent._fk_wrist_context(torch.device("cpu"))

    obs, expected = _real_max_coords_obs(robot_config, root_quat)
    got = agent._fk_wrist_root_heading_local_rot(
        {"max_coords_obs": obs}, ctx, torch.zeros(batch, 2, 3)
    )
    assert torch.allclose(got, expected, atol=1e-5)
    # And it is a genuine tilt, not a degenerate identity that would pass anyway.
    assert (expected - torch.eye(3)).abs().max() > 0.02


def test_full_loss_runs_against_the_real_observation_spec(robot_config):
    """The exact path the live run took: nothing hand-rolled but the reference."""
    torch.manual_seed(10)
    num_dofs = robot_config.kinematic_info.num_dofs
    batch = 4
    actions = (torch.randn(batch, num_dofs) * 0.2).requires_grad_(True)

    angle = torch.full((batch,), 0.3)
    axis = torch.nn.functional.normalize(
        torch.tensor([[0.5, 0.7, 0.0]]).repeat(batch, 1), dim=-1
    )
    root_quat = rotations.quat_from_angle_axis(angle, axis, w_last=True)
    max_coords_obs, root_mat = _real_max_coords_obs(robot_config, root_quat)

    with torch.no_grad():
        ref_pos, ref_tan = _commanded_wrist_pose(robot_config, actions, root_mat)
    ref, masks = _pack_reference(robot_config, ref_pos, ref_tan)

    agent = _make_agent(
        robot_config,
        {"fk_wrist_pos_weight": 1.0, "fk_wrist_ori_weight": 1.0},
    )
    agent.env.config.observation_components = {
        "max_coords_obs": _experiment_max_coords_component()
    }
    batch_td = {
        "privileged_action": actions,
        "masked_mimic_target_poses": ref,
        "masked_mimic_target_masks": masks,
        "max_coords_obs": max_coords_obs,
    }

    extra_loss, log_dict = agent.calculate_extra_loss(batch_td, actions)
    assert log_dict["supervised/fk_wrist_pos_loss"].item() < 1e-8
    assert log_dict["supervised/fk_wrist_ori_loss"].item() < 1e-8
    assert torch.isfinite(extra_loss)


def test_contact_tail_does_not_disturb_the_reader(robot_config):
    """observe_contacts appends AFTER everything read; must still resolve."""
    torch.manual_seed(11)
    batch = 3
    angle = torch.full((batch,), 0.2)
    axis = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0, 0.0]]).repeat(batch, 1), dim=-1
    )
    root_quat = rotations.quat_from_angle_axis(angle, axis, w_last=True)

    agent = _make_agent(robot_config, {"fk_wrist_pos_weight": 1.0})
    ctx = agent._fk_wrist_context(torch.device("cpu"))

    obs, expected = _real_max_coords_obs(
        robot_config, root_quat, observe_contacts=True
    )
    assert obs.shape[-1] > ctx["root_rot_min_width"]
    got = agent._fk_wrist_root_heading_local_rot(
        {"max_coords_obs": obs}, ctx, torch.zeros(batch, 2, 3)
    )
    assert torch.allclose(got, expected, atol=1e-5)


def test_world_frame_observation_is_rejected_up_front(robot_config):
    """local_obs=False has the SAME width but the wrong frame -> must raise."""
    agent = _make_agent(robot_config, {"fk_wrist_pos_weight": 1.0})
    agent.env.config.observation_components = {
        "max_coords_obs": _experiment_max_coords_component(local_obs=False)
    }
    with pytest.raises(ValueError, match="local_obs=False"):
        agent._fk_wrist_context(torch.device("cpu"))


def test_noisy_root_rotation_source_is_flagged(robot_config, caplog):
    """This run's max_coords_obs is use_noisy=True; the clean twin is preferred."""
    import logging

    agent = _make_agent(robot_config, {"fk_wrist_pos_weight": 1.0})
    agent.env.config.observation_components = {
        "max_coords_obs": _experiment_max_coords_component(use_noisy=True),
        "clean_max_coords_obs": _experiment_max_coords_component(use_noisy=False),
    }
    with caplog.at_level(logging.WARNING):
        agent._fk_wrist_context(torch.device("cpu"))
    assert "clean_max_coords_obs" in caplog.text

    # The clean twin must NOT warn.
    caplog.clear()
    clean_agent = _make_agent(
        robot_config,
        {
            "fk_wrist_pos_weight": 1.0,
            "fk_wrist_root_rot_obs_key": "clean_max_coords_obs",
        },
    )
    clean_agent.env.config.observation_components = (
        agent.env.config.observation_components
    )
    with caplog.at_level(logging.WARNING):
        clean_agent._fk_wrist_context(torch.device("cpu"))
    assert "clean_max_coords_obs" not in caplog.text


# ---------------------------------------------------------------------------
# 7. config surface
# ---------------------------------------------------------------------------


def test_config_defaults_are_off():
    config = SupervisedAgentConfig(batch_size=1, training_max_steps=1)
    assert config.fk_wrist_pos_weight == 0.0
    assert config.fk_wrist_ori_weight == 0.0
    assert config.fk_wrist_body_names is None
    assert config.fk_wrist_ref_key == "masked_mimic_target_poses"
    assert config.fk_wrist_ref_mask_key == "masked_mimic_target_masks"
    assert config.fk_wrist_root_rot_obs_key == "max_coords_obs"
    assert config.fk_wrist_future_step == 0


def test_missing_reference_key_raises_a_named_error(robot_config):
    num_dofs = robot_config.kinematic_info.num_dofs
    actions = torch.zeros(2, num_dofs)
    agent = _make_agent(robot_config, {"fk_wrist_pos_weight": 1.0})
    with pytest.raises(KeyError, match="masked_mimic_target_poses"):
        agent._calculate_fk_wrist_loss({"privileged_action": actions}, actions)
