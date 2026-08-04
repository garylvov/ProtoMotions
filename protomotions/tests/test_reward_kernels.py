# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pure reward kernels."""

import math

import torch

from protomotions.envs.rewards import base, regularization, task, tracking


def _identity_quat(*shape: int) -> torch.Tensor:
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


def test_base_reward_primitives_handle_shapes_indices_and_exp_modes():
    x = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[2.0, 2.0], [4.0, 4.0]],
        ]
    )
    ref = torch.zeros_like(x)
    indices = torch.tensor([1])

    assert torch.allclose(
        base.mean_squared_error(x, ref),
        torch.tensor([0.5, 10.0]),
    )
    assert torch.allclose(
        base.mean_squared_error(x, ref, indices=indices),
        torch.tensor([1.0, 16.0]),
    )
    assert torch.allclose(
        base.mean_squared_error(torch.tensor([[1.0, 3.0]]), torch.zeros(1, 2)),
        torch.tensor([5.0]),
    )
    assert torch.allclose(
        base.mean_squared_error(torch.tensor([2.0]), torch.zeros(1)),
        torch.tensor([4.0]),
    )

    assert torch.allclose(
        base.mean_squared_error_exp(x, ref, coefficient=-1.0),
        torch.exp(torch.tensor([-0.5, -10.0])),
    )
    assert torch.allclose(
        base.mean_squared_error_exp(
            x,
            ref,
            coefficient=-1.0,
            indices=indices,
            mean_before_exp=False,
        ),
        torch.exp(torch.tensor([[-1.0], [-16.0]])).mean(dim=-1),
    )
    assert torch.allclose(
        base.mean_squared_error_exp(
            torch.tensor([[1.0, 3.0]]),
            torch.zeros(1, 2),
            coefficient=-2.0,
        ),
        torch.exp(torch.tensor([-10.0])),
    )
    assert torch.allclose(
        base.mean_squared_error_exp(
            torch.tensor([2.0]),
            torch.zeros(1),
            coefficient=-0.5,
        ),
        torch.exp(torch.tensor([-2.0])),
    )

    assert torch.allclose(
        base.norm(torch.tensor([[[3.0, 4.0], [5.0, 12.0]]])),
        torch.tensor([[5.0, 13.0]]),
    )
    assert torch.allclose(
        base.norm(torch.tensor([[[3.0, 4.0], [5.0, 12.0]]]), indices=indices),
        torch.tensor([[13.0]]),
    )
    assert torch.allclose(
        base.delta_norm(torch.tensor([[3.0, 4.0]]), torch.zeros(1, 2)),
        torch.tensor([5.0]),
    )
    assert torch.allclose(
        base.delta_norm(x, ref, indices=indices),
        torch.tensor([[2.0**0.5], [32.0**0.5]]),
    )
    assert torch.allclose(
        base.delta_logmeanexp(
            torch.tensor([[1.0, 3.0]]),
            torch.zeros(1, 2),
            beta=2.0,
        ),
        (torch.logsumexp(torch.tensor([[2.0, 6.0]]), dim=-1) - math.log(2)) / 2.0,
    )
    assert torch.allclose(
        base.delta_logmeanexp(x, ref, indices=indices, beta=2.0),
        torch.tensor([[1.0], [4.0]]),
    )
    assert torch.allclose(
        base.absolute_difference_sum(x, ref),
        torch.tensor([2.0, 12.0]),
    )
    assert torch.allclose(
        base.absolute_difference_sum(x, ref, indices=indices),
        torch.tensor([2.0, 8.0]),
    )
    assert torch.allclose(
        base.absolute_difference_sum(torch.tensor([[1.0, -2.0]]), torch.zeros(1, 2)),
        torch.tensor([3.0]),
    )
    assert torch.allclose(
        base.absolute_difference_sum(torch.tensor([-3.0]), torch.zeros(1)),
        torch.tensor([3.0]),
    )


def test_base_rotation_and_power_primitives():
    quat = _identity_quat(2, 2)
    ref_quat = quat.clone()
    ref_quat[1, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])

    assert torch.allclose(
        base.rotation_error_exp(quat, quat, coefficient=-1.0),
        torch.ones(2),
    )
    assert torch.allclose(
        base.rotation_error_exp(
            quat,
            quat,
            coefficient=-1.0,
            indices=torch.tensor([1]),
            mean_before_exp=False,
        ),
        torch.ones(2),
    )
    assert base.rotation_error_exp(quat, ref_quat, coefficient=-1.0)[1] < 1.0
    assert torch.allclose(base.rotation_error(quat, quat), torch.zeros(2))
    assert torch.allclose(
        base.rotation_error(quat, quat, indices=torch.tensor([1])),
        torch.zeros(2),
    )

    dof_forces = torch.tensor([[2.0, -3.0], [4.0, 5.0]])
    dof_vel = torch.tensor([[10.0, -2.0], [0.5, -1.0]])
    assert torch.allclose(
        base.power_consumption_sum(dof_forces, dof_vel),
        torch.tensor([26.0, 7.0]),
    )
    assert torch.allclose(
        base.power_consumption_sum(
            dof_forces,
            dof_vel,
            indices=torch.tensor([1]),
        ),
        torch.tensor([6.0, 5.0]),
    )
    assert torch.allclose(
        base.power_consumption_sum(dof_forces, dof_vel, use_torque_squared=True),
        torch.tensor([13.0, 41.0]),
    )
    assert torch.allclose(
        base.power_consumption_exp(dof_forces, dof_vel, coefficient=-0.1),
        torch.exp(torch.tensor([-2.6, -0.7])),
    )
    assert torch.allclose(
        base.power_consumption_exp(
            dof_forces,
            dof_vel,
            coefficient=-0.1,
            use_torque_squared=True,
            indices=torch.tensor([1]),
        ),
        torch.exp(torch.tensor([-0.9, -2.5])),
    )
    assert torch.allclose(
        base.velocity_squared_sum(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])),
        torch.tensor([30.0]),
    )
    assert torch.allclose(
        base.velocity_squared_sum(torch.tensor([[1.0, 2.0, 3.0]])),
        torch.tensor([14.0]),
    )
    assert torch.allclose(
        base.velocity_squared_sum(
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            indices=torch.tensor([1]),
        ),
        torch.tensor([25.0]),
    )


def test_regularization_rewards_and_helpers():
    current_action = torch.tensor([[1.0, 3.0], [4.0, 4.0]])
    previous_action = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    dof_pos = torch.tensor([[-2.0, 0.0, 2.0], [0.0, 2.0, 4.0]])
    lower = torch.tensor([-1.0, -1.0, -1.0])
    upper = torch.tensor([1.0, 1.0, 3.0])

    assert torch.allclose(
        regularization.compute_action_smoothness(current_action, previous_action),
        torch.tensor([2.0, 5.0]),
    )
    assert torch.allclose(
        regularization.compute_action_smoothness_logmeanexp(
            current_action,
            previous_action,
            beta=2.0,
        ),
        base.delta_logmeanexp(current_action, previous_action, beta=2.0),
    )
    assert torch.allclose(
        regularization.compute_pow_rew(
            torch.tensor([[2.0, -3.0]]),
            torch.tensor([[10.0, -2.0]]),
        ),
        torch.tensor([26.0]),
    )
    assert torch.allclose(
        regularization.compute_pow_rew(
            torch.tensor([[2.0, -3.0]]),
            torch.tensor([[10.0, -2.0]]),
            use_torque_squared=True,
        ),
        torch.tensor([13.0]),
    )
    assert torch.allclose(
        regularization.compute_soft_pos_limit_rew(dof_pos, lower, upper),
        torch.tensor([1.0, 2.0]),
    )
    assert torch.allclose(
        regularization.joint_limit_violation(
            dof_pos,
            lower,
            upper,
            indices=torch.tensor([0, 2]),
        ),
        torch.tensor([1.0, 1.0]),
    )

    sim_contacts = torch.tensor([[True, False, True], [False, True, False]])
    ref_contacts = torch.tensor([[True, True, False], [True, True, False]])
    assert torch.allclose(
        regularization.compute_contact_match_rew(
            sim_contacts,
            ref_contacts,
            contact_body_ids=torch.tensor([1, 2]),
        ),
        torch.tensor([2.0, 0.0]),
    )
    assert torch.allclose(
        regularization.contact_mismatch_sum(
            sim_contacts,
            ref_contacts,
            indices=torch.tensor([0, 1]),
        ),
        torch.tensor([1.0, 1.0]),
    )

    sim_contacts_liftoff = torch.tensor(
        [
            [False, False, False, True],
            [False, False, False, False],
            [False, False, False, False],
        ]
    )
    ref_contacts_liftoff = torch.tensor(
        [
            [False, True, False, True],
            [False, False, False, 0.75],
            [False, True, False, True],
        ]
    )
    historical_body_contacts = torch.tensor(
        [
            [[True, True]],
            [[True, True]],
            [[False, True]],
        ]
    )
    assert torch.allclose(
        regularization.compute_reference_contact_liftoff_penalty(
            sim_contacts_liftoff,
            ref_contacts_liftoff,
            contact_body_ids=torch.tensor([1, 3]),
            historical_body_contacts=historical_body_contacts,
            ref_contact_threshold=0.5,
        ),
        torch.tensor([1.0, 0.5, 1.0]),
    )
    persistent_air = regularization.compute_reference_contact_liftoff_penalty(
        torch.tensor([[False, False]]),
        torch.tensor([[False, True]]),
        contact_body_ids=torch.tensor([1]),
        historical_body_contacts=torch.tensor([[[False]]]),
    )
    assert torch.allclose(persistent_air, torch.zeros(1))
    try:
        regularization.compute_reference_contact_liftoff_penalty(
            torch.tensor([[False, False]]),
            None,
            contact_body_ids=torch.tensor([1]),
            historical_body_contacts=torch.tensor([[[True]]]),
        )
        assert False, "missing reference contacts must raise"
    except ValueError:
        pass
    assert torch.allclose(
        regularization.compute_contact_force_change_rew(
            torch.tensor([[10.0, 50.0], [100.0, 0.0]]),
            torch.tensor([[0.0, 0.0], [20.0, 40.0]]),
            threshold=30.0,
        ),
        torch.tensor([20.0, 60.0]),
    )
    assert torch.allclose(
        regularization.impact_force_penalty(
            torch.tensor([[10.0, 50.0], [100.0, 0.0]]),
            torch.tensor([[0.0, 0.0], [20.0, 40.0]]),
            indices=torch.tensor([0]),
            threshold=30.0,
        ),
        torch.tensor([0.0, 50.0]),
    )


def test_task_rewards_cover_direction_path_target_and_object_terms():
    root_rot = _identity_quat(3)
    heading_reward = task.compute_heading_velocity_rew(
        root_pos=torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        prev_root_pos=torch.zeros(3, 3),
        root_rot=root_rot,
        tar_dir=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        tar_speed=torch.tensor([1.0, 1.0, 1.0]),
        tar_face_dir=torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        dt=1.0,
    )
    assert torch.allclose(heading_reward[0], torch.tensor(1.0))
    assert torch.allclose(heading_reward[1], torch.tensor(0.3))
    assert heading_reward[2] < 0.7

    assert torch.allclose(
        task.compute_path_following_rew(
            head_pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
            tar_pos=torch.tensor([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]]),
            height_conditioned=True,
            pos_err_scale=2.0,
            height_err_scale=1.0,
        ),
        torch.tensor([math.exp(-1.0), math.exp(-2.0)]),
    )
    assert torch.allclose(
        task.compute_path_following_rew(
            head_pos=torch.tensor([[0.0, 0.0, 0.0]]),
            tar_pos=torch.tensor([[0.0, 0.0, 10.0]]),
            height_conditioned=False,
        ),
        torch.ones(1),
    )

    target_reward = task.compute_target_rew(
        root_pos=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        tar_pos=torch.tensor([[0.1, 0.0, 0.0], [12.0, 0.0, 0.0]]),
        tar_proximity_threshold=0.5,
        pos_err_scale=0.5,
    )
    assert torch.allclose(target_reward, torch.tensor([1.0, math.exp(-1.0)]))


def test_tracking_rewards_cover_standard_and_beyond_mimic_variants():
    current_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )
    ref_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [2.0, 0.0, 0.0]],
        ]
    )
    current_rot = _identity_quat(2, 2)
    ref_rot = _identity_quat(2, 2)
    ref_rot[1, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    current_vel = torch.ones(2, 2, 3)
    ref_vel = torch.zeros(2, 2, 3)
    current_anchor_pos = current_pos[:, 0, :]
    current_anchor_rot = current_rot[:, 0, :]

    assert torch.allclose(tracking.compute_gt_rew(current_pos, current_pos), torch.ones(2))
    assert torch.allclose(tracking.compute_gr_rew(current_rot, current_rot), torch.ones(2))
    assert torch.allclose(tracking.compute_gv_rew(current_vel, current_vel), torch.ones(2))
    assert torch.allclose(tracking.compute_gav_rew(current_vel, current_vel), torch.ones(2))
    assert torch.allclose(
        tracking.compute_rh_rew(current_pos[:, 0, 2], current_pos),
        torch.ones(2),
    )
    assert torch.allclose(
        tracking.compute_global_position_error_exp(current_pos, current_pos, sigma=0.5),
        torch.ones(2),
    )
    assert tracking.compute_global_position_error_exp(
        current_pos,
        ref_pos,
        sigma=1.0,
        indices=torch.tensor([1]),
    )[1] < 1.0
    assert torch.allclose(
        tracking.compute_global_anchor_pos_rew(
            current_anchor_pos,
            ref_pos,
            anchor_idx=0,
            sigma=1.0,
        ),
        torch.tensor([1.0, math.exp(-1.0)]),
    )
    assert torch.allclose(
        tracking.compute_global_orientation_error_exp(current_rot, current_rot, sigma=1.0),
        torch.ones(2),
    )
    assert tracking.compute_global_anchor_ori_rew(
        current_anchor_rot,
        ref_rot,
        anchor_idx=0,
        sigma=1.0,
    )[1] < 1.0

    assert torch.allclose(
        tracking.compute_relative_body_pos_rew(
            current_pos,
            current_pos,
            current_anchor_rot,
            current_rot,
            current_anchor_pos,
            anchor_idx=0,
            sigma=1.0,
        ),
        torch.ones(2),
    )
    assert tracking.compute_relative_body_pos_rew(
        current_pos,
        ref_pos,
        current_anchor_rot,
        ref_rot,
        current_anchor_pos,
        anchor_idx=0,
        sigma=1.0,
        body_indices=torch.tensor([1]),
    )[1] < 1.0
    assert torch.allclose(
        tracking.compute_relative_body_ori_rew(
            current_rot,
            current_rot,
            current_anchor_rot,
            anchor_idx=0,
            sigma=1.0,
        ),
        torch.ones(2),
    )
    assert tracking.compute_relative_body_ori_rew(
        current_rot,
        ref_rot,
        current_anchor_rot,
        anchor_idx=0,
        sigma=1.0,
        body_indices=torch.tensor([0]),
    )[1] < 1.0
    assert torch.allclose(
        tracking.compute_global_body_lin_vel_rew(current_vel, current_vel),
        torch.ones(2),
    )
    assert torch.allclose(
        tracking.compute_global_body_ang_vel_rew(current_vel, current_vel),
        torch.ones(2),
    )
    assert torch.allclose(
        tracking.compute_gt_rel_rew(
            current_pos,
            current_pos,
            current_anchor_rot,
            current_rot,
            anchor_idx=0,
            body_indices=[0, 1],
        ),
        torch.ones(2),
    )
    assert tracking.compute_gt_rel_rew(
        current_pos,
        ref_pos,
        current_anchor_rot,
        ref_rot,
        anchor_idx=0,
    )[1] < 1.0
    assert torch.allclose(
        tracking.compute_anchor_xy_rew(
            current_anchor_pos,
            current_pos,
            anchor_idx=0,
        ),
        torch.ones(2),
    )


def test_dof_pos_track_reward_pins_joint_space_posture():
    """Joint-space DOF posture tracking reward (posture-tightening bundle).

    Covers: a perfect / stance-held track (reward == 1), a tight vs loose track
    ordering, the exp(-mean(dof_err^2)/sigma^2) math, sigma sensitivity, and DOF
    subsetting.
    """
    # 27-DOF reference posture (H1_2 has 27 dofs); a fixed stance/reference pose.
    ref = torch.linspace(-1.0, 1.0, 27).unsqueeze(0).repeat(3, 1)

    # Env 0: PERFECT / STANCE-HELD track (current == ref) -> reward == 1.
    # Env 1: TIGHT track (small per-DOF error).
    # Env 2: LOOSE track (large per-DOF error, the sim2sim loose optimum).
    current = ref.clone()
    current[1] = ref[1] + 0.30  # ~0.30 rad, the transferring-policy scale
    current[2] = ref[2] + 1.50  # ~1.50 rad, the loose PhysX-only optimum

    sigma = 0.35
    rew = tracking.compute_dof_pos_track_rew(current, ref, sigma=sigma)

    # Perfect track scores exactly 1.
    assert torch.isclose(rew[0], torch.tensor(1.0))
    # Reward strictly decreases as the posture loosens.
    assert rew[0] > rew[1] > rew[2]
    # The loose optimum is squeezed to ~0; the tight track stays well above it.
    assert rew[2] < 0.01
    assert rew[1] > 0.4

    # Exact kernel math: exp(-mean(dof_err^2) / sigma^2).
    expected = torch.exp(
        -((current - ref).pow(2).mean(dim=-1)) / (sigma ** 2)
    )
    assert torch.allclose(rew, expected)

    # A smaller sigma penalizes the SAME error harder (tighter posture demand).
    rew_tight_sigma = tracking.compute_dof_pos_track_rew(current, ref, sigma=0.20)
    assert rew_tight_sigma[1] < rew[1]

    # DOF subsetting: restricting to a single DOF matches the single-DOF kernel.
    idx = torch.tensor([5])
    rew_sub = tracking.compute_dof_pos_track_rew(
        current, ref, sigma=sigma, indices=idx
    )
    expected_sub = torch.exp(
        -((current[:, idx] - ref[:, idx]).pow(2).mean(dim=-1)) / (sigma ** 2)
    )
    assert torch.allclose(rew_sub, expected_sub)


def test_contact_match_threshold_and_match_reward_modes():
    """v5.4 contact_match: smoothed-ref binarization + match-reward mode."""
    feet = torch.tensor([0, 1])

    # --- Smoothed ref floats binarize at ref_contact_threshold (H1 lesson:
    # raw |sim - 0.43| would half-charge every smoothed swing edge).
    sim = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    ref_smoothed = torch.tensor(
        [[0.14, 0.43], [0.71, 0.43], [0.71, 0.94]]
    )
    mismatch = regularization.compute_contact_match_rew(
        sim, ref_smoothed, contact_body_ids=feet, ref_contact_threshold=0.5
    )
    # 0.14 -> 0, 0.43 -> 0, 0.71 -> 1, 0.94 -> 1 after binarization.
    assert torch.allclose(mismatch, torch.tensor([2.0, 1.0, 0.0]))

    # --- match_reward=True: max reward when policy contacts == (binarized)
    # ref contacts everywhere; a foot planted while the ref is airborne (the
    # early-landing gap-filler double-step) forfeits that foot's match unit.
    match = regularization.compute_contact_match_rew(
        sim, ref_smoothed, contact_body_ids=feet,
        ref_contact_threshold=0.5, match_reward=True,
    )
    assert torch.allclose(match, torch.tensor([0.0, 1.0, 2.0]))
    # Perfect match => max reward == num feet.
    perfect = regularization.compute_contact_match_rew(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[0.9, 0.1]]),
        contact_body_ids=feet, match_reward=True,
    )
    assert torch.allclose(perfect, torch.tensor([2.0]))
    # Early landing (policy foot down, ref airborne) scores strictly less.
    early_landing = regularization.compute_contact_match_rew(
        torch.tensor([[1.0, 1.0]]), torch.tensor([[0.9, 0.1]]),
        contact_body_ids=feet, match_reward=True,
    )
    assert early_landing.item() < perfect.item()
    assert torch.allclose(early_landing, torch.tensor([1.0]))

    # --- Legacy back-compat: hard bool refs at default threshold 0.5 are
    # byte-identical to the pre-v5.4 kernel.
    sim_b = torch.tensor([[True, False, True], [False, True, False]])
    ref_b = torch.tensor([[True, True, False], [True, True, False]])
    assert torch.allclose(
        regularization.compute_contact_match_rew(
            sim_b, ref_b, contact_body_ids=torch.tensor([1, 2])
        ),
        torch.tensor([2.0, 0.0]),
    )


def test_action_smoothness_lme_graced_matches_lme_and_respects_grace():
    """v5.4 arm-flail tax: LME kernel + perturbation grace zeroing."""
    prev = torch.zeros(3, 4)
    cur = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],   # no motion
            [0.1, 0.1, 0.1, 0.1],   # gentle uniform motion
            [0.0, 0.0, 0.0, 2.0],   # single-joint flail
        ]
    )
    ungraced = regularization.compute_action_smoothness_lme_graced(cur, prev)
    # No grace mask => identical to the stock LME kernel.
    assert torch.allclose(
        ungraced,
        regularization.compute_action_smoothness_logmeanexp(cur, prev),
    )
    # Single-joint flail (env 2) out-prices the same total |delta| spread
    # uniformly (env 1) -- the soft-max flavor the mean-flavored action_rate
    # term lacks.
    assert ungraced[2] > ungraced[1] > ungraced[0]

    # Grace mask zeroes exactly the graced envs (push/wrench recovery).
    grace = torch.tensor([False, True, True])
    graced = regularization.compute_action_smoothness_lme_graced(
        cur, prev, perturbation_grace_mask=grace
    )
    assert torch.allclose(graced[0], ungraced[0])
    assert graced[1] == 0.0 and graced[2] == 0.0


def test_reference_contact_liftoff_penalty_fires_only_on_unnecessary_liftoffs():
    """Liftoff penalty: ref-planted liftoff pays, ref-matched swing is free."""
    feet = torch.tensor([0, 1])
    # Previous step: both feet in contact for all envs.
    prev = torch.ones(3, 1, 2)
    # Current sim: env 0 keeps both planted; env 1 lifts foot 0 while the ref
    # keeps it planted (UNNECESSARY); env 2 lifts foot 0 with the ref in
    # swing (ref-matched liftoff).
    sim = torch.tensor([[1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    ref = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.1, 1.0]])
    pen = regularization.compute_reference_contact_liftoff_penalty(
        sim, ref, contact_body_ids=feet,
        historical_body_contacts=prev, ref_contact_threshold=0.5,
    )
    assert pen[0] == 0.0          # no liftoff at all
    assert pen[1] > 0.0           # unnecessary liftoff pays
    assert pen[2] == 0.0          # ref-matched liftoff is free


# =============================================================================
# DUAL-SIGMA companion + static-hold velocity penalty (2026-08-04)
# =============================================================================

_DUAL_SIGMA_POSITION_KERNELS = (
    "compute_global_position_error_exp",
    "compute_global_anchor_pos_rew",
    "compute_relative_body_pos_rew",
    "compute_dof_pos_track_rew",
    "compute_heading_local_anchor_drift_rew",
)


def _pos_kernel_call(name, fine_kwargs):
    """Invoke each dual-sigma-capable kernel on a fixed fixture."""
    torch.manual_seed(0)
    n_env, n_body = 4, 6
    cur_pos = torch.randn(n_env, n_body, 3)
    ref_pos = cur_pos + 0.07 * torch.randn(n_env, n_body, 3)
    cur_rot = _identity_quat(n_env)          # _identity_quat appends the 4
    ref_rot = _identity_quat(n_env, n_body)
    if name == "compute_global_position_error_exp":
        return tracking.compute_global_position_error_exp(
            cur_pos, ref_pos, 0.3, **fine_kwargs
        )
    if name == "compute_global_anchor_pos_rew":
        return tracking.compute_global_anchor_pos_rew(
            cur_pos[:, 0], ref_pos, anchor_idx=0, sigma=0.3, **fine_kwargs
        )
    if name == "compute_relative_body_pos_rew":
        return tracking.compute_relative_body_pos_rew(
            cur_pos, ref_pos, cur_rot, ref_rot, cur_pos[:, 0],
            anchor_idx=0, sigma=0.3, **fine_kwargs,
        )
    if name == "compute_dof_pos_track_rew":
        return tracking.compute_dof_pos_track_rew(
            cur_pos.reshape(n_env, -1),
            ref_pos.reshape(n_env, -1),
            sigma=0.35,
            **fine_kwargs,
        )
    if name == "compute_heading_local_anchor_drift_rew":
        return tracking.compute_heading_local_anchor_drift_rew(
            cur_pos[:, 0], cur_rot, ref_pos[:, 0], sigma=0.3, **fine_kwargs
        )
    raise AssertionError(name)


def test_dual_sigma_off_path_is_bit_exact_for_every_position_kernel():
    """RULE 10: unset fine companion == the ORIGINAL single-sigma expression.

    Exact bitwise equality, not allclose: the OFF path must return literally
    ``torch.exp(-error / sigma**2)``, never an algebraically-equal vectorized
    rewrite (a sibling lane hit float32 ulp drift doing exactly that).
    """
    for name in _DUAL_SIGMA_POSITION_KERNELS:
        default = _pos_kernel_call(name, {})
        explicit_zero = _pos_kernel_call(name, {"fine_weight": 0.0})
        zero_with_sigma = _pos_kernel_call(
            name, {"fine_weight": 0.0, "fine_sigma": 0.05}
        )
        assert torch.equal(default, explicit_zero), name
        assert torch.equal(default, zero_with_sigma), name

    # The shared helper itself, on a hand-built error vector.
    err = torch.tensor([0.0, 1e-4, 4e-4, 2.5e-3, 1e-2, 9e-2])
    assert torch.equal(
        tracking._dual_sigma_exp(err, 0.3), torch.exp(-err / (0.3 ** 2))
    )


def test_dual_sigma_math_is_coarse_plus_weighted_fine():
    """r = exp(-e^2/sc^2) + w * exp(-e^2/sf^2), exactly."""
    err = torch.tensor([0.0, 1e-4, 4e-4, 2.5e-3, 1e-2, 9e-2])  # squared error
    sc, sf, w = 0.3, 0.05, 0.5
    got = tracking._dual_sigma_exp(err, sc, w, sf)
    want = torch.exp(-err / sc ** 2) + w * torch.exp(-err / sf ** 2)
    assert torch.equal(got, want)

    # Coarse channel is UNTOUCHED: where the narrow companion has decayed to
    # nothing (large error), the dual reward IS today's reward.
    far = torch.tensor([0.09, 0.25, 1.0])  # squared error: 30 cm, 50 cm, 1 m
    assert torch.allclose(
        tracking._dual_sigma_exp(far, sc, w, sf),
        torch.exp(-far / sc ** 2),
        rtol=0, atol=1e-12,
    )

    # Maximum at zero error is exactly 1 + fine_weight.
    assert math.isclose(float(got[0]), 1.0 + w, rel_tol=0, abs_tol=1e-6)


def test_dual_sigma_preserves_capture_range_and_restores_near_zero_gradient():
    """The whole point: precision near 0, capture range untouched far away.

    Compares three configurations at a spread of position errors:
      A) today   sigma=0.3, no companion
      B) naive   sigma=0.05 alone (what "just tighten it" would do)
      C) dual    sigma=0.3 + 0.5 * exp(-e^2/0.05^2)
    """
    sc, sf, w = 0.3, 0.05, 0.5
    e = torch.tensor([0.01, 0.02, 0.05, 0.10, 0.30], dtype=torch.float64)
    e.requires_grad_(True)

    def grad_of(fn):
        r = fn(e.pow(2))
        g, = torch.autograd.grad(r.sum(), e, retain_graph=False)
        return g.abs()

    coarse = lambda q: torch.exp(-q / sc ** 2)          # noqa: E731
    naive = lambda q: torch.exp(-q / sf ** 2)           # noqa: E731
    dual = lambda q: tracking._dual_sigma_exp(q, sc, w, sf)  # noqa: E731

    g_coarse, g_naive, g_dual = grad_of(coarse), grad_of(naive), grad_of(dual)

    # TODAY'S DEAD ZONE: at 1-2 cm the coarse gradient is tiny in absolute
    # terms and far below what the narrow kernel provides.
    assert g_coarse[0] < 0.25          # 1 cm: ~0.22 /m
    assert g_dual[0] > 10 * g_coarse[0]
    assert g_dual[1] > 10 * g_coarse[1]  # 2 cm

    # CAPTURE RANGE PRESERVED: at 30 cm the dual kernel's gradient is
    # indistinguishable from today's, while the naive tightening has collapsed.
    assert torch.allclose(g_dual[4], g_coarse[4], rtol=1e-6)
    assert g_naive[4] < 1e-12
    assert g_dual[4] > 1e6 * g_naive[4]

    # And the dual REWARD at 30 cm still equals today's (companion ~0 there).
    with torch.no_grad():
        assert torch.allclose(
            dual(e[4:].pow(2)), coarse(e[4:].pow(2)), rtol=1e-9
        )


def test_dual_sigma_rejects_invalid_parameters():
    err = torch.tensor([0.0, 1e-3])
    for bad in ({"fine_weight": 0.5}, {"fine_weight": 0.5, "fine_sigma": 0.0},
                {"fine_weight": 0.5, "fine_sigma": -0.1},
                {"fine_weight": -0.5, "fine_sigma": 0.05}):
        try:
            tracking._dual_sigma_exp(err, 0.3, **bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_static_hold_vel_penalty_gates_on_reference_stillness():
    """Velocity term fires ONLY where the reference body is (near) static."""
    # 3 envs, 2 bodies (think: the two wrists).
    vel = torch.tensor([
        [[0.10, 0.0, 0.0], [0.00, 0.0, 0.0]],   # env0: body0 drifting, body1 still
        [[0.10, 0.0, 0.0], [0.10, 0.0, 0.0]],   # env1: both drifting
        [[3.00, 0.0, 0.0], [3.00, 0.0, 0.0]],   # env2: both moving fast
    ])
    ref_vel = torch.tensor([
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],     # env0: reference STATIC
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],     # env1: reference STATIC
        [[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]],     # env2: reference MOVING
    ])
    out = regularization.compute_static_hold_body_vel_penalty(
        vel, ref_vel, ref_speed_gate=0.05
    )
    # env0: mean(0.10, 0.0) = 0.05 ; env1: mean(0.10, 0.10) = 0.10
    # env2: reference is moving -> fully EXEMPT -> 0.0
    assert torch.allclose(out, torch.tensor([0.05, 0.10, 0.0]))

    # A perfectly still hand against a still reference pays exactly zero.
    still = torch.zeros(3, 2, 3)
    assert torch.equal(
        regularization.compute_static_hold_body_vel_penalty(
            still, ref_vel, ref_speed_gate=0.05
        ),
        torch.zeros(3),
    )

    # Gate 0.0 disables the term entirely, even with a dead-still reference.
    assert torch.equal(
        regularization.compute_static_hold_body_vel_penalty(
            vel, ref_vel, ref_speed_gate=0.0
        ),
        torch.zeros(3),
    )

    # LINEAR in speed (not squared): the whole point is gradient at slow drift.
    slow = regularization.compute_static_hold_body_vel_penalty(
        torch.full((1, 1, 3), 0.0), torch.zeros(1, 1, 3), ref_speed_gate=0.05
    )
    assert float(slow) == 0.0
    for speed in (0.01, 0.02, 0.04):
        v = torch.zeros(1, 1, 3)
        v[0, 0, 0] = speed
        got = regularization.compute_static_hold_body_vel_penalty(
            v, torch.zeros(1, 1, 3), ref_speed_gate=0.05
        )
        assert math.isclose(float(got), speed, rel_tol=1e-6)


def test_static_hold_vel_penalty_body_subset_and_still_mask():
    vel = torch.tensor([[[1.0, 0, 0], [0.0, 0, 0], [0.5, 0, 0]]])
    ref_vel = torch.zeros(1, 3, 3)

    # All bodies: mean(1.0, 0.0, 0.5) = 0.5
    assert torch.allclose(
        regularization.compute_static_hold_body_vel_penalty(vel, ref_vel),
        torch.tensor([0.5]),
    )
    # Subset to bodies [0, 2]: mean(1.0, 0.5) = 0.75
    assert torch.allclose(
        regularization.compute_static_hold_body_vel_penalty(
            vel, ref_vel, body_indices=[0, 2]
        ),
        torch.tensor([0.75]),
    )
    # Env-level still mask False -> everything suppressed.
    assert torch.equal(
        regularization.compute_static_hold_body_vel_penalty(
            vel, ref_vel, reference_still_mask=torch.tensor([False])
        ),
        torch.zeros(1),
    )
    assert torch.allclose(
        regularization.compute_static_hold_body_vel_penalty(
            vel, ref_vel, reference_still_mask=torch.tensor([True])
        ),
        torch.tensor([0.5]),
    )


def test_anchor_relative_terms_are_BLIND_to_base_translation_world_term_is_not():
    """THE FRAME PROOF (2026-08-04 audit) -- headline guard for the hand fix.

    Construct the exact failure mode: the reference is a static hold, the arm
    holds its pose PERFECTLY relative to the pelvis, and the pelvis translates
    4 cm sideways carrying the hand with it.

    The anchor-relative term (which is what ``wrist_relative_body_pos`` and
    ``relative_body_pos`` are) must score this as PERFECT -- proving nothing in
    the existing stack objects to a hand riding a swaying base. The new
    world-frame term must score it as an error.
    """
    n_body = 4
    wrist = 3
    ref_pos = torch.zeros(1, n_body, 3)
    ref_pos[0, 0] = torch.tensor([0.0, 0.0, 1.0])    # pelvis (anchor, body 0)
    ref_pos[0, wrist] = torch.tensor([0.4, 0.1, 1.2])  # hand at 0.54 m reach
    ref_rot = _identity_quat(1, n_body)

    # The WHOLE BODY translates 4 cm in +y -- the measured static-hold drift.
    # Arm joint angles unchanged => hand-in-pelvis vector is bit-identical.
    sway = torch.tensor([0.0, 0.04, 0.0])
    cur_pos = ref_pos + sway
    cur_anchor_rot = _identity_quat(1)

    anchor_rel = tracking.compute_relative_body_pos_rew(
        cur_pos, ref_pos, cur_anchor_rot, ref_rot, cur_pos[:, 0],
        anchor_idx=0, sigma=0.3, body_indices=[wrist],
    )
    # PERFECT score: the pelvis offset was subtracted from both sides.
    assert torch.allclose(anchor_rel, torch.ones(1), atol=1e-6), (
        "anchor-relative term should be blind to pure base translation"
    )
    # Even a NARROW companion cannot see it -- the error it is given is zero.
    anchor_rel_fine = tracking.compute_relative_body_pos_rew(
        cur_pos, ref_pos, cur_anchor_rot, ref_rot, cur_pos[:, 0],
        anchor_idx=0, sigma=0.3, body_indices=[wrist],
        fine_weight=0.5, fine_sigma=0.05,
    )
    assert torch.allclose(anchor_rel_fine, torch.full((1,), 1.5), atol=1e-6), (
        "a fine companion on an anchor-relative term still sees zero error -- "
        "tightening sigma there cannot fix world-frame hand drift"
    )

    # The WORLD-frame term DOES see the 4 cm.
    world = tracking.compute_global_body_pos_rew(
        cur_pos, ref_pos, sigma=0.3, body_indices=[wrist]
    )
    assert float(world) < 1.0
    assert math.isclose(float(world), math.exp(-(0.04 ** 2) / 0.3 ** 2), rel_tol=1e-5)

    # ...and the narrow companion is what makes 4 cm actually COST something.
    world_fine = tracking.compute_global_body_pos_rew(
        cur_pos, ref_pos, sigma=0.3, body_indices=[wrist],
        fine_weight=0.5, fine_sigma=0.05,
    )
    coarse_loss = 1.0 - float(world)                       # ~0.0177
    fine_loss = 1.5 - float(world_fine)                    # ~0.254
    assert fine_loss > 10 * coarse_loss

    # A hand that is genuinely in the right world place scores 1 (+ companion).
    perfect = tracking.compute_global_body_pos_rew(
        ref_pos, ref_pos, sigma=0.3, body_indices=[wrist],
        fine_weight=0.5, fine_sigma=0.05,
    )
    assert torch.allclose(perfect, torch.full((1,), 1.5), atol=1e-6)


def test_world_body_pos_rew_off_path_and_still_mask():
    torch.manual_seed(3)
    cur = torch.randn(3, 5, 3)
    ref = cur + 0.05 * torch.randn(3, 5, 3)

    # OFF path is bit-exact with the plain global position kernel.
    assert torch.equal(
        tracking.compute_global_body_pos_rew(cur, ref, 0.3),
        tracking.compute_global_position_error_exp(cur, ref, 0.3),
    )
    # Still-mask suppresses non-hold envs and leaves hold envs untouched.
    mask = torch.tensor([True, False, True])
    gated = tracking.compute_global_body_pos_rew(
        cur, ref, 0.3, reference_still_mask=mask
    )
    ungated = tracking.compute_global_body_pos_rew(cur, ref, 0.3)
    assert torch.equal(gated[0], ungated[0])
    assert float(gated[1]) == 0.0
    assert torch.equal(gated[2], ungated[2])
    # No mask == mask of all-True.
    assert torch.equal(
        tracking.compute_global_body_pos_rew(
            cur, ref, 0.3, reference_still_mask=torch.ones(3, dtype=torch.bool)
        ),
        ungated,
    )


def test_world_hand_sigma_choice_has_gradient_at_the_measured_droop():
    """Recommended fine sigma must BITE at 40-70 mm, not just at 0.

    Measured static hold: ~40 mm drift on top of a ~66-72 mm standing droop.
    A narrow Gaussian centered on the reference is useless out there if it is
    too narrow -- this guards the 0.05 recommendation against a 0.03 regression.
    """
    def grad(e, s):
        return 2 * e / s ** 2 * math.exp(-((e / s) ** 2))

    coarse = grad(0.07, 0.30)
    fine_05 = 0.5 * grad(0.07, 0.05)
    fine_03 = 0.5 * grad(0.07, 0.03)
    # sigma 0.05 more than doubles the restoring gradient at the droop...
    assert fine_05 > 2.0 * coarse
    # ...while sigma 0.03 adds under a quarter of it: too narrow to matter.
    assert fine_03 < 0.25 * coarse
    # Both still leave the 30 cm capture gradient untouched.
    for s in (0.05, 0.03):
        assert 0.5 * grad(0.30, s) < 1e-6 * grad(0.30, 0.30)
