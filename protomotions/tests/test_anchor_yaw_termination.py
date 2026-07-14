# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the night13/T3 anchor-yaw-drift termination (NEW).

Covers:
- anchor_yaw_error_value: zero drift, known yaw offsets, wrap-around at
  +/-180deg (e.g. 350deg vs 10deg current should read as ~20deg drift, not
  340deg).
- compute_anchor_yaw_error_term: threshold gating.
- anchor_yaw_error_term_factory: MdpComponent wiring + settle_steps plumbing
  (mirrors tracking_error_term_factory's settle_steps contract).
- Distinctness from compute_anchor_ori_error_term (tilt): a pure yaw
  rotation should read ~0 tilt-error but nonzero yaw-error, and vice versa
  for a pure roll/pitch flip.
"""

import math

import torch

from protomotions.envs.component_factories import anchor_yaw_error_term_factory
from protomotions.envs.terminations import tracking


def _yaw_quat(angle_rad: float) -> torch.Tensor:
    """w-last quaternion for a pure yaw (Z-axis) rotation."""
    return torch.tensor(
        [0.0, 0.0, math.sin(angle_rad / 2.0), math.cos(angle_rad / 2.0)]
    )


def _batch_rot(quats):
    """[N, 4] current-anchor-rot tensor from a list of quaternions."""
    return torch.stack(quats, dim=0)


def _ref_rot_single_body(quats):
    """[N, 1, 4] ref_rigid_body_rot tensor (anchor_idx=0) from a list of quaternions."""
    return _batch_rot(quats).unsqueeze(1)


def test_anchor_yaw_error_value_zero_drift():
    cur = _batch_rot([_yaw_quat(0.3)])
    ref = _ref_rot_single_body([_yaw_quat(0.3)])
    err = tracking.anchor_yaw_error_value(cur, ref, anchor_idx=0)
    assert torch.allclose(err, torch.zeros(1), atol=1e-6)


def test_anchor_yaw_error_value_known_offset():
    cur = _batch_rot([_yaw_quat(0.0), _yaw_quat(math.radians(45))])
    ref = _ref_rot_single_body([_yaw_quat(math.radians(30)), _yaw_quat(0.0)])
    err = tracking.anchor_yaw_error_value(cur, ref, anchor_idx=0)
    assert torch.allclose(err, torch.tensor([math.radians(30), math.radians(45)]), atol=1e-5)


def test_anchor_yaw_error_value_wraps_around_pi():
    # current at +170deg, ref at -170deg -> true drift is 20deg, not 340deg.
    cur = _batch_rot([_yaw_quat(math.radians(170))])
    ref = _ref_rot_single_body([_yaw_quat(math.radians(-170))])
    err = tracking.anchor_yaw_error_value(cur, ref, anchor_idx=0)
    assert torch.allclose(err, torch.tensor([math.radians(20)]), atol=1e-4)


def test_compute_anchor_yaw_error_term_threshold_gating():
    cur = _batch_rot([_yaw_quat(0.0), _yaw_quat(math.radians(70))])
    ref = _ref_rot_single_body([_yaw_quat(0.0), _yaw_quat(0.0)])
    terminate = tracking.compute_anchor_yaw_error_term(
        cur, ref, anchor_idx=0, threshold=math.radians(60)
    )
    assert torch.equal(terminate, torch.tensor([False, True]))


def test_anchor_yaw_distinct_from_tilt_metric():
    # Pure yaw rotation: yaw error nonzero, tilt (projected-gravity) error ~0.
    cur = _batch_rot([_yaw_quat(math.radians(90))])
    ref = _ref_rot_single_body([_yaw_quat(0.0)])
    yaw_err = tracking.anchor_yaw_error_value(cur, ref, anchor_idx=0)
    tilt_err = tracking.anchor_ori_error_value(cur, ref, anchor_idx=0)
    assert yaw_err.item() > math.radians(45)
    assert tilt_err.item() < 1e-5

    # Pure X-axis flip (roll 180deg): tilt error large, yaw error ~0 (heading
    # direction of the flipped +X ref_dir axis is unchanged by a roll about X).
    roll_180 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    cur2 = roll_180.unsqueeze(0)
    ref2 = torch.tensor([0.0, 0.0, 0.0, 1.0]).unsqueeze(0).unsqueeze(1)
    tilt_err2 = tracking.anchor_ori_error_value(cur2, ref2, anchor_idx=0)
    assert tilt_err2.item() > 1.0


def test_anchor_yaw_error_term_factory_wires_mdp_component_and_settle_steps():
    component_no_settle = anchor_yaw_error_term_factory(threshold=1.0472)
    params = component_no_settle.get_params()
    assert params["threshold"] == 1.0472
    assert "settle_steps" not in params

    component_settle = anchor_yaw_error_term_factory(threshold=1.0472, settle_steps=15)
    params_settle = component_settle.get_params()
    assert params_settle["settle_steps"] == 15
    assert params_settle["threshold"] == 1.0472
