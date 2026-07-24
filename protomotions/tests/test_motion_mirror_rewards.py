# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GOLD reward-invariance test for online sagittal mirror augmentation.

Rewards are mirror-invariant scalars: if the SAME sagittal reflection is applied
to BOTH the reference and the (actual) robot state, every tracking reward must be
UNCHANGED. This catches hidden handedness anywhere in the reward pipeline --
heading-frame construction, angular-velocity signs, per-body indexing -- WITHOUT
needing to derive the full obs mirror map (the coordinator's weaker-but-sufficient
variant). We drive the REAL teacher reward compute_funcs directly (pure functions
of named tensors), so this exercises production reward math.

    PYTHONPATH=. python protomotions/tests/test_motion_mirror_rewards.py
"""

import torch

from protomotions.robot_configs.h1_2 import H1_2RobotConfig
from protomotions.components.motion_mirror import build_mirror_maps, mirror_robot_state
from protomotions.components.pose_lib import fk_batch_mjcf_with_velocities
from protomotions.envs.rewards.tracking import (
    compute_relative_body_pos_rew,
    compute_relative_body_ori_rew,
    compute_global_body_lin_vel_rew,
    compute_global_body_ang_vel_rew,
    compute_global_anchor_pos_rew,
    compute_global_anchor_ori_rew,
    compute_heading_local_anchor_drift_rew,
)

torch.manual_seed(3)
ATOL = 1e-5


def _ref_qpos(ki, B):
    qpos = torch.zeros(B, ki.nq)
    q = torch.randn(B, 4); q = q / q.norm(dim=-1, keepdim=True)
    qpos[:, 3:7] = q                                   # random root quat (wxyz)
    qpos[:, :3] = torch.randn(B, 3)                    # random root pos
    lo, hi = ki.dof_limits_lower, ki.dof_limits_upper
    qpos[:, 7:] = lo + torch.rand(B, ki.num_dofs) * (hi - lo)
    return qpos


def _fk_state(ki, qpos):
    st = fk_batch_mjcf_with_velocities(ki, qpos, fps=50, compute_velocities=False)
    B = qpos.shape[0]
    st.rigid_body_vel = torch.randn(B, ki.num_bodies, 3)
    st.rigid_body_ang_vel = torch.randn(B, ki.num_bodies, 3)
    return st


def _anchor(st, idx):
    return st.rigid_body_pos[:, idx, :], st.rigid_body_rot[:, idx, :]


def main():
    cfg = H1_2RobotConfig()
    ki = cfg.kinematic_info
    maps = build_mirror_maps(ki.body_names, ki.dof_names, ki.hinge_axes_map,
                             w_last=True)
    B = 64
    anchor_idx = 0  # pelvis (central, self-mirrored)
    # cur = ref + small tracking error, so rewards are non-trivial (spread > 0)
    # AND every reward's Gaussian kernel is in its sensitive range.
    ref_qpos = _ref_qpos(ki, B)
    cur_qpos = ref_qpos.clone()
    cur_qpos[:, :3] += 0.08 * torch.randn(B, 3)
    cur_qpos[:, 3:7] += 0.05 * torch.randn(B, 4)
    cur_qpos[:, 3:7] /= cur_qpos[:, 3:7].norm(dim=-1, keepdim=True)
    cur_qpos[:, 7:] += 0.15 * torch.randn(B, ki.num_dofs)
    ref = _fk_state(ki, ref_qpos)
    cur = _fk_state(ki, cur_qpos)
    # correlate velocities (ref vel + noise) so vel rewards also have spread
    cur.rigid_body_vel = ref.rigid_body_vel + 0.3 * torch.randn_like(ref.rigid_body_vel)
    cur.rigid_body_ang_vel = ref.rigid_body_ang_vel + 0.3 * torch.randn_like(ref.rigid_body_ang_vel)

    ref_m = ref.clone(); mirror_robot_state(ref_m, torch.ones(B, dtype=torch.bool), maps)
    cur_m = cur.clone(); mirror_robot_state(cur_m, torch.ones(B, dtype=torch.bool), maps)

    ca_pos, ca_rot = _anchor(cur, anchor_idx)
    ca_pos_m, ca_rot_m = _anchor(cur_m, anchor_idx)
    ra_pos, _ = _anchor(ref, anchor_idx)
    ra_pos_m, _ = _anchor(ref_m, anchor_idx)

    checks = []

    def check(name, r0, r1):
        e = (r0 - r1).abs().max().item()
        rng = (r0.max() - r0.min()).abs().item()
        checks.append((name, e, r0.mean().item(), rng))
        print(f"[rew] {name:32s} |R-R_mirror|max={e:.2e}  mean={r0.mean():.4f}  "
              f"spread={rng:.3f}")

    # 1. relative body position (heading-local frame) -- the core tracking reward
    check("relative_body_pos",
          compute_relative_body_pos_rew(cur.rigid_body_pos, ref.rigid_body_pos,
                                        ca_rot, ref.rigid_body_rot, ca_pos, anchor_idx),
          compute_relative_body_pos_rew(cur_m.rigid_body_pos, ref_m.rigid_body_pos,
                                        ca_rot_m, ref_m.rigid_body_rot, ca_pos_m,
                                        anchor_idx))

    # 1b. relative body position with a per-wrist upweight (the teacher upweights
    #     wrists) -- verifies the L/R body permutation keeps the weighting coherent
    lw = ki.body_names.index("left_wrist_yaw_link")
    rw = ki.body_names.index("right_wrist_yaw_link")
    bi = torch.tensor([lw, rw])
    bw = torch.tensor([3.0, 3.0])
    check("relative_body_pos[wrists x3]",
          compute_relative_body_pos_rew(cur.rigid_body_pos, ref.rigid_body_pos,
                                        ca_rot, ref.rigid_body_rot, ca_pos, anchor_idx,
                                        body_indices=bi, body_weights=bw),
          compute_relative_body_pos_rew(cur_m.rigid_body_pos, ref_m.rigid_body_pos,
                                        ca_rot_m, ref_m.rigid_body_rot, ca_pos_m,
                                        anchor_idx, body_indices=bi, body_weights=bw))

    # 2. relative body orientation (heading-local)
    check("relative_body_ori",
          compute_relative_body_ori_rew(cur.rigid_body_rot, ref.rigid_body_rot,
                                        ca_rot, anchor_idx),
          compute_relative_body_ori_rew(cur_m.rigid_body_rot, ref_m.rigid_body_rot,
                                        ca_rot_m, anchor_idx))

    # 3. global body linear velocity (world frame)
    check("global_body_lin_vel",
          compute_global_body_lin_vel_rew(cur.rigid_body_vel, ref.rigid_body_vel),
          compute_global_body_lin_vel_rew(cur_m.rigid_body_vel, ref_m.rigid_body_vel))

    # 4. global body angular velocity (world frame; PSEUDOVECTOR under mirror)
    check("global_body_ang_vel",
          compute_global_body_ang_vel_rew(cur.rigid_body_ang_vel, ref.rigid_body_ang_vel),
          compute_global_body_ang_vel_rew(cur_m.rigid_body_ang_vel, ref_m.rigid_body_ang_vel))

    # 5. global anchor position + orientation (world frame)
    check("global_anchor_pos",
          compute_global_anchor_pos_rew(ca_pos, ref.rigid_body_pos, anchor_idx),
          compute_global_anchor_pos_rew(ca_pos_m, ref_m.rigid_body_pos, anchor_idx))
    check("global_anchor_ori",
          compute_global_anchor_ori_rew(ca_rot, ref.rigid_body_rot, anchor_idx),
          compute_global_anchor_ori_rew(ca_rot_m, ref_m.rigid_body_rot, anchor_idx))

    # 6. heading-local anchor drift (the reward twin of the future-displacement cmd)
    check("heading_local_anchor_drift",
          compute_heading_local_anchor_drift_rew(ca_pos, ca_rot, ra_pos),
          compute_heading_local_anchor_drift_rew(ca_pos_m, ca_rot_m, ra_pos_m))

    worst = max(e for _, e, _, _ in checks)
    # guard: rewards must actually VARY across envs (else invariance is trivial)
    min_spread = min(s for _, _, _, s in checks)
    print(f"\n[rew] worst invariance error = {worst:.2e} (tol {ATOL}); "
          f"min reward spread across envs = {min_spread:.3f} (must be > 0)")
    assert worst < ATOL, f"REWARD INVARIANCE FAILED (worst {worst})"
    assert min_spread > 1e-3, "rewards are constant -> test is trivial"
    print("[rew] PASS: every teacher tracking reward is mirror-invariant "
          "(heading frame, ang-vel pseudovector, L/R body weighting all coherent)")


if __name__ == "__main__":
    main()
