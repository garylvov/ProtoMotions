# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard tests for ``env/wrist_relative_body_pos/err_m_mean``.

Why this stat exists: ``wrist_relative_body_pos`` (w 1.3, sigma 0.3) is the term
that governs hand placement, and it had NO error stat. The only wrist number
logged was ``env/global_wrist_pos/err_m_mean``, which is WORLD frame and is
dominated by root drift (bounded only by the 0.4 m ``anchor_pos_drift``
termination threshold), so it cannot gate a run whose objective is
anchor-relative wrist tracking.

What is pinned here:

1. **Rule 10 absence.** Component not registered => not one extras key.
2. **Type / prefix / shape.** Tensors (the agent's extras aggregator silently
   drops python floats), no ``raw/`` prefix (the aggregator skips it), one value
   per env.
3. **Correctness against a hand-computed error**, including the case that
   separates this stat from the world-frame one: a pure root translation +
   heading rotation must read ZERO here while the world-frame stat reads large.
4. **The frame refactor is bit-exact.** ``compute_relative_body_pos_rew`` now
   delegates its frame math to ``compute_anchor_relative_local_body_pos``; this
   asserts the reward is BITWISE equal to an inline copy of the pre-refactor
   expression. ``torch.equal``, not ``allclose`` -- Rule 10 means byte identity.
"""

from types import SimpleNamespace

import torch

from protomotions.envs.rewards.tracking import (
    compute_anchor_relative_local_body_pos,
    compute_relative_body_pos_rew,
)
from protomotions.utils.rotations import calc_heading_quat_inv, quat_rotate


def _yaw_quat(theta: torch.Tensor) -> torch.Tensor:
    """w-last quaternion for a rotation of ``theta`` about +z."""
    half = theta * 0.5
    zeros = torch.zeros_like(theta)
    return torch.stack([zeros, zeros, torch.sin(half), torch.cos(half)], dim=-1)


def _ctx(cur_pos, ref_pos, cur_rot=None, ref_rot=None, anchor_idx=0):
    n, b = cur_pos.shape[0], cur_pos.shape[1]
    if cur_rot is None:
        cur_rot = torch.zeros(n, b, 4)
        cur_rot[..., 3] = 1.0
    if ref_rot is None:
        ref_rot = torch.zeros(n, b, 4)
        ref_rot[..., 3] = 1.0
    return SimpleNamespace(
        current=SimpleNamespace(
            rigid_body_pos=cur_pos,
            anchor_pos=cur_pos[:, anchor_idx, :],
            anchor_rot=cur_rot[:, anchor_idx, :],
        ),
        mimic=SimpleNamespace(
            anchor_idx=anchor_idx,
            ref_state=SimpleNamespace(
                rigid_body_pos=ref_pos, rigid_body_rot=ref_rot
            ),
        ),
    )


def _env(component):
    return SimpleNamespace(
        config=SimpleNamespace(
            reward_components=(
                {} if component is None else {"wrist_relative_body_pos": component}
            )
        ),
        extras={},
    )


def _wrist_component(body_indices):
    from protomotions.envs import component_factories as factories

    return factories.relative_body_pos_rew_factory(
        weight=1.3, sigma=0.3, body_indices=body_indices
    )


# =============================================================================
# 1 + 2 + 3. THE WRITER
# =============================================================================


def test_stat_is_silent_when_the_component_is_absent():
    """Rule 10: an unset config emits no new keys and pays no cost."""
    from protomotions.envs.base_env.env import BaseEnv

    cur = torch.zeros(2, 4, 3)
    env = _env(None)
    BaseEnv._log_wrist_relative_body_pos_extras(env, _ctx(cur, cur.clone()))
    assert env.extras == {}

    # ... and likewise with no mimic reference (a non-mimic env).
    env2 = _env(_wrist_component([2, 3]))
    ctx = _ctx(cur, cur.clone())
    BaseEnv._log_wrist_relative_body_pos_extras(
        env2, SimpleNamespace(current=ctx.current, mimic=None)
    )
    assert env2.extras == {}

    env3 = _env(_wrist_component([2, 3]))
    BaseEnv._log_wrist_relative_body_pos_extras(
        env3,
        SimpleNamespace(
            current=ctx.current, mimic=SimpleNamespace(ref_state=None, anchor_idx=0)
        ),
    )
    assert env3.extras == {}


def test_stat_type_prefix_shape_and_hand_computed_value():
    """Registered => metre-valued per-env error over the component's body set."""
    from protomotions.envs.base_env.env import BaseEnv

    # body 0 is the anchor (pelvis); bodies 2 and 3 are the two wrists.
    ref = torch.zeros(2, 4, 3)
    cur = torch.zeros(2, 4, 3)
    # env0: left wrist 3 cm off in x, right wrist 7 cm off in y. env1: exact.
    cur[0, 2, 0] = 0.03
    cur[0, 3, 1] = 0.07

    env = _env(_wrist_component([2, 3]))
    BaseEnv._log_wrist_relative_body_pos_extras(env, _ctx(cur, ref))

    assert set(env.extras) == {
        "wrist_relative_body_pos/err_m",
        "wrist_relative_body_pos/err_max_m",
    }
    for key, value in env.extras.items():
        assert isinstance(value, torch.Tensor), f"{key} must be a Tensor"
        assert not key.startswith("raw/"), f"{key} would be skipped by the agent"
        assert value.shape == (2,)

    # Hand-computed: env0 mean of (0.03, 0.07) = 0.05, max = 0.07.
    assert torch.allclose(
        env.extras["wrist_relative_body_pos/err_m"],
        torch.tensor([0.05, 0.0]),
        atol=1e-6,
    )
    assert torch.allclose(
        env.extras["wrist_relative_body_pos/err_max_m"],
        torch.tensor([0.07, 0.0]),
        atol=1e-6,
    )


def test_stat_is_anchor_relative_and_heading_local_not_world():
    """The whole point: root drift must contribute ZERO.

    Translate and yaw the robot bodily away from the reference with the arm
    perfectly placed relative to its own pelvis. The world-frame stat would read
    the full displacement; this one must read ~0. If this assertion ever flips,
    the v59 gate is measuring root drift again and is worthless.
    """
    from protomotions.envs.base_env.env import BaseEnv

    torch.manual_seed(0)
    ref = torch.randn(3, 5, 3)
    ref_rot = torch.zeros(3, 5, 4)
    ref_rot[..., 3] = 1.0

    theta = torch.tensor([0.3, -1.1, 2.0])
    q = _yaw_quat(theta)  # [3, 4]
    offset = torch.tensor([[1.5, -2.0, 0.0], [0.0, 4.0, 0.0], [-3.0, 3.0, 0.0]])

    # cur = R_yaw * (ref - ref_anchor) + ref_anchor + offset  -- i.e. the robot
    # is displaced and re-headed, but its pose IN ITS OWN FRAME is exact.
    rel = ref - ref[:, 0:1, :]
    q_exp = q.unsqueeze(1).expand(-1, ref.shape[1], -1).reshape(-1, 4)
    rot_rel = quat_rotate(q_exp, rel.reshape(-1, 3), w_last=True).reshape(ref.shape)
    cur = rot_rel + ref[:, 0:1, :] + offset.unsqueeze(1)
    cur_rot = q.unsqueeze(1).expand(-1, ref.shape[1], -1).contiguous()

    env = _env(_wrist_component([3, 4]))
    BaseEnv._log_wrist_relative_body_pos_extras(env, _ctx(cur, ref, cur_rot, ref_rot))

    err = env.extras["wrist_relative_body_pos/err_m"]
    assert torch.allclose(err, torch.zeros(3), atol=1e-5), err
    # Sanity: the WORLD-frame error over the same bodies is large, so the test
    # is really discriminating the frame and not just a zero input.
    world = (cur - ref).pow(2).sum(-1).sqrt()[:, [3, 4]].mean(-1)
    assert (world > 0.5).all(), world


def test_stat_matches_the_reward_kernels_own_frame_math():
    """The stat and the reward must agree body-for-body, bitwise."""
    from protomotions.envs.base_env.env import BaseEnv

    torch.manual_seed(7)
    cur = torch.randn(4, 6, 3)
    ref = torch.randn(4, 6, 3)
    cur_rot = torch.nn.functional.normalize(torch.randn(4, 6, 4), dim=-1)
    ref_rot = torch.nn.functional.normalize(torch.randn(4, 6, 4), dim=-1)
    idx = [4, 5]

    env = _env(_wrist_component(idx))
    BaseEnv._log_wrist_relative_body_pos_extras(
        env, _ctx(cur, ref, cur_rot, ref_rot, anchor_idx=0)
    )

    cur_local, ref_local = compute_anchor_relative_local_body_pos(
        cur, ref, cur_rot[:, 0, :], ref_rot, cur[:, 0, :], 0
    )
    expected = (cur_local[:, idx] - ref_local[:, idx]).pow(2).sum(-1).sqrt()
    assert torch.equal(
        env.extras["wrist_relative_body_pos/err_m"], expected.mean(-1)
    )
    assert torch.equal(
        env.extras["wrist_relative_body_pos/err_max_m"], expected.max(-1).values
    )


def test_the_tb_tag_the_v59_gate_reads_is_exactly_what_the_writer_produces():
    """READER/WRITER: pin the literal tag string, and the three aggregator
    rules that turn the extras key into it.

    The v59 go/no-go table reads ``env/wrist_relative_body_pos/err_m_mean``.
    That name is not chosen here -- it is produced by
    ``agent.py``'s extras aggregator: skip ``raw/``, Tensors only, and for a
    multi-element tensor emit ``<key>_mean`` / ``<key>_std``, then prefix
    ``env/``. If any of those three rules changes, the gate silently stops
    existing; this test makes that a red build instead.
    """
    import pathlib

    agent_src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "agents"
        / "base_agent"
        / "agent.py"
    ).read_text()
    assert 'if key.startswith("raw/"):' in agent_src
    assert 'extras_mean_std_dict[f"{key}_mean"] = extra_val.mean()' in agent_src
    assert 'env_log_dict = {f"env/{k}": v for k, v in env_log_dict.items()}' in agent_src

    writer_key = "wrist_relative_body_pos/err_m"
    assert f"env/{writer_key}_mean" == "env/wrist_relative_body_pos/err_m_mean"
    assert (
        f"env/{writer_key.replace('err_m', 'err_max_m')}_mean"
        == "env/wrist_relative_body_pos/err_max_m_mean"
    )

    # And the writer really emits that key, with numel > 1 so the _mean/_std
    # branch (not the scalar branch) is the one taken.
    from protomotions.envs.base_env.env import BaseEnv

    cur = torch.zeros(2, 4, 3)
    cur[0, 2, 0] = 0.03
    env = _env(_wrist_component([2, 3]))
    BaseEnv._log_wrist_relative_body_pos_extras(env, _ctx(cur, torch.zeros(2, 4, 3)))
    assert writer_key in env.extras
    assert env.extras[writer_key].numel() == 2 > 1


# =============================================================================
# 4. THE FRAME REFACTOR IS BIT-EXACT
# =============================================================================


def _pre_refactor_relative_body_pos_rew(
    current_rigid_body_pos,
    ref_rigid_body_pos,
    current_anchor_rot,
    ref_rigid_body_rot,
    current_anchor_pos,
    anchor_idx,
    sigma=0.3,
    body_indices=None,
    body_weights=None,
    fine_weight=0.0,
    fine_sigma=None,
):
    """VERBATIM copy of ``compute_relative_body_pos_rew`` as it stood at
    a954901, before the frame block was extracted. Do not tidy this."""
    from protomotions.envs.rewards.tracking import (
        compute_global_position_error_exp,
    )

    # Extract reference anchor pos and rot
    ref_anchor_pos = ref_rigid_body_pos[:, anchor_idx, :]
    ref_anchor_rot = ref_rigid_body_rot[:, anchor_idx, :]

    # Compute heading rotations (yaw-only)
    current_heading_rot_inv = calc_heading_quat_inv(current_anchor_rot, w_last=True)
    ref_heading_rot_inv = calc_heading_quat_inv(ref_anchor_rot, w_last=True)

    # Compute relative positions in world frame
    current_rel_pos = current_rigid_body_pos - current_anchor_pos.unsqueeze(1)
    ref_rel_pos = ref_rigid_body_pos - ref_anchor_pos.unsqueeze(1)

    # Rotate to anchor's local frame
    current_rel_pos_flat = current_rel_pos.reshape(-1, 3)
    current_heading_rot_inv_exp = current_heading_rot_inv.unsqueeze(1).expand(
        -1, current_rigid_body_pos.shape[1], -1
    ).reshape(-1, 4)
    current_rel_pos_local = quat_rotate(
        current_heading_rot_inv_exp, current_rel_pos_flat, w_last=True
    ).reshape(current_rigid_body_pos.shape)

    ref_rel_pos_flat = ref_rel_pos.reshape(-1, 3)
    ref_heading_rot_inv_exp = ref_heading_rot_inv.unsqueeze(1).expand(
        -1, ref_rigid_body_pos.shape[1], -1
    ).reshape(-1, 4)
    ref_rel_pos_local = quat_rotate(
        ref_heading_rot_inv_exp, ref_rel_pos_flat, w_last=True
    ).reshape(ref_rigid_body_pos.shape)

    return compute_global_position_error_exp(
        current_rel_pos_local,
        ref_rel_pos_local,
        sigma,
        body_indices,
        body_weights,
        fine_weight=fine_weight,
        fine_sigma=fine_sigma,
    )


def test_frame_extraction_left_the_reward_bitwise_identical():
    """RULE 10: extracting the frame block must not move a single ulp.

    A sibling lane found float32 ulp drift from a "harmless" vectorisation of
    exactly this kind of block. ``torch.equal`` is deliberate -- ``allclose``
    would pass on a drifted rewrite and let a silently different reward ship.
    """
    torch.manual_seed(11)
    for dtype in (torch.float32, torch.float64):
        cur = torch.randn(8, 7, 3, dtype=dtype)
        ref = torch.randn(8, 7, 3, dtype=dtype)
        cur_rot = torch.nn.functional.normalize(
            torch.randn(8, 7, 4, dtype=dtype), dim=-1
        )
        ref_rot = torch.nn.functional.normalize(
            torch.randn(8, 7, 4, dtype=dtype), dim=-1
        )
        for kwargs in (
            {},
            {"sigma": 0.3, "body_indices": torch.tensor([5, 6])},
            {"sigma": 0.3, "fine_weight": 0.5, "fine_sigma": 0.03},
        ):
            got = compute_relative_body_pos_rew(
                cur, ref, cur_rot[:, 0, :], ref_rot, cur[:, 0, :], 0, **kwargs
            )
            want = _pre_refactor_relative_body_pos_rew(
                cur, ref, cur_rot[:, 0, :], ref_rot, cur[:, 0, :], 0, **kwargs
            )
            assert torch.equal(got, want), (
                f"frame refactor changed the reward ({dtype}, {kwargs}): "
                f"max abs delta {(got - want).abs().max().item():.3e}"
            )
