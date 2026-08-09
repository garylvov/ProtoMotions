# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard tests for ``env/relative_body_pos/err_m_mean`` -- the FULL-BODY stat.

Why this stat exists: the owner's target is "full body to <2 cm with wrists",
and the full-body number was **invisible during training**. ``relative_body_pos``
(w 1.0, sigma 0.3, ~29 H1-2 bodies) logged only its Gaussian REWARD value, which
is flat across the entire sub-decimetre range the target lives in, so the
binding metric for a stated goal could only be read by running an offline eval.

What is pinned here:

1. **Rule 10 absence.** Component not registered (or no mimic reference) =>
   not one extras key.
2. **Type / prefix / shape / literal tag names.** Tensors (the agent's extras
   aggregator silently drops python floats), no ``raw/`` prefix (the aggregator
   skips it), one value per env, and the exact ``env/...`` strings a reader
   would put on a dashboard.
3. **METRE units against a hand-computed error**, in the anchor-relative
   heading-local frame (a pure root translation + yaw must read ZERO).
4. **UNWEIGHTED headline vs WEIGHTED companion.** With ``PM_BODY_WEIGHTS``-style
   weights live on this very component, ``err_m`` must be the plain mean and
   must NOT move when only the weights change; the weighted number is a
   separately-named companion.
5. **Anchor exclusion BY INDEX, not by literal 0.** A non-zero anchor must drop
   that body and keep body 0.
"""

from types import SimpleNamespace

import torch

from protomotions.envs.rewards.tracking import (
    compute_anchor_relative_local_body_pos,
)
from protomotions.utils.rotations import quat_rotate

KEY_MEAN = "relative_body_pos/err_m"
KEY_MAX = "relative_body_pos/err_max_m"
KEY_WEIGHTED = "relative_body_pos/err_m_weighted"


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
                {} if component is None else {"relative_body_pos": component}
            )
        ),
        extras={},
    )


def _component(body_indices=None, body_weights=None, body_names=None):
    """The ALL-BODY term, exactly as the teacher builds it (w 1.0, sigma 0.3)."""
    from protomotions.envs import component_factories as factories

    return factories.relative_body_pos_rew_factory(
        weight=1.0,
        sigma=0.3,
        body_indices=body_indices,
        body_weights=body_weights,
        body_names=body_names,
    )


def _weighted_component(weights, body_names):
    """Build the component the way ``PM_BODY_WEIGHTS`` builds it on a resume.

    The env gate writes ``body_indices=list(range(n))`` plus a full aligned
    ``body_weights`` vector onto the already-built component -- that is the
    shape the live v60 config has, so the stat must be tested against it and
    not only against the factory's name-map form.
    """
    from protomotions.envs.body_weight_env_gates import (
        apply_body_weight_env_overrides,
    )

    component = _component()
    components = {"relative_body_pos": component}
    changed = apply_body_weight_env_overrides(
        components,
        body_names,
        lambda _msg: None,
        "TEST",
        env={"PM_BODY_WEIGHTS": weights},
    )
    assert changed, "the gate must actually have armed for this fixture"
    return component


# =============================================================================
# 1. RULE 10 ABSENCE
# =============================================================================


def test_stat_is_silent_when_the_component_is_absent():
    """Rule 10: an unset config emits no new keys and pays no cost."""
    from protomotions.envs.base_env.env import BaseEnv

    cur = torch.zeros(2, 4, 3)
    env = _env(None)
    BaseEnv._log_relative_body_pos_extras(env, _ctx(cur, cur.clone()))
    assert env.extras == {}

    # ... and likewise with no mimic reference (a non-mimic env).
    ctx = _ctx(cur, cur.clone())
    env2 = _env(_component())
    BaseEnv._log_relative_body_pos_extras(
        env2, SimpleNamespace(current=ctx.current, mimic=None)
    )
    assert env2.extras == {}

    env3 = _env(_component())
    BaseEnv._log_relative_body_pos_extras(
        env3,
        SimpleNamespace(
            current=ctx.current, mimic=SimpleNamespace(ref_state=None, anchor_idx=0)
        ),
    )
    assert env3.extras == {}


def test_no_nan_when_the_only_scored_body_is_the_anchor():
    """Degenerate subset => emit NOTHING, never a 0/0 NaN.

    A NaN would render as a blank/complete-looking curve; an absent key is
    unambiguous.
    """
    from protomotions.envs.base_env.env import BaseEnv

    cur = torch.zeros(2, 4, 3)
    env = _env(_component(body_indices=[0]))
    BaseEnv._log_relative_body_pos_extras(env, _ctx(cur, cur.clone(), anchor_idx=0))
    assert env.extras == {}


# =============================================================================
# 2 + 3. TYPE, TAGS, METRES
# =============================================================================


def test_stat_type_prefix_shape_and_hand_computed_metres():
    """Registered => metre-valued per-env error, unweighted mean over bodies."""
    from protomotions.envs.base_env.env import BaseEnv

    # 4 bodies; body 0 is the anchor (pelvis) and is EXCLUDED, leaving 3.
    ref = torch.zeros(2, 4, 3)
    cur = torch.zeros(2, 4, 3)
    # env0: 3 cm, 7 cm, 2 cm on bodies 1..3  => mean 4 cm, max 7 cm.
    cur[0, 1, 0] = 0.03
    cur[0, 2, 1] = 0.07
    cur[0, 3, 2] = 0.02

    env = _env(_component())
    BaseEnv._log_relative_body_pos_extras(env, _ctx(cur, ref))

    assert set(env.extras) == {KEY_MEAN, KEY_MAX}, (
        "unweighted config must NOT emit the weighted companion"
    )
    for key, value in env.extras.items():
        assert isinstance(value, torch.Tensor), f"{key} must be a Tensor"
        assert not key.startswith("raw/"), f"{key} would be skipped by the agent"
        assert value.shape == (2,)

    assert torch.allclose(
        env.extras[KEY_MEAN], torch.tensor([0.04, 0.0]), atol=1e-6
    ), env.extras[KEY_MEAN]
    assert torch.allclose(
        env.extras[KEY_MAX], torch.tensor([0.07, 0.0]), atol=1e-6
    )


def test_units_are_metres_not_squared_metres():
    """A 10 cm displacement must read 0.10, not 0.01.

    The reward consumes SQUARED error; if this stat ever picked up the reward's
    pre-exponent quantity the "<2 cm" gate would read 4x optimistic near target
    and would cross the line while the robot was still 14 cm off.
    """
    from protomotions.envs.base_env.env import BaseEnv

    ref = torch.zeros(1, 3, 3)
    cur = torch.zeros(1, 3, 3)
    cur[0, 1, 0] = 0.10
    cur[0, 2, 0] = 0.10

    env = _env(_component())
    BaseEnv._log_relative_body_pos_extras(env, _ctx(cur, ref))
    assert torch.allclose(env.extras[KEY_MEAN], torch.tensor([0.10]), atol=1e-7)


def test_stat_is_anchor_relative_and_heading_local_not_world():
    """Root drift must contribute ZERO -- same frame as the reward it labels."""
    from protomotions.envs.base_env.env import BaseEnv

    torch.manual_seed(0)
    ref = torch.randn(3, 5, 3)
    ref_rot = torch.zeros(3, 5, 4)
    ref_rot[..., 3] = 1.0

    theta = torch.tensor([0.3, -1.1, 2.0])
    q = _yaw_quat(theta)
    offset = torch.tensor([[1.5, -2.0, 0.0], [0.0, 4.0, 0.0], [-3.0, 3.0, 0.0]])

    rel = ref - ref[:, 0:1, :]
    q_exp = q.unsqueeze(1).expand(-1, ref.shape[1], -1).reshape(-1, 4)
    rot_rel = quat_rotate(q_exp, rel.reshape(-1, 3), w_last=True).reshape(ref.shape)
    cur = rot_rel + ref[:, 0:1, :] + offset.unsqueeze(1)
    cur_rot = q.unsqueeze(1).expand(-1, ref.shape[1], -1).contiguous()

    env = _env(_component())
    BaseEnv._log_relative_body_pos_extras(env, _ctx(cur, ref, cur_rot, ref_rot))

    err = env.extras[KEY_MEAN]
    assert torch.allclose(err, torch.zeros(3), atol=1e-5), err
    world = (cur - ref).pow(2).sum(-1).sqrt()[:, 1:].mean(-1)
    assert (world > 0.5).all(), world


def test_stat_matches_the_reward_kernels_own_frame_math():
    """The stat and the reward must agree body-for-body, bitwise."""
    from protomotions.envs.base_env.env import BaseEnv

    torch.manual_seed(7)
    cur = torch.randn(4, 6, 3)
    ref = torch.randn(4, 6, 3)
    cur_rot = torch.nn.functional.normalize(torch.randn(4, 6, 4), dim=-1)
    ref_rot = torch.nn.functional.normalize(torch.randn(4, 6, 4), dim=-1)

    env = _env(_component())
    BaseEnv._log_relative_body_pos_extras(
        env, _ctx(cur, ref, cur_rot, ref_rot, anchor_idx=0)
    )

    cur_local, ref_local = compute_anchor_relative_local_body_pos(
        cur, ref, cur_rot[:, 0, :], ref_rot, cur[:, 0, :], 0
    )
    expected = (cur_local[:, 1:] - ref_local[:, 1:]).pow(2).sum(-1).sqrt()
    assert torch.equal(env.extras[KEY_MEAN], expected.mean(-1))
    assert torch.equal(env.extras[KEY_MAX], expected.max(-1).values)


def test_the_tb_tags_a_reader_would_dashboard():
    """READER/WRITER: pin the literal tag strings and the aggregator rules.

    The tag name is not chosen in the writer -- it is produced by ``agent.py``'s
    extras aggregator: skip ``raw/``, Tensors only, and for a multi-element
    tensor emit ``<key>_mean`` / ``<key>_std``, then prefix ``env/``. If any of
    those three rules changes, the full-body gate silently stops existing.
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
    assert 'extras_mean_std_dict[f"{key}_std"] = extra_val.std()' in agent_src
    assert 'env_log_dict = {f"env/{k}": v for k, v in env_log_dict.items()}' in agent_src

    assert f"env/{KEY_MEAN}_mean" == "env/relative_body_pos/err_m_mean"
    assert f"env/{KEY_MEAN}_std" == "env/relative_body_pos/err_m_std"
    assert f"env/{KEY_MAX}_mean" == "env/relative_body_pos/err_max_m_mean"
    assert (
        f"env/{KEY_WEIGHTED}_mean" == "env/relative_body_pos/err_m_weighted_mean"
    )

    # And the writer really emits those keys with numel > 1, so the _mean/_std
    # branch (not the scalar branch) is the one taken.
    from protomotions.envs.base_env.env import BaseEnv

    cur = torch.zeros(2, 4, 3)
    cur[0, 2, 0] = 0.03
    env = _env(_component())
    BaseEnv._log_relative_body_pos_extras(env, _ctx(cur, torch.zeros(2, 4, 3)))
    assert KEY_MEAN in env.extras and KEY_MAX in env.extras
    assert env.extras[KEY_MEAN].numel() == 2 > 1


# =============================================================================
# 4. UNWEIGHTED HEADLINE vs WEIGHTED COMPANION
# =============================================================================


def _five_body_names():
    return [
        "pelvis",
        "left_knee_link",
        "torso_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
    ]


def test_headline_is_UNWEIGHTED_and_does_not_move_when_weights_change():
    """THE point of this stat.

    v60 runs ``PM_BODY_WEIGHTS`` on THIS component, so the reward's reduction is
    a normalized WEIGHTED mean. If the stat inherited that weighting, the
    "<2 cm full body" number would improve merely because we re-weighted --
    the metric would stop describing the physical quantity the target names.
    Same geometry, two different weight specs, one identical headline.
    """
    from protomotions.envs.base_env.env import BaseEnv

    names = _five_body_names()
    ref = torch.zeros(1, 5, 3)
    cur = torch.zeros(1, 5, 3)
    # bodies 1..4: 8 cm, 8 cm, 2 cm, 2 cm  => unweighted mean 5 cm.
    cur[0, 1, 0] = 0.08
    cur[0, 2, 0] = 0.08
    cur[0, 3, 0] = 0.02
    cur[0, 4, 0] = 0.02

    headlines = []
    for spec in (
        "*_wrist_yaw_link=4.0,*_elbow_link=1.5",
        "*_wrist_yaw_link=40.0,*_elbow_link=15.0",
    ):
        env = _env(_weighted_component(spec, names))
        BaseEnv._log_relative_body_pos_extras(env, _ctx(cur, ref))
        headlines.append(env.extras[KEY_MEAN].clone())
        assert KEY_WEIGHTED in env.extras, "weighted runs emit the companion"

    assert torch.allclose(headlines[0], torch.tensor([0.05]), atol=1e-7), headlines
    assert torch.equal(headlines[0], headlines[1]), (
        "err_m moved when only the WEIGHTS changed -- it is not the unweighted "
        "physical mean any more"
    )


def test_weighted_companion_is_named_apart_and_is_the_weighted_metre_mean():
    """The weighted number is emitted, but under an unmistakable name.

    Hand-computed: bodies 1..4 at (8, 8, 2, 2) cm with weights (1, 1, 1.5, 4.0)
    => sum(w*e)/sum(w) = (0.08 + 0.08 + 0.03 + 0.08) / 7.5 = 0.036 m, clearly
    BELOW the 0.05 m unweighted truth. That gap is the dilution the weighting
    bought, and it is exactly why the headline may not be this number.
    """
    from protomotions.envs.base_env.env import BaseEnv

    names = _five_body_names()
    ref = torch.zeros(1, 5, 3)
    cur = torch.zeros(1, 5, 3)
    cur[0, 1, 0] = 0.08
    cur[0, 2, 0] = 0.08
    cur[0, 3, 0] = 0.02
    cur[0, 4, 0] = 0.02

    env = _env(_weighted_component("*_wrist_yaw_link=4.0,*_elbow_link=1.5", names))
    BaseEnv._log_relative_body_pos_extras(env, _ctx(cur, ref))

    assert set(env.extras) == {KEY_MEAN, KEY_MAX, KEY_WEIGHTED}
    assert isinstance(env.extras[KEY_WEIGHTED], torch.Tensor)
    assert torch.allclose(
        env.extras[KEY_WEIGHTED], torch.tensor([0.036]), atol=1e-7
    ), env.extras[KEY_WEIGHTED]
    assert env.extras[KEY_WEIGHTED] < env.extras[KEY_MEAN]


def test_weights_are_aligned_to_the_kept_bodies_after_anchor_removal():
    """Dropping the anchor must drop its WEIGHT too, not shift the vector.

    If the anchor's weight were left in ``sum(w)`` (or, worse, the weight vector
    slid by one body), every weighted number would be wrong by a body. Weight
    the LAST body heavily and check the companion lands where hand arithmetic
    puts it with the anchor's weight removed from both sums.
    """
    from protomotions.envs.base_env.env import BaseEnv

    names = _five_body_names()
    ref = torch.zeros(1, 5, 3)
    cur = torch.zeros(1, 5, 3)
    cur[0, 1, 0] = 0.10  # knee
    cur[0, 4, 0] = 0.01  # heavily-weighted wrist

    env = _env(_weighted_component("*_wrist_yaw_link=4.0", names))
    BaseEnv._log_relative_body_pos_extras(env, _ctx(cur, ref))

    # kept bodies 1..4, weights (1, 1, 1, 4), errors (0.10, 0, 0, 0.01)
    expected = (0.10 * 1 + 0.0 + 0.0 + 0.01 * 4) / 7.0
    assert torch.allclose(
        env.extras[KEY_WEIGHTED], torch.tensor([expected]), atol=1e-7
    ), env.extras[KEY_WEIGHTED]
    # ... and the headline is still the plain mean of the four kept errors.
    assert torch.allclose(
        env.extras[KEY_MEAN], torch.tensor([0.11 / 4.0]), atol=1e-7
    )


# =============================================================================
# 5. ANCHOR EXCLUSION BY INDEX
# =============================================================================


def test_anchor_is_excluded_and_by_index_not_by_literal_zero():
    """Move the anchor off body 0 and the exclusion must move with it.

    A sibling lane found this exact pattern hardcoded to body 0: a no-op while
    the anchor IS the pelvis at index 0, and a silently wrong full-body number
    the day the anchor moves. Here the anchor is body 2; body 0 carries a real
    error and MUST be counted, body 2 is a structural zero and MUST NOT be.
    """
    from protomotions.envs.base_env.env import BaseEnv

    anchor = 2
    ref = torch.zeros(1, 4, 3)
    cur = torch.zeros(1, 4, 3)
    cur[0, 0, 0] = 0.06  # body 0 -- real error, must be INCLUDED
    cur[0, 1, 0] = 0.03
    cur[0, 3, 0] = 0.03

    env = _env(_component())
    BaseEnv._log_relative_body_pos_extras(env, _ctx(cur, ref, anchor_idx=anchor))

    # In the anchor-centred frame every body shifts by -cur[anchor]; with the
    # anchor at the origin of both frames the errors of bodies 0/1/3 are
    # unchanged here (anchor has zero displacement), so mean = (6+3+3)/3 cm.
    assert torch.allclose(
        env.extras[KEY_MEAN], torch.tensor([0.04]), atol=1e-6
    ), env.extras[KEY_MEAN]
    # Three bodies, not four: a literal-0 exclusion would average over
    # {1, 2, 3} = (3 + 0 + 3)/3 = 2 cm and read HALF the true error.
    assert not torch.allclose(env.extras[KEY_MEAN], torch.tensor([0.02]), atol=1e-6)


def test_anchor_exclusion_survives_an_explicit_body_indices_subset():
    """Exclusion is applied to the SELECTED subset, positionally."""
    from protomotions.envs.base_env.env import BaseEnv

    anchor = 1
    ref = torch.zeros(1, 5, 3)
    cur = torch.zeros(1, 5, 3)
    cur[0, 0, 0] = 0.10
    cur[0, 3, 0] = 0.02

    # Subset explicitly CONTAINS the anchor; it must still be dropped.
    env = _env(_component(body_indices=[0, 1, 3]))
    BaseEnv._log_relative_body_pos_extras(env, _ctx(cur, ref, anchor_idx=anchor))
    assert torch.allclose(
        env.extras[KEY_MEAN], torch.tensor([0.06]), atol=1e-6
    ), env.extras[KEY_MEAN]


def test_full_body_stat_is_a_different_quantity_from_the_wrist_stat():
    """READER/WRITER sanity: the two stats must not collide or alias.

    Same step, both components registered: distinct keys, and the full-body
    mean must be pulled up by the non-wrist bodies. If a future edit points
    both writers at the same key, the full-body target silently reads the wrist
    number -- which is precisely the gap this stat was built to close.
    """
    from protomotions.envs import component_factories as factories
    from protomotions.envs.base_env.env import BaseEnv

    ref = torch.zeros(1, 5, 3)
    cur = torch.zeros(1, 5, 3)
    cur[0, 1, 0] = 0.10  # a leg: far off
    cur[0, 2, 0] = 0.10
    cur[0, 3, 0] = 0.01  # wrists: on target
    cur[0, 4, 0] = 0.01

    env = SimpleNamespace(
        config=SimpleNamespace(
            reward_components={
                "relative_body_pos": _component(),
                "wrist_relative_body_pos": factories.relative_body_pos_rew_factory(
                    weight=1.3, sigma=0.3, body_indices=[3, 4]
                ),
            }
        ),
        extras={},
    )
    ctx = _ctx(cur, ref)
    BaseEnv._log_wrist_relative_body_pos_extras(env, ctx)
    BaseEnv._log_relative_body_pos_extras(env, ctx)

    assert torch.allclose(
        env.extras["wrist_relative_body_pos/err_m"], torch.tensor([0.01]), atol=1e-7
    )
    assert torch.allclose(
        env.extras[KEY_MEAN], torch.tensor([0.055]), atol=1e-7
    )
    assert env.extras[KEY_MEAN] > env.extras["wrist_relative_body_pos/err_m"]
