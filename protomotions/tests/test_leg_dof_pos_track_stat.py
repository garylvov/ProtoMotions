# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard tests for ``env/leg_dof_pos_track/err_rad_mean`` and its siblings.

WHY THIS FILE EXISTS -- the v66' reader/writer failure, stated plainly so it is
not repeated. v66' shipped ``leg_dof_pos_track`` with an eval surface
(``eval/leg_dof_pos_track/{mean,min,max}``, registered UNCONDITIONALLY) and that
surface WORKED: 0.21393 rad at epoch 100 falling to 0.20089 by epoch 250. But
the evaluator runs only every ``eval_metrics_every`` epochs, so it produced FOUR
points across 297 epochs, and nothing was ever written under
``env/leg_dof_pos_track/err_rad_mean`` -- a different namespace entirely. Anyone
querying the per-epoch tag got ``nan``, and the honest reading of that is that
the term's physical error was NOT observable during training. The metric was not
dead; the reader and the writer lived in two namespaces. That is the
reader/writer law failing quietly, which is exactly what the unconditional
metric was supposed to prevent.

So the test that matters here is the LAST one: it asserts the exact TB tag
strings a human or a gate will query, derived from the writer's own keys rather
than typed independently. A test that only checked "some stat is emitted" would
have passed in v66' too.

What is pinned:
1. **Rule 10 absence.** Component not registered => not one extras key.
2. **Type / prefix / shape.** Tensors (the agent's extras aggregator silently
   drops python floats), no ``raw/`` prefix (the aggregator skips it), one value
   per env.
3. **Correctness against hand-computed values**, including the SIGN of the knee
   channel -- an absolute value cannot tell under-flexion from over-flexion, and
   under-flexion (-0.3756 rad) is the entire finding.
4. **The subset comes from the REWARD COMPONENT**, never re-derived, so the
   stat cannot drift from the term it measures.
5. **The exact TB tag strings**, including ``env/leg_dof_pos_track/err_rad_mean``
   -- the tag whose absence was the v66' defect.
"""

from types import SimpleNamespace

import pytest
import torch

LEG_DOF_INDICES = [3, 9, 2, 8]  # l_knee, r_knee, l_hip_roll, r_hip_roll


def _ctx(cur_dof, ref_dof):
    return SimpleNamespace(
        current=SimpleNamespace(dof_pos=cur_dof),
        mimic=SimpleNamespace(ref_state=SimpleNamespace(dof_pos=ref_dof)),
    )


def _env(component):
    return SimpleNamespace(
        config=SimpleNamespace(
            reward_components=(
                {} if component is None else {"leg_dof_pos_track": component}
            )
        ),
        extras={},
    )


def _leg_component(dof_indices=LEG_DOF_INDICES):
    from protomotions.envs import component_factories as factories

    return factories.dof_pos_track_rew_factory(
        weight=1.3, sigma=0.40, dof_indices=dof_indices
    )


def _write(component, cur, ref):
    from protomotions.envs.base_env.env import BaseEnv

    env = _env(component)
    BaseEnv._log_leg_dof_pos_track_extras(env, _ctx(cur, ref))
    return env.extras


# =============================================================================
# 1. ABSENCE
# =============================================================================


def test_stat_is_silent_when_the_component_is_absent():
    """Rule 10: an unset config emits no new keys and pays no cost."""
    cur = torch.zeros(2, 27)
    assert _write(None, cur, cur.clone()) == {}


def test_stat_is_silent_without_a_mimic_reference():
    """A non-mimic env has no reference joint angles; emit nothing rather than a
    NaN, which would read as a live channel."""
    from protomotions.envs.base_env.env import BaseEnv

    cur = torch.zeros(2, 27)
    env = _env(_leg_component())
    BaseEnv._log_leg_dof_pos_track_extras(
        env, SimpleNamespace(current=SimpleNamespace(dof_pos=cur), mimic=None)
    )
    assert env.extras == {}

    env2 = _env(_leg_component())
    BaseEnv._log_leg_dof_pos_track_extras(
        env2,
        SimpleNamespace(
            current=SimpleNamespace(dof_pos=cur),
            mimic=SimpleNamespace(ref_state=None),
        ),
    )
    assert env2.extras == {}


# =============================================================================
# 2 + 3. THE WRITER
# =============================================================================


def test_stat_type_prefix_shape_and_hand_computed_values():
    ref = torch.zeros(2, 27)
    cur = torch.zeros(2, 27)
    # env 0 -- leg errors: l_knee -0.4, r_knee -0.2, l_hip_roll +0.1, r_hip_roll -0.1
    cur[0, 3], cur[0, 9], cur[0, 2], cur[0, 8] = -0.4, -0.2, 0.1, -0.1
    # env 1 -- all zero on the legs, but LARGE elsewhere: must not leak in.
    cur[1, 0], cur[1, 13] = 5.0, -5.0

    extras = _write(_leg_component(), cur, ref)

    assert set(extras) == {
        "leg_dof_pos_track/err_rad",
        "leg_dof_pos_track/err_max_rad",
        "leg_dof_pos_track/err_sq_rad2",
        "leg_dof_pos_track/knee_err_rad",
    }
    for key, value in extras.items():
        assert isinstance(value, torch.Tensor), f"{key} must be a Tensor"
        assert value.shape == (2,), f"{key} must be one value per env"
        assert not key.startswith("raw/"), f"{key} would be skipped by the aggregator"

    # mean |e| = (0.4 + 0.2 + 0.1 + 0.1) / 4 = 0.2
    assert torch.allclose(
        extras["leg_dof_pos_track/err_rad"], torch.tensor([0.2, 0.0]), atol=1e-6
    )
    # worst scored DOF = 0.4 (the knee)
    assert torch.allclose(
        extras["leg_dof_pos_track/err_max_rad"], torch.tensor([0.4, 0.0]), atol=1e-6
    )
    # e = (0.16 + 0.04 + 0.01 + 0.01) / 4 = 0.055 -- the kernel's own argument
    assert torch.allclose(
        extras["leg_dof_pos_track/err_sq_rad2"], torch.tensor([0.055, 0.0]), atol=1e-6
    )
    # env 1's 5.0 rad errors are on NON-scored DOFs and must read zero.
    assert float(extras["leg_dof_pos_track/err_max_rad"][1]) == 0.0


def test_knee_channel_is_SIGNED_because_under_flexion_is_the_whole_finding():
    """An absolute value cannot distinguish -0.3756 rad (under-flexed, the
    measured deep-crouch failure) from +0.3756 (over-flexed). If this ever
    becomes an abs(), the crouch story stops being readable."""
    ref = torch.zeros(2, 27)
    cur = torch.zeros(2, 27)
    cur[0, 3], cur[0, 9] = -0.40, -0.32   # under-flexed  -> mean -0.36
    cur[1, 3], cur[1, 9] = +0.40, +0.32   # over-flexed   -> mean +0.36

    knee = _write(_leg_component(), cur, ref)["leg_dof_pos_track/knee_err_rad"]
    assert torch.allclose(knee, torch.tensor([-0.36, 0.36]), atol=1e-6)
    assert float(knee[0]) < 0 < float(knee[1]), "the sign was destroyed"


def test_subset_is_taken_from_the_reward_component_not_re_derived():
    """The stat and the reward must score the SAME DOFs by construction. Passing
    a deliberately different subset must move the stat -- if it does not, the
    writer is using its own hardcoded indices and can silently disagree with the
    objective."""
    ref = torch.zeros(1, 27)
    cur = torch.zeros(1, 27)
    cur[0, 3] = 0.4     # a leg DOF
    cur[0, 20] = 0.9    # NOT in the leg subset

    leg = _write(_leg_component(), cur, ref)["leg_dof_pos_track/err_max_rad"]
    other = _write(_leg_component([20, 21]), cur, ref)["leg_dof_pos_track/err_max_rad"]
    assert float(leg) == pytest.approx(0.4, abs=1e-6)
    assert float(other) == pytest.approx(0.9, abs=1e-6)


def test_no_indices_means_all_dofs():
    """Defensive: a component built without dof_indices must not crash or
    silently score nothing."""
    from protomotions.envs import component_factories as factories

    ref = torch.zeros(1, 27)
    cur = torch.zeros(1, 27)
    cur[0, 20] = 0.9
    extras = _write(factories.dof_pos_track_rew_factory(weight=1.0), cur, ref)
    assert float(extras["leg_dof_pos_track/err_max_rad"]) == pytest.approx(0.9, abs=1e-6)


# =============================================================================
# 4. THE TAGS -- the guard that would have caught the v66' defect
# =============================================================================


def test_the_tb_tags_are_exactly_what_a_gate_queries():
    """THE REGRESSION GUARD. v66' emitted a real metric under
    ``eval/leg_dof_pos_track/mean`` while every query went to
    ``env/leg_dof_pos_track/err_rad_mean`` and got nan.

    The agent's extras aggregator publishes ``env/<key>_mean`` (and ``_std``)
    for each extras key, so the tag a human reads is derived here from the
    writer's OWN keys rather than typed independently -- a test that hardcoded
    both sides could agree with itself while disagreeing with the run.
    """
    ref = torch.zeros(3, 27)
    cur = torch.zeros(3, 27)
    cur[:, 3] = -0.3756  # the measured deep-crouch knee error
    keys = set(_write(_leg_component(), cur, ref))

    published = {f"env/{k}_mean" for k in keys} | {f"env/{k}_std" for k in keys}

    # The exact tag whose absence WAS the v66' defect.
    assert "env/leg_dof_pos_track/err_rad_mean" in published, (
        "the per-epoch joint-error tag is missing again -- this is the v66' "
        "reader/writer failure repeating. eval/leg_dof_pos_track/* is NOT a "
        "substitute: it fires every eval_metrics_every epochs (4 points in 297)."
    )
    for tag in (
        "env/leg_dof_pos_track/err_max_rad_mean",
        "env/leg_dof_pos_track/err_sq_rad2_mean",
        "env/leg_dof_pos_track/knee_err_rad_mean",
    ):
        assert tag in published, f"missing {tag}"


def test_stat_emits_on_every_call_not_only_on_eval_epochs():
    """The v66' surface was correct but 50x too sparse. This writer is called
    from the per-step extras path, so N calls must produce N live readings --
    the property that makes it readable from epoch 1."""
    ref = torch.zeros(1, 27)
    values = []
    for step in range(5):
        cur = torch.zeros(1, 27)
        cur[0, 3] = -0.40 + 0.02 * step  # a knee error that is actually moving
        extras = _write(_leg_component(), cur, ref)
        assert extras, f"call {step} emitted nothing"
        value = float(extras["leg_dof_pos_track/err_rad"])
        assert value == value, f"call {step} emitted NaN"  # NaN != NaN
        values.append(value)
    assert values == sorted(values, reverse=True), (
        f"a monotonically shrinking knee error must show up monotonically: {values}"
    )
