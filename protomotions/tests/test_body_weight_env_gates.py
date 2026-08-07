# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard tests for the PER-BODY WEIGHTS env gate (PM_BODY_WEIGHTS).

The contract these tests pin, in order of importance:

1. **Rule 10 byte-identity.** ``PM_BODY_WEIGHTS`` unset -- and, separately, a
   spec that resolves to uniform -- must leave the reward config bit-for-bit
   what it was.
2. **A mistyped body name is a HARD ERROR**, never a silently-uniform run. This
   is the "looks like it worked" failure class that cost v57 its run.
3. **The weights actually reach the kernel and move the gradient** in the
   direction and by the factor the docstring claims
   (``w_i * N / sum(w)`` on the per-body gradient share).
4. **Idempotence.** The gate stores an absolute vector, so re-applying it on
   every autoresume writes nothing.
5. **Fresh-build and resume call the SAME function**, so they cannot drift.
"""

import copy
import pickle

import pytest
import torch

from protomotions.envs.body_weight_env_gates import (
    BODY_WEIGHTS_COMPONENTS_VAR,
    BODY_WEIGHTS_DEFAULT_VAR,
    BODY_WEIGHTS_VAR,
    DEFAULT_BODY_WEIGHT_COMPONENTS,
    apply_body_weight_env_overrides,
    body_weight_env_gate_requested,
    parse_body_weight_spec,
)
from protomotions.envs.component_factories import relative_body_pos_rew_factory
from protomotions.envs.rewards.tracking import compute_relative_body_pos_rew


#: The REAL H1-2 ordered body list (mjcf/h1_2_box_feet.xml), so the numbers
#: under test are the numbers that are actually live on the fleet.
H1_2_BODIES = [
    "pelvis",
    "left_hip_yaw_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_yaw_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "head_aux",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

#: The v60 recommendation, verbatim.
V60_SPEC = (
    "*_wrist_yaw_link=4.0,"
    "*_wrist_pitch_link=1.5,*_wrist_roll_link=1.5,"
    "*_elbow_link=1.5"
)


def _components():
    return {
        "relative_body_pos": relative_body_pos_rew_factory(weight=1.0, sigma=0.3),
        "wrist_relative_body_pos": relative_body_pos_rew_factory(
            weight=1.3, sigma=0.3, body_indices=[21, 28]
        ),
    }


def _collect(components, env, body_names=H1_2_BODIES):
    lines = []
    changed = apply_body_weight_env_overrides(
        components, body_names, log_fn=lines.append, label="TEST", env=env
    )
    return changed, lines


# =============================================================================
# 1. RULE 10 -- BYTE IDENTITY
# =============================================================================


def test_unset_is_a_hard_no_op_byte_identical():
    components = _components()
    before = pickle.dumps(components)

    assert body_weight_env_gate_requested({}) is False
    changed, lines = _collect(components, {})

    assert changed is False
    assert lines == [], f"the OFF path must be silent, got {lines}"
    assert pickle.dumps(components) == before


def test_empty_spec_is_treated_as_unset():
    components = _components()
    before = pickle.dumps(components)
    changed, lines = _collect(components, {BODY_WEIGHTS_VAR: ""})
    assert changed is False
    assert pickle.dumps(components) == before


def test_uniform_spec_writes_nothing_byte_identical():
    """Uniform-via-weights and uniform-via-None are the same reduction, so the
    gate must not perturb the pickle for no effect."""
    components = _components()
    before = pickle.dumps(components)

    changed, lines = _collect(components, {BODY_WEIGHTS_VAR: "*=2.0"})

    assert changed is False
    assert pickle.dumps(components) == before
    assert any("UNIFORM" in line for line in lines), lines


def test_modifier_vars_without_the_spec_warn_and_do_nothing():
    components = _components()
    before = pickle.dumps(components)
    changed, lines = _collect(
        components,
        {BODY_WEIGHTS_DEFAULT_VAR: "0.5", BODY_WEIGHTS_COMPONENTS_VAR: "relative_body_pos"},
    )
    assert changed is False
    assert pickle.dumps(components) == before
    assert any("SKIPPED" in line and "NO effect" in line for line in lines), lines


# =============================================================================
# 2. A MISTYPED BODY NAME IS A HARD ERROR
# =============================================================================


def test_pattern_matching_no_body_is_a_hard_error():
    with pytest.raises(ValueError, match="matched NO body"):
        parse_body_weight_spec("left_wrist_link=4.0", H1_2_BODIES)


def test_typo_in_a_glob_is_a_hard_error_not_a_uniform_run():
    components = _components()
    with pytest.raises(ValueError, match="matched NO body"):
        _collect(components, {BODY_WEIGHTS_VAR: "*_wrist_yaw=4.0"})


@pytest.mark.parametrize(
    "spec", ["left_elbow_link", "left_elbow_link=", "left_elbow_link=abc"]
)
def test_malformed_tokens_are_hard_errors(spec):
    with pytest.raises(ValueError):
        parse_body_weight_spec(spec, H1_2_BODIES)


@pytest.mark.parametrize("bad", ["-1", "nan", "inf"])
def test_negative_or_nonfinite_weights_are_hard_errors(bad):
    with pytest.raises(ValueError):
        parse_body_weight_spec(f"pelvis={bad}", H1_2_BODIES)


def test_missing_body_names_is_a_hard_error_not_a_guessed_alignment():
    components = _components()
    with pytest.raises(ValueError, match="ordered body name list"):
        apply_body_weight_env_overrides(
            components, None, log_fn=lambda _: None, label="TEST",
            env={BODY_WEIGHTS_VAR: V60_SPEC},
        )


# =============================================================================
# 3. PARSING
# =============================================================================


def test_v60_spec_resolves_to_the_documented_vector():
    weights = parse_body_weight_spec(V60_SPEC, H1_2_BODIES)
    by_name = dict(zip(H1_2_BODIES, weights))

    assert by_name["left_wrist_yaw_link"] == 4.0
    assert by_name["right_wrist_yaw_link"] == 4.0
    assert by_name["left_wrist_pitch_link"] == 1.5
    assert by_name["right_wrist_roll_link"] == 1.5
    assert by_name["left_elbow_link"] == 1.5
    # Untouched: legs, torso, head, shoulders, pelvis.
    for name in (
        "pelvis",
        "left_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "head_aux",
        "left_shoulder_pitch_link",
    ):
        assert by_name[name] == 1.0, name

    # The numbers the recommendation is argued from.
    assert sum(weights) == pytest.approx(38.0)
    assert len(weights) == 29


def test_later_tokens_override_earlier_ones():
    weights = parse_body_weight_spec(
        "*_wrist_*_link=2.0,left_wrist_yaw_link=9.0", H1_2_BODIES
    )
    by_name = dict(zip(H1_2_BODIES, weights))
    assert by_name["left_wrist_yaw_link"] == 9.0
    assert by_name["right_wrist_yaw_link"] == 2.0
    assert by_name["left_wrist_roll_link"] == 2.0


def test_default_weight_var_moves_the_unmatched_bodies():
    weights = parse_body_weight_spec(
        "*_wrist_yaw_link=4.0", H1_2_BODIES, default_weight=0.5
    )
    by_name = dict(zip(H1_2_BODIES, weights))
    assert by_name["left_wrist_yaw_link"] == 4.0
    assert by_name["left_knee_link"] == 0.5


# =============================================================================
# 4. THE WEIGHTS REACH THE KERNEL AND MOVE THE GRADIENT
# =============================================================================


def _scene(num_envs=2, num_bodies=29):
    torch.manual_seed(0)
    current_pos = torch.zeros(num_envs, num_bodies, 3, dtype=torch.float64)
    ref_pos = torch.zeros(num_envs, num_bodies, 3, dtype=torch.float64)
    rot = torch.zeros(num_envs, num_bodies, 4, dtype=torch.float64)
    rot[..., 3] = 1.0
    anchor_rot = torch.zeros(num_envs, 4, dtype=torch.float64)
    anchor_rot[:, 3] = 1.0
    anchor_pos = torch.zeros(num_envs, 3, dtype=torch.float64)
    return current_pos, ref_pos, rot, anchor_rot, anchor_pos


def _wrist_gradient(body_weights, wrist_err=0.0605, sigma=0.3):
    """d(reward)/d(left wrist error) at ``wrist_err`` metres, all else perfect."""
    current_pos, ref_pos, rot, anchor_rot, anchor_pos = _scene()
    e = torch.tensor(wrist_err, dtype=torch.float64, requires_grad=True)
    ref = ref_pos.clone()
    ref[:, 21, 0] = e  # left_wrist_yaw_link
    reward = compute_relative_body_pos_rew(
        current_pos, ref, anchor_rot, rot, anchor_pos,
        anchor_idx=0, sigma=sigma,
        body_weights=body_weights,
    )
    reward.sum().backward()
    return float(e.grad)


def test_weighting_the_wrist_scales_its_gradient_by_w_times_n_over_sumw():
    """The claim in the docstring, checked against autograd on the real kernel:
    per-body gradient share scales by ``w_i * N / sum(w)``."""
    weights = parse_body_weight_spec(V60_SPEC, H1_2_BODIES)
    n, total = len(weights), sum(weights)

    uniform = _wrist_gradient(None)
    weighted = _wrist_gradient(weights)

    predicted = 4.0 * n / total  # 4.0 * 29 / 38 = 3.0526...
    assert predicted == pytest.approx(3.0526, abs=1e-3)
    # Not exactly equal to the ratio because the exponential prefactor also
    # moves (the weighted mean squared error changes), but the wrist gradient
    # must GROW by close to the predicted factor.
    assert weighted / uniform == pytest.approx(predicted, rel=0.02)


def test_unweighted_bodies_lose_exactly_n_over_sumw_of_their_share():
    """The cost side of the trade, so a wrong weighting cannot starve balance
    bodies unnoticed."""
    weights = parse_body_weight_spec(V60_SPEC, H1_2_BODIES)
    n, total = len(weights), sum(weights)

    def knee_gradient(bw):
        current_pos, ref_pos, rot, anchor_rot, anchor_pos = _scene()
        e = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)
        ref = ref_pos.clone()
        ref[:, 4, 0] = e  # left_knee_link
        reward = compute_relative_body_pos_rew(
            current_pos, ref, anchor_rot, rot, anchor_pos,
            anchor_idx=0, sigma=0.3, body_weights=bw,
        )
        reward.sum().backward()
        return float(e.grad)

    ratio = knee_gradient(weights) / knee_gradient(None)
    assert ratio == pytest.approx(n / total, rel=0.02)  # 29/38 = 0.763
    assert ratio > 0.7, "the recommended weighting must not halve leg attention"


def test_gate_writes_keys_the_kernel_actually_consumes():
    components = _components()
    changed, lines = _collect(components, {BODY_WEIGHTS_VAR: V60_SPEC})
    assert changed is True

    sp = components["relative_body_pos"].static_params
    assert sp["body_indices"] == list(range(29))
    assert sp["body_weights"][21] == 4.0
    assert sp["body_weights"][4] == 1.0
    assert sum(sp["body_weights"]) == pytest.approx(38.0)

    # Default targets only relative_body_pos; the wrist-only term is untouched.
    assert "body_weights" not in components["wrist_relative_body_pos"].static_params


def test_component_subset_alignment_is_by_index_not_by_position():
    """Targeting a component that already has body_indices must project the
    full-body vector onto that subset, in the subset's own order."""
    components = _components()
    changed, _ = _collect(
        components,
        {
            BODY_WEIGHTS_VAR: "left_wrist_yaw_link=4.0",
            BODY_WEIGHTS_COMPONENTS_VAR: "wrist_relative_body_pos",
        },
    )
    assert changed is True
    sp = components["wrist_relative_body_pos"].static_params
    assert sp["body_indices"] == [21, 28]
    assert sp["body_weights"] == [4.0, 1.0]


def test_absent_component_warns_loudly():
    components = {"relative_body_pos": relative_body_pos_rew_factory()}
    changed, lines = _collect(
        components,
        {BODY_WEIGHTS_VAR: V60_SPEC, BODY_WEIGHTS_COMPONENTS_VAR: "does_not_exist"},
    )
    assert changed is False
    assert any("SKIPPED for 'does_not_exist'" in line for line in lines), lines


# =============================================================================
# 5. IDEMPOTENCE
# =============================================================================


def test_double_apply_is_a_no_op():
    """The v59 NOISE-DR defect must not be reintroduced here: this gate stores
    an ABSOLUTE vector, so every autoresume re-application is a no-op."""
    components = _components()
    changed_1, _ = _collect(components, {BODY_WEIGHTS_VAR: V60_SPEC})
    assert changed_1 is True
    settled = pickle.dumps(components)

    for _ in range(3):
        changed, lines = _collect(components, {BODY_WEIGHTS_VAR: V60_SPEC})
        assert changed is False
        assert pickle.dumps(components) == settled
        assert any("ALREADY at the requested weights" in line for line in lines), lines


def test_changed_spec_across_resumes_lands_on_the_new_absolute_vector():
    components = _components()
    _collect(components, {BODY_WEIGHTS_VAR: V60_SPEC})
    _collect(components, {BODY_WEIGHTS_VAR: "*_wrist_yaw_link=2.0"})

    sp = components["relative_body_pos"].static_params
    assert sp["body_weights"][21] == 2.0
    # The v60 spec's elbow/wrist-chain weights are GONE, not compounded.
    assert sp["body_weights"][18] == 1.0
    assert sum(sp["body_weights"]) == pytest.approx(31.0)


def test_idempotent_across_a_pickle_round_trip_like_a_real_resume():
    components = _components()
    _collect(components, {BODY_WEIGHTS_VAR: V60_SPEC})
    reloaded = pickle.loads(pickle.dumps(components))
    settled = pickle.dumps(reloaded)

    changed, _ = _collect(reloaded, {BODY_WEIGHTS_VAR: V60_SPEC})
    assert changed is False
    assert pickle.dumps(reloaded) == settled


# =============================================================================
# 6. BOTH WIRING PATHS CALL THE SAME IMPLEMENTATION
# =============================================================================


def test_fresh_build_and_resume_share_one_implementation():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    train_agent = (root / "protomotions" / "train_agent.py").read_text()
    assert "body_weight_env_gates import" in train_agent
    assert "apply_body_weight_env_overrides" in train_agent
    assert 'label="RESUME"' in train_agent

    teacher = (
        root.parent.parent
        / "src"
        / "imprint"
        / "integrations"
        / "wbc"
        / "training"
        / "teacher.py"
    )
    if teacher.exists():  # imprint parent repo; ProtoMotions may stand alone
        text = teacher.read_text()
        assert "body_weight_env_gates import" in text
        assert "apply_body_weight_env_overrides" in text
        assert 'label="FRESH-BUILD"' in text


def test_default_target_is_the_all_body_position_term_only():
    """Silently reshaping relative_body_ori too would confound the experiment."""
    assert DEFAULT_BODY_WEIGHT_COMPONENTS == ("relative_body_pos",)
