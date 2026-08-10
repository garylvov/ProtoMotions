# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PER-DOF weighting of the distillation MSE (PM_MM_DOF_WEIGHTS).

The load-bearing guarantee is the FIRST test: uniform weights must reproduce
``F.mse_loss`` exactly. Everything downstream -- loss scale, learning-rate
meaning, resume dynamics across the weighting boundary -- rests on it.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from protomotions.agents.common.supervision import (
    SupervisionLossConfig,
    SupervisionLossType,
)
from protomotions.agents.supervised.agent import SupervisedAgent
from protomotions.agents.supervised.config import SupervisedAgentConfig
from protomotions.agents.supervised.dof_weight_env_gates import (
    BASE_DOF_WEIGHT,
    DEFAULT_DOF_WEIGHT_SPEC,
    DOF_WEIGHT_SPEC_VAR,
    apply_dof_weight_env_overrides,
    dof_weight_env_gate_requested,
    format_dof_weight_proof,
    parse_dof_weight_spec,
    resolve_dof_groups,
    resolve_dof_weights,
    validate_dof_weights,
)

# The real H1-2 DOF ordering, as read out of a live
# results/<EXP>/resolved_configs.pt (robot.kinematic_info.dof_names). Hard-coded
# so these tests pin the ACTUAL production ordering rather than a convenient
# invention -- an index-addressed weighting bug would slip past a synthetic list.
H1_2_DOF_NAMES = [
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

NUM_ACTIONS = len(H1_2_DOF_NAMES)


class _IdentityPolicy(torch.nn.Module):
    """Passes ``privileged_action`` straight through, with a trainable scale."""

    out_keys = ["action", "privileged_action"]

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(NUM_ACTIONS))

    def forward(self, tensordict):
        tensordict["privileged_action"] = tensordict["raw_action"] * self.scale
        return tensordict

    def compute_model_loss(self, tensordict, current_epoch, zero_loss, log_prefix):
        return zero_loss * 0.0, {}


def _make_agent(policy, action_dim_weights=None, dof_names=H1_2_DOF_NAMES):
    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        model=SimpleNamespace(),
        loss=SupervisionLossConfig(
            loss_type=SupervisionLossType.MSE,
            prediction_key="privileged_action",
            target_key="expert_actions",
            log_prefix="masked_mimic",
        ),
        action_dim_weights=action_dim_weights,
    )
    agent.training_model = policy
    agent.model = policy
    agent.device = torch.device("cpu")
    agent.current_epoch = 0
    agent.calculate_extra_loss = lambda batch, actions: (
        torch.zeros((), device=agent.device),
        {},
    )
    agent.env = SimpleNamespace(
        robot_config=SimpleNamespace(
            number_of_actions=len(dof_names) if dof_names else 0,
            kinematic_info=SimpleNamespace(dof_names=dof_names),
        )
    )
    return agent


def _batch(seed=0, batch_size=5):
    generator = torch.Generator().manual_seed(seed)
    return {
        "action": torch.zeros(batch_size, NUM_ACTIONS),
        "raw_action": torch.randn(batch_size, NUM_ACTIONS, generator=generator),
        "expert_actions": torch.randn(batch_size, NUM_ACTIONS, generator=generator),
    }


# ---------------------------------------------------------------- compatibility


def test_uniform_weights_reproduce_stock_mse_loss_exactly():
    """THE compatibility guard: all-ones weights == today's ``F.mse_loss``.

    Normalizing by ``w.sum()`` (not by the number of dims, and not leaving the
    weights unnormalized) is what makes this hold. If this test ever goes red,
    every learning rate and every resume in this campaign silently changed
    meaning.
    """
    batch = _batch()
    ones = [1.0] * NUM_ACTIONS

    weighted_agent = _make_agent(_IdentityPolicy(), action_dim_weights=ones)
    stock_agent = _make_agent(_IdentityPolicy(), action_dim_weights=None)

    weighted_loss, weighted_log = weighted_agent.supervised_step(dict(batch))
    stock_loss, stock_log = stock_agent.supervised_step(dict(batch))

    expected = F.mse_loss(batch["raw_action"], batch["expert_actions"])

    assert torch.allclose(weighted_loss, expected, rtol=0.0, atol=1e-6)
    assert torch.allclose(weighted_loss, stock_loss, rtol=0.0, atol=1e-6)
    assert torch.allclose(
        weighted_log["masked_mimic/mse"], expected.detach(), rtol=0.0, atol=1e-6
    )
    assert torch.allclose(
        weighted_log["masked_mimic/mse"], stock_log["masked_mimic/mse"],
        rtol=0.0, atol=1e-6,
    )


def test_uniform_weights_match_stock_mse_to_within_one_ulp():
    """The same identity in float64, pinned to floating-point noise.

    The two expressions differ only in reduction ORDER (per-row sum then mean,
    versus a flat mean over every element), so they cannot be required to be
    bitwise equal. What CAN be required -- and is the thing that matters -- is
    that the residual stays at rounding scale rather than being a scale factor:
    a missing ``/ w.sum()`` normalizer would show up here as a factor of 27.
    """
    generator = torch.Generator().manual_seed(7)
    prediction = torch.randn(9, NUM_ACTIONS, generator=generator, dtype=torch.float64)
    target = torch.randn(9, NUM_ACTIONS, generator=generator, dtype=torch.float64)
    weights = torch.ones(NUM_ACTIONS, dtype=torch.float64)

    weighted = ((prediction - target).pow(2) * weights).sum(-1).div(weights.sum()).mean()
    stock = F.mse_loss(prediction, target)

    residual = (weighted - stock).abs().item()
    assert residual <= 4.0 * abs(stock.item()) * 2.220446049250313e-16
    assert torch.allclose(weighted, stock, rtol=1e-15, atol=0.0)


def test_nonuniform_weights_preserve_loss_scale_on_a_uniform_error():
    """A constant per-DOF error gives the SAME loss under ANY weight profile.

    This is the scale-invariance the ``/ w.sum()`` normalizer buys: weighting
    redistributes gradient between DOFs, it does not inflate or deflate the
    objective. Without it, the default profile would multiply the loss by
    50/27 = 1.85 and every logged curve would step at the moment of the change.
    """
    batch = _batch()
    batch["expert_actions"] = batch["raw_action"] + 0.5

    weights = resolve_dof_weights(DEFAULT_DOF_WEIGHT_SPEC, H1_2_DOF_NAMES)
    weighted_loss, _ = _make_agent(
        _IdentityPolicy(), action_dim_weights=weights
    ).supervised_step(dict(batch))
    stock_loss, _ = _make_agent(_IdentityPolicy()).supervised_step(dict(batch))

    assert torch.allclose(weighted_loss, stock_loss, rtol=0.0, atol=1e-6)
    assert torch.allclose(weighted_loss, torch.tensor(0.25), rtol=0.0, atol=1e-6)


# ---------------------------------------------------------------------- gradient


def test_zero_weight_on_a_group_removes_that_groups_gradient_entirely():
    """Weight 0 on the wrists means the wrist DOFs earn NO gradient at all."""
    weights = resolve_dof_weights("*_wrist_*=0.0", H1_2_DOF_NAMES)
    wrist_idx = resolve_dof_groups(H1_2_DOF_NAMES)["wrists"]
    assert len(wrist_idx) == 6

    policy = _IdentityPolicy()
    agent = _make_agent(policy, action_dim_weights=weights)
    loss, _ = agent.supervised_step(_batch(seed=3))
    loss.backward()

    grad = policy.scale.grad
    assert grad is not None
    for i in wrist_idx:
        assert grad[i].item() == 0.0, f"dof {H1_2_DOF_NAMES[i]} still has gradient"
    other = [i for i in range(NUM_ACTIONS) if i not in wrist_idx]
    assert grad[other].abs().sum().item() > 0.0


def test_upweighting_wrists_raises_their_share_of_the_gradient():
    """The default profile really does move gradient toward the hands.

    The batch is built with an IDENTICAL per-DOF error so that every DOF's
    gradient magnitude is equal under the stock loss. The measured shares are
    then exactly the design numbers -- wrists 22.22% -> 48.00% -- rather than
    an artifact of which DOFs happened to be noisier in this batch.
    """
    groups = resolve_dof_groups(H1_2_DOF_NAMES)
    batch = _batch(seed=11)
    # Identical input AND identical error on every DOF, so d(loss)/d(scale_i)
    # is the same for all i under the stock loss and the measured shares are
    # purely the weight profile.
    batch["raw_action"] = torch.ones(batch["raw_action"].shape)
    batch["expert_actions"] = batch["raw_action"] + 0.5

    def grad_shares(action_dim_weights):
        policy = _IdentityPolicy()
        agent = _make_agent(policy, action_dim_weights=action_dim_weights)
        loss, _ = agent.supervised_step(dict(batch))
        loss.backward()
        grad = policy.scale.grad.abs()
        total = grad.sum()
        return {g: (grad[idx].sum() / total).item() for g, idx in groups.items()}

    stock = grad_shares(None)
    weighted = grad_shares(resolve_dof_weights(DEFAULT_DOF_WEIGHT_SPEC, H1_2_DOF_NAMES))

    assert stock["wrists"] == pytest.approx(6 / 27, abs=1e-4)
    assert weighted["wrists"] == pytest.approx(24 / 50, abs=1e-4)
    assert weighted["elbows"] == pytest.approx(4 / 50, abs=1e-4)
    assert weighted["shoulders"] == pytest.approx(9 / 50, abs=1e-4)
    # Locomotion must not be starved: the lower body keeps ~26% of the gradient.
    assert weighted["lower"] == pytest.approx(13 / 50, abs=1e-4)
    assert weighted["lower"] > 0.25


# ------------------------------------------------------------------- resolution


def test_glob_matching_nothing_raises_value_error():
    """A typo'd pattern is FATAL, never a silent fall-back to uniform."""
    with pytest.raises(ValueError) as error:
        resolve_dof_weights("*_wrist_joint=4.0", H1_2_DOF_NAMES)
    assert "matched NONE" in str(error.value)

    with pytest.raises(ValueError):
        resolve_dof_weights("*_wrist_*=4.0,*_finger_*=2.0", H1_2_DOF_NAMES)


def test_resolved_vector_length_mismatch_raises():
    with pytest.raises(ValueError) as error:
        validate_dof_weights([1.0] * 26, NUM_ACTIONS)
    assert "number_of_actions=27" in str(error.value)

    validate_dof_weights([1.0] * NUM_ACTIONS, NUM_ACTIONS)


def test_default_spec_resolves_to_the_documented_profile():
    weights = resolve_dof_weights("default", H1_2_DOF_NAMES)
    assert weights == resolve_dof_weights(DEFAULT_DOF_WEIGHT_SPEC, H1_2_DOF_NAMES)
    assert len(weights) == NUM_ACTIONS

    by_name = dict(zip(H1_2_DOF_NAMES, weights))
    assert by_name["left_wrist_yaw_joint"] == 4.0
    assert by_name["right_wrist_roll_joint"] == 4.0
    assert by_name["left_elbow_joint"] == 2.0
    assert by_name["right_shoulder_pitch_joint"] == 1.5
    assert by_name["torso_joint"] == BASE_DOF_WEIGHT
    assert by_name["left_knee_joint"] == BASE_DOF_WEIGHT
    assert sum(weights) == pytest.approx(50.0)


def test_weights_follow_names_not_indices_when_dof_order_changes():
    """Reversing the DOF ordering moves the weights with the NAMES.

    This is the whole reason the spec is glob-addressed: an index list would
    keep pointing at slots 17-19 and would start weighting the legs.
    """
    reordered = list(reversed(H1_2_DOF_NAMES))
    weights = resolve_dof_weights("default", reordered)
    by_name = dict(zip(reordered, weights))
    assert by_name["left_wrist_yaw_joint"] == 4.0
    assert by_name["left_knee_joint"] == BASE_DOF_WEIGHT


def test_later_pattern_wins_on_overlap():
    weights = resolve_dof_weights("*_joint=2.0,*_wrist_*=5.0", H1_2_DOF_NAMES)
    by_name = dict(zip(H1_2_DOF_NAMES, weights))
    assert by_name["left_wrist_pitch_joint"] == 5.0
    assert by_name["left_knee_joint"] == 2.0


def test_spec_parse_rejects_malformed_and_negative_entries():
    assert parse_dof_weight_spec("a*=1.5") == [("a*", 1.5)]
    for bad in ("", "  ", "*_wrist_*", "*_wrist_*=abc", "=4.0", "*_wrist_*=-1.0",
                "*_wrist_*=nan", "*_wrist_*=inf"):
        with pytest.raises(ValueError):
            parse_dof_weight_spec(bad)


def test_resolve_requires_dof_names():
    with pytest.raises(ValueError) as error:
        resolve_dof_weights("default", [])
    assert "dof_names" in str(error.value)


# ------------------------------------------------------------------------ groups


def test_dof_groups_partition_the_h1_2_action_vector():
    groups = resolve_dof_groups(H1_2_DOF_NAMES)
    assert len(groups["wrists"]) == 6
    assert len(groups["elbows"]) == 2
    assert len(groups["shoulders"]) == 6
    assert len(groups["waist"]) == 1
    assert len(groups["legs"]) == 12
    assert len(groups["arms"]) == 14
    assert len(groups["lower"]) == 13
    assert "other" not in groups

    fine = ["wrists", "elbows", "shoulders", "waist", "legs"]
    covered = sorted(i for name in fine for i in groups[name])
    assert covered == list(range(NUM_ACTIONS))


def test_unmatched_dofs_land_in_other_rather_than_vanishing():
    groups = resolve_dof_groups(["left_knee_joint", "gripper_joint"])
    assert groups["legs"] == [0]
    assert groups["other"] == [1]


# ------------------------------------------------------------------- the reader


def test_per_group_mse_is_logged_even_when_weights_are_absent():
    """The reader works on BOTH sides of the change, so curves are comparable."""
    batch = _batch(seed=5)
    _, stock_log = _make_agent(_IdentityPolicy()).supervised_step(dict(batch))

    for group in ("wrists", "elbows", "shoulders", "waist", "legs", "arms", "lower"):
        assert f"masked_mimic/mse_group/{group}" in stock_log

    weights = resolve_dof_weights("default", H1_2_DOF_NAMES)
    _, weighted_log = _make_agent(
        _IdentityPolicy(), action_dim_weights=weights
    ).supervised_step(dict(batch))

    # The readout is UNWEIGHTED, so the same batch gives the same per-group
    # numbers regardless of the weight profile -- that is what makes a weighted
    # run comparable against the unweighted history.
    for group in ("wrists", "legs"):
        key = f"masked_mimic/mse_group/{group}"
        assert torch.allclose(stock_log[key], weighted_log[key])


def test_per_group_mse_matches_a_hand_computed_group_mean():
    batch = _batch(seed=9)
    agent = _make_agent(_IdentityPolicy())
    _, log_dict = agent.supervised_step(dict(batch))

    wrist_idx = resolve_dof_groups(H1_2_DOF_NAMES)["wrists"]
    expected = (
        (batch["raw_action"] - batch["expert_actions"])[:, wrist_idx].pow(2).mean()
    )
    assert torch.allclose(log_dict["masked_mimic/mse_group/wrists"], expected)


def test_per_group_readout_disabled_without_dof_names_but_loss_still_works():
    batch = _batch(seed=13)
    agent = _make_agent(_IdentityPolicy(), dof_names=None)
    loss, log_dict = agent.supervised_step(dict(batch))

    assert torch.allclose(
        loss, F.mse_loss(batch["raw_action"], batch["expert_actions"])
    )
    assert not [k for k in log_dict if "mse_group" in k]


# --------------------------------------------------------------------- the gate


def test_gate_is_a_hard_noop_when_the_env_var_is_absent():
    assert dof_weight_env_gate_requested({}) is False
    config = SupervisedAgentConfig(batch_size=1, training_max_steps=1)
    calls = []

    changed = apply_dof_weight_env_overrides(
        config,
        dof_names=H1_2_DOF_NAMES,
        log_fn=calls.append,
        label="FRESH-BUILD",
        env={},
        number_of_actions=NUM_ACTIONS,
    )

    assert changed is False
    assert calls == []
    assert config.action_dim_weights is None


def test_gate_writes_the_resolved_vector_and_a_loud_proof():
    config = SupervisedAgentConfig(batch_size=1, training_max_steps=1)
    lines = []

    changed = apply_dof_weight_env_overrides(
        config,
        dof_names=H1_2_DOF_NAMES,
        log_fn=lines.append,
        label="RESUME",
        env={DOF_WEIGHT_SPEC_VAR: "default"},
        number_of_actions=NUM_ACTIONS,
    )

    assert changed is True
    assert config.action_dim_weights == resolve_dof_weights("default", H1_2_DOF_NAMES)
    text = "\n".join(lines)
    assert "[DOF-WEIGHTS] RESUME" in text
    for group in ("wrists", "elbows", "shoulders", "waist", "legs"):
        assert group in text
    assert "gradient share" in text
    assert "mse_group" in text  # the proof names its own reader


def test_gate_raises_on_a_bad_pattern_rather_than_training_unweighted():
    config = SupervisedAgentConfig(batch_size=1, training_max_steps=1)
    with pytest.raises(ValueError):
        apply_dof_weight_env_overrides(
            config,
            dof_names=H1_2_DOF_NAMES,
            log_fn=lambda _m: None,
            label="RESUME",
            env={DOF_WEIGHT_SPEC_VAR: "*_gripper_*=4.0"},
            number_of_actions=NUM_ACTIONS,
        )
    assert config.action_dim_weights is None


def test_gate_raises_when_dof_names_are_unavailable():
    config = SupervisedAgentConfig(batch_size=1, training_max_steps=1)
    with pytest.raises(ValueError) as error:
        apply_dof_weight_env_overrides(
            config,
            dof_names=None,
            log_fn=lambda _m: None,
            label="RESUME",
            env={DOF_WEIGHT_SPEC_VAR: "default"},
        )
    assert "dof_names" in str(error.value)


def test_gate_raises_when_the_agent_config_cannot_carry_weights():
    with pytest.raises(ValueError) as error:
        apply_dof_weight_env_overrides(
            SimpleNamespace(),
            dof_names=H1_2_DOF_NAMES,
            log_fn=lambda _m: None,
            label="FRESH-BUILD",
            env={DOF_WEIGHT_SPEC_VAR: "default"},
        )
    assert "action_dim_weights" in str(error.value)


def test_gate_raises_on_action_dim_mismatch():
    config = SupervisedAgentConfig(batch_size=1, training_max_steps=1)
    with pytest.raises(ValueError) as error:
        apply_dof_weight_env_overrides(
            config,
            dof_names=H1_2_DOF_NAMES,
            log_fn=lambda _m: None,
            label="RESUME",
            env={DOF_WEIGHT_SPEC_VAR: "default"},
            number_of_actions=42,
        )
    assert "number_of_actions=42" in str(error.value)


def test_proof_line_reports_the_shifted_gradient_share():
    weights = resolve_dof_weights("default", H1_2_DOF_NAMES)
    text = "\n".join(
        format_dof_weight_proof(weights, H1_2_DOF_NAMES, "LOSS", "test")
    )
    # wrists: 24/50 = 48.00%, uniform 6/27 = 22.22%
    assert "48.00%" in text
    assert "22.22%" in text
    assert "sum(w)=50.0000" in text


# ---------------------------------------------------- non-MSE branches untouched


def test_non_mse_loss_types_are_delegated_untouched():
    from protomotions.agents.common.supervision import compute_supervision_loss

    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        loss=SupervisionLossConfig(
            loss_type=SupervisionLossType.DISCRETE_KL,
            prediction_key="logits",
            target_key="target_logits",
            log_prefix="latent",
        ),
        action_dim_weights=None,
    )
    batch = TensorDict(
        {
            "logits": torch.randn(4, 3),
            "target_logits": torch.randn(4, 3),
        },
        batch_size=4,
    )

    loss, log_dict = agent._compute_supervision_loss(batch)
    expected, expected_log = compute_supervision_loss(batch, agent.config.loss)

    assert torch.equal(loss, expected)
    assert set(log_dict) == set(expected_log)


def test_weights_on_a_non_mse_loss_type_raise():
    agent = object.__new__(SupervisedAgent)
    agent.config = SimpleNamespace(
        loss=SupervisionLossConfig(
            loss_type=SupervisionLossType.DISCRETE_KL,
            prediction_key="logits",
            target_key="target_logits",
        ),
        action_dim_weights=[1.0] * NUM_ACTIONS,
    )
    batch = TensorDict(
        {"logits": torch.randn(4, 3), "target_logits": torch.randn(4, 3)},
        batch_size=4,
    )

    with pytest.raises(ValueError) as error:
        agent._compute_supervision_loss(batch)
    assert "only supported for loss_type=mse" in str(error.value)


# ------------------------------------------------------------------ resume wiring


def _read(*parts) -> str:
    return (Path(__file__).resolve().parents[2].joinpath(*parts)).read_text()


def test_train_agent_applies_the_shared_gate_on_both_paths():
    """The fresh-build and resume paths must use ONE shared implementation.

    The gate call sits AFTER the fresh/resume branches converge, so there is
    exactly one call site and the two paths cannot drift.
    """
    text = _read("protomotions", "train_agent.py")
    assert "dof_weight_env_gates import" in text
    assert text.count("apply_dof_weight_env_overrides(") == 1
    assert 'label="RESUME" if mode == "resume" else "FRESH-BUILD"' in text


def test_stage_resume_config_can_bake_weights_into_the_frozen_pickle():
    """``detect_checkpoint_mode`` never loads the experiment file on a resume.

    ``results/<EXP>/config.yaml`` is written back onto args and
    ``resolved_configs.pt`` supplies the real values, so a CLI flag alone is
    inert. ``stage_resume_config.py --dof-weights`` rewrites the frozen pickle,
    through this same module, and verifies the round-trip.
    """
    # tests/ -> protomotions/ -> ProtoMotions/ -> third_party/ -> <run tree>/
    # The staging tool belongs to the RUN tree that vendors this repo, not to
    # this repo, so a bare ProtoMotions checkout legitimately has no such file
    # and skips. Inside the run tree the guard is live.
    script = Path(__file__).resolve().parents[4] / "stage_resume_config.py"
    if not script.is_file():
        pytest.skip(f"no vendoring run tree at {script.parent} (bare checkout)")
    text = script.read_text()
    assert "--dof-weights" in text
    assert "dof_weight_env_gates import" in text
    assert "resolve_dof_weights" in text
    assert "validate_dof_weights" in text
    assert "action_dim_weights did not survive the round-trip" in text


def test_weights_survive_a_resolved_configs_round_trip(tmp_path):
    """The resume contract, end to end: resolve -> pickle -> unpickle -> loss.

    ``resolved_configs.pt`` is the only thing a resume reads, so the vector has
    to come back out of a torch.save/torch.load byte-for-byte and still drive
    the loss.
    """
    config = SupervisedAgentConfig(batch_size=1, training_max_steps=1)
    apply_dof_weight_env_overrides(
        config,
        dof_names=H1_2_DOF_NAMES,
        log_fn=lambda _m: None,
        label="STAGED",
        env={DOF_WEIGHT_SPEC_VAR: "default"},
        number_of_actions=NUM_ACTIONS,
    )
    expected = list(config.action_dim_weights)

    path = tmp_path / "resolved_configs.pt"
    torch.save({"agent": config}, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)["agent"]

    assert list(restored.action_dim_weights) == expected
    validate_dof_weights(restored.action_dim_weights, NUM_ACTIONS)

    # And the restored vector actually drives the loss it was staged for.
    agent = _make_agent(_IdentityPolicy(), action_dim_weights=restored.action_dim_weights)
    batch = _batch(seed=17)
    loss, _ = agent.supervised_step(dict(batch))
    squared_error = (batch["raw_action"] - batch["expert_actions"]).pow(2)
    weights = torch.tensor(expected)
    manual = (squared_error * weights).sum(-1).div(weights.sum()).mean()
    assert torch.allclose(loss, manual, rtol=0.0, atol=1e-6)
