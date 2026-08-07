# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard tests for the NOISE-DR env gate (PM_ACTION_NOISE_SCALE / PM_OBS_NOISE_SCALE
/ PM_ANCHOR_ROT_NOISE_SCALE).

The contract these tests pin, in order of importance:

1. **Rule 10 byte-identity.** An unset environment -- and, separately, a scale
   that resolves to exactly 1.0 -- must leave the config *bit-for-bit* what it
   was. Not "numerically equal": the pickled bytes must match. Both are proved
   by round-tripping ``pickle.dumps`` around the gate call, so a future rewrite
   that switches the OFF path to ``value * 1.0`` (numerically exact today, one
   ulp off the moment anything becomes a float32 or a tensor) fails here.
2. **The gate actually moves the two magnitudes the wrist analysis blames** --
   ``action_noise_range`` (26.4 mm rms wrist-equivalent) and
   ``observation_noise.dof_pos_noise`` (10.6 mm) -- by the requested factor.
3. **Absent targets warn loudly.** A set knob whose config block is stripped is
   the exact "delta silently no-ops" failure class that cost v57 its run.
4. **Validation.** Negative / non-finite / unparseable scales are hard errors.
5. **Fresh-build and resume call the SAME function**, so the two paths cannot
   drift apart.
"""

import pickle
import re

import pytest

from protomotions.simulator.base_simulator.config import (
    ActionNoiseDomainRandomizationConfig,
    DomainRandomizationConfig,
    RobotNoiseConfig,
)
from protomotions.simulator.base_simulator.noise_scale_env_gates import (
    ANCHOR_ROT_NOISE_SCALE_VAR,
    ACTION_NOISE_SCALE_VAR,
    NOISE_SCALE_ENV_VARS,
    OBS_NOISE_FIELDS,
    OBS_NOISE_SCALE_VAR,
    apply_noise_scale_env_overrides,
    noise_scale_env_gate_requested,
)


def _v57_dr() -> DomainRandomizationConfig:
    """The real v57 / v58 non-ramped noise block.

    Mirrors ``imprint``'s ``stages_night13.build_domain_randomization`` exactly
    (``stages_night13.py:286-288`` and ``:303-307``) so the numbers under test
    are the numbers that are actually live on the fleet.
    """
    return DomainRandomizationConfig(
        action_noise=ActionNoiseDomainRandomizationConfig(
            action_noise_range=(-0.05, 0.05), dof_names=[".*"], dof_indices=None
        ),
        observation_noise=RobotNoiseConfig(
            dof_pos_noise=0.02,
            dof_vel_noise=1.0,
            anchor_ang_vel_noise=0.4,
            anchor_rot_noise=0.1,
        ),
    )


def _collect(dr, env):
    """Run the gate against ``env``, returning (changed, log lines)."""
    lines = []
    changed = apply_noise_scale_env_overrides(
        dr, log_fn=lines.append, label="TEST", env=env
    )
    return changed, lines


# =============================================================================
# 1. RULE 10 -- BYTE IDENTITY
# =============================================================================


def test_unset_environment_is_a_hard_no_op_byte_identical():
    """No knob present => not one field written, not one line logged."""
    dr = _v57_dr()
    before = pickle.dumps(dr)

    assert noise_scale_env_gate_requested({}) is False
    changed, lines = _collect(dr, {})

    assert changed is False
    assert lines == [], f"the OFF path must be silent, got {lines}"
    assert pickle.dumps(dr) == before, "unset environment perturbed the config"


@pytest.mark.parametrize(
    "env",
    [
        {ACTION_NOISE_SCALE_VAR: "1.0"},
        {OBS_NOISE_SCALE_VAR: "1.0"},
        {ANCHOR_ROT_NOISE_SCALE_VAR: "1.0"},
        {
            ACTION_NOISE_SCALE_VAR: "1",
            OBS_NOISE_SCALE_VAR: "1.0",
            ANCHOR_ROT_NOISE_SCALE_VAR: "1.00",
        },
    ],
)
def test_scale_of_exactly_one_writes_nothing_byte_identical(env):
    """An EXPLICIT 1.0 is byte-identical too.

    This is the ulp guard. ``0.05 * 1.0`` happens to be exact in python doubles,
    so a naive implementation passes an ``==`` check and still violates the
    contract the moment a magnitude is a float32 or a tensor. The gate must keep
    the LITERAL ORIGINAL object, so the pickled bytes cannot move.
    """
    dr = _v57_dr()
    before = pickle.dumps(dr)

    changed, lines = _collect(dr, env)

    assert changed is False
    assert pickle.dumps(dr) == before, "an explicit 1.0 scale perturbed the config"
    # ... but it is NOT silent: a set-and-inert knob must still be visible.
    assert lines, "an explicitly-set knob must log, even when it writes nothing"
    assert any("UNCHANGED" in ln or "wrote\nNOTHING" in ln or "NOTHING" in ln
               for ln in lines), lines


def test_already_zero_magnitudes_are_not_rewritten():
    """0 * s is 0, so an all-zero frozen config stays byte-identical."""
    dr = DomainRandomizationConfig(observation_noise=RobotNoiseConfig())
    before = pickle.dumps(dr)
    changed, _ = _collect(dr, {OBS_NOISE_SCALE_VAR: "0.25"})
    assert changed is False
    assert pickle.dumps(dr) == before


# =============================================================================
# 2. THE GATE MOVES THE MAGNITUDES THE WRIST ANALYSIS BLAMES
# =============================================================================


def test_action_noise_scale_moves_the_persistent_pd_bias():
    """PM_ACTION_NOISE_SCALE=0.2 : +/-0.05 rad -> +/-0.01 rad (26.4 -> 5.3 mm)."""
    dr = _v57_dr()
    changed, lines = _collect(dr, {ACTION_NOISE_SCALE_VAR: "0.2"})

    assert changed is True
    lo, hi = dr.action_noise.action_noise_range
    assert lo == pytest.approx(-0.01) and hi == pytest.approx(0.01)
    # Observation noise is a DIFFERENT knob and must be untouched.
    assert dr.observation_noise.dof_pos_noise == 0.02
    assert dr.observation_noise.anchor_rot_noise == 0.1
    # Loud proof line naming old -> new.
    proof = "\n".join(lines)
    assert "action_noise_range" in proof
    assert "-0.05" in proof and "0.2" in proof


def test_obs_noise_scale_moves_every_magnitude_field():
    """PM_OBS_NOISE_SCALE=0.25 scales the whole RobotNoiseConfig."""
    dr = _v57_dr()
    changed, lines = _collect(dr, {OBS_NOISE_SCALE_VAR: "0.25"})

    assert changed is True
    obs = dr.observation_noise
    assert obs.dof_pos_noise == pytest.approx(0.005)  # 10.6 -> 2.6 mm
    assert obs.dof_vel_noise == pytest.approx(0.25)
    assert obs.anchor_ang_vel_noise == pytest.approx(0.1)
    assert obs.anchor_rot_noise == pytest.approx(0.025)  # 10.9 deg -> 2.7 deg
    # Action noise is a DIFFERENT knob and must be untouched.
    assert dr.action_noise.action_noise_range == (-0.05, 0.05)
    assert any("dof_pos_noise" in ln for ln in lines)


def test_anchor_rot_scale_composes_on_top_of_the_obs_scale():
    """The anchor knob is an EXTRA multiplier, not a replacement.

    It earns its own gate because anchor_rot_noise=0.1 is 0.190 rad = 10.9 deg
    of pelvis attitude error -- large, physically implausible for a precision
    task, and with an effect (bias-path conservatism) the analysis could only
    bound, not measure. Attributing it needs it to move independently.
    """
    dr = _v57_dr()
    _collect(dr, {OBS_NOISE_SCALE_VAR: "0.5", ANCHOR_ROT_NOISE_SCALE_VAR: "0.5"})
    obs = dr.observation_noise
    assert obs.anchor_rot_noise == pytest.approx(0.025)  # 0.1 * 0.5 * 0.5
    assert obs.dof_pos_noise == pytest.approx(0.01)  # 0.02 * 0.5 only

    # And it works alone, with PM_OBS_NOISE_SCALE unset (base = 1.0).
    dr2 = _v57_dr()
    _collect(dr2, {ANCHOR_ROT_NOISE_SCALE_VAR: "0.25"})
    assert dr2.observation_noise.anchor_rot_noise == pytest.approx(0.025)
    assert dr2.observation_noise.dof_pos_noise == 0.02  # untouched


def test_per_axis_lists_are_scaled_elementwise():
    dr = DomainRandomizationConfig(
        observation_noise=RobotNoiseConfig(root_pos_noise=[0.01, 0.02, 0.04])
    )
    _collect(dr, {OBS_NOISE_SCALE_VAR: "0.5"})
    assert dr.observation_noise.root_pos_noise == [0.005, 0.01, 0.02]


def test_zero_action_scale_removes_the_block_not_a_degenerate_range():
    """0 must not leave (0.0, 0.0): __post_init__ rejects low >= high."""
    dr = _v57_dr()
    changed, lines = _collect(dr, {ACTION_NOISE_SCALE_VAR: "0"})
    assert changed is True
    assert dr.action_noise is None
    assert any("REMOVED" in ln for ln in lines)
    # The degenerate range really would be rejected on any rebuild:
    with pytest.raises(ValueError):
        ActionNoiseDomainRandomizationConfig(
            action_noise_range=(0.0, 0.0), dof_names=[".*"]
        )


def test_zero_obs_scale_turns_has_noise_off_and_says_so():
    dr = _v57_dr()
    changed, lines = _collect(dr, {OBS_NOISE_SCALE_VAR: "0"})
    assert changed is True
    assert dr.observation_noise.has_noise() is False
    assert any("has_noise() is now" in ln for ln in lines)


def test_every_robot_noise_config_magnitude_field_is_covered():
    """OBS_NOISE_FIELDS must not silently fall behind RobotNoiseConfig."""
    import dataclasses

    declared = {f.name for f in dataclasses.fields(RobotNoiseConfig)}
    assert set(OBS_NOISE_FIELDS) <= declared
    missing = declared - set(OBS_NOISE_FIELDS)
    assert not missing, (
        "RobotNoiseConfig grew magnitude field(s) the NOISE-DR gate does not "
        f"scale: {sorted(missing)}. Add them to OBS_NOISE_FIELDS (or, if they "
        "are not magnitudes, say so here)."
    )


# =============================================================================
# 3. ABSENT TARGETS WARN LOUDLY (never a silent no-op)
# =============================================================================


def test_absent_domain_randomization_logs_a_loud_skip():
    changed, lines = _collect(None, {OBS_NOISE_SCALE_VAR: "0.25"})
    assert changed is False
    assert len(lines) == 1
    assert "SKIPPED" in lines[0] and "NO effect" in lines[0]
    assert OBS_NOISE_SCALE_VAR in lines[0]


def test_absent_action_noise_block_logs_a_loud_skip():
    dr = DomainRandomizationConfig(observation_noise=RobotNoiseConfig())
    changed, lines = _collect(dr, {ACTION_NOISE_SCALE_VAR: "0.2"})
    assert changed is False
    assert any(
        "SKIPPED for actions" in ln and ACTION_NOISE_SCALE_VAR in ln for ln in lines
    )


def test_absent_observation_noise_block_logs_a_loud_skip():
    dr = DomainRandomizationConfig(
        action_noise=ActionNoiseDomainRandomizationConfig(
            action_noise_range=(-0.05, 0.05), dof_names=[".*"]
        )
    )
    changed, lines = _collect(
        dr, {OBS_NOISE_SCALE_VAR: "0.25", ANCHOR_ROT_NOISE_SCALE_VAR: "0.5"}
    )
    assert changed is False
    assert any("SKIPPED for observations" in ln for ln in lines)


# =============================================================================
# 4. VALIDATION
# =============================================================================


@pytest.mark.parametrize("bad", ["-1", "-0.001", "nan", "inf", "abc", ""])
@pytest.mark.parametrize("var", NOISE_SCALE_ENV_VARS)
def test_invalid_scales_are_hard_errors(var, bad):
    dr = _v57_dr()
    with pytest.raises(ValueError, match=re.escape(var)):
        apply_noise_scale_env_overrides(
            dr, log_fn=lambda _: None, label="TEST", env={var: bad}
        )


# =============================================================================
# 5. BOTH WIRING PATHS CALL THE SAME IMPLEMENTATION
# =============================================================================


def test_fresh_build_and_resume_share_one_implementation():
    """teacher.py (fresh build) and train_agent.py (resume) must both import
    ``apply_noise_scale_env_overrides`` from this module, so they can never
    drift -- the same law the GAIN-DR gate is held to."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    train_agent = (root / "protomotions" / "train_agent.py").read_text()
    assert "noise_scale_env_gates import" in train_agent
    assert "apply_noise_scale_env_overrides" in train_agent
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
        assert "noise_scale_env_gates import" in text
        assert "apply_noise_scale_env_overrides" in text
        assert 'label="FRESH-BUILD"' in text
