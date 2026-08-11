# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guards for the masked-mimic conditioning-spec proof and validator.

These pin the two things that make a conditioning spec silently wrong:
the bit semantics of ``constraint_state``, and the fact that a *fixed* spec is
still gutted by ``visible_target_pose_prob < 1``.
"""

import pytest

from protomotions.envs.control.masked_mimic_control import (
    FixedBodyCondition,
    MaskedMimicControlConfig,
)
from protomotions.envs.control.mm_conditioning_proof import (
    CONSTRAINT_STATE_BITS,
    constraint_state_bits,
    format_conditioning_proof,
    validate_conditioning_spec,
)

CONDITIONABLE = [
    "torso_link",
    "head_aux",
    "right_ankle_roll_link",
    "left_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

TELEOP5 = [
    ("left_ankle_roll_link", 1),
    ("right_ankle_roll_link", 1),
    ("left_wrist_yaw_link", 1),
    ("right_wrist_yaw_link", 1),
    ("torso_link", 1),
]


def _cfg(**kw):
    return MaskedMimicControlConfig(**kw)


def _fixed(spec):
    return [FixedBodyCondition(body_name=n, constraint_state=s) for n, s in spec]


def test_constraint_state_bits_match_the_control_implementation():
    """``translation = state <= 1``, ``rotation = state >= 1``.

    If this table and ``MaskedMimicControl._sample_body_masks`` ever disagree,
    every proof line printed at boot becomes a lie about what the student sees.
    """
    for state in (0, 1, 2):
        pos, rot = constraint_state_bits(state)
        assert pos == (state <= 1)
        assert rot == (state >= 1)
    assert CONSTRAINT_STATE_BITS[1] == (True, True), "1 must be pos AND rot"


def test_unknown_constraint_state_raises():
    with pytest.raises(ValueError):
        constraint_state_bits(3)


def test_validate_rejects_a_non_conditionable_body():
    """Otherwise it dies as ``'x' is not in list`` inside the first env reset."""
    cfg = _cfg(fixed_conditioning=_fixed([("left_elbow_link", 1)]))
    with pytest.raises(ValueError, match="not a CONDITIONABLE body"):
        validate_conditioning_spec(cfg, CONDITIONABLE)


def test_validate_rejects_a_duplicated_body():
    cfg = _cfg(fixed_conditioning=_fixed([("torso_link", 1), ("torso_link", 2)]))
    with pytest.raises(ValueError, match="twice"):
        validate_conditioning_spec(cfg, CONDITIONABLE)


def test_validate_accepts_the_teleop5_deployment_spec():
    cfg = _cfg(fixed_conditioning=_fixed(TELEOP5), visible_target_pose_prob=1.0)
    validate_conditioning_spec(cfg, CONDITIONABLE)  # must not raise


def test_validate_is_a_noop_for_the_random_sampler():
    validate_conditioning_spec(_cfg(fixed_conditioning=None), CONDITIONABLE)


def test_proof_is_emitted_even_for_the_stock_random_sampler():
    """A MISSING block must mean a stale binary, never 'the spec is stock'."""
    lines = format_conditioning_proof(_cfg(), CONDITIONABLE, "FRESH-BUILD")
    assert lines
    assert any("RANDOM SUBSET SAMPLER" in line for line in lines)
    assert all(line.startswith("[MM-COND] FRESH-BUILD") for line in lines)


def test_proof_warns_when_a_fixed_spec_is_paired_with_pose_dropout():
    """THE TRAP: fixed_conditioning does NOT disable visible_target_pose_prob.

    ``_shift_and_sample_body_masks`` blanks the whole step at rate
    ``1 - visible_target_pose_prob`` AFTER the fixed mask is applied, so the
    stock 0.8 throws away ~20% of a 'fixed' spec's frames.
    """
    cfg = _cfg(fixed_conditioning=_fixed(TELEOP5), visible_target_pose_prob=0.8)
    lines = format_conditioning_proof(cfg, CONDITIONABLE, "RESUME")
    assert any("WARNING" in line for line in lines)
    assert any("20.0% of steps have the ENTIRE conditioning blanked" in line for line in lines)


def test_proof_does_not_warn_at_full_visibility():
    cfg = _cfg(fixed_conditioning=_fixed(TELEOP5), visible_target_pose_prob=1.0)
    lines = format_conditioning_proof(cfg, CONDITIONABLE, "RESUME")
    assert not any("WARNING" in line for line in lines)
    assert any("0.0% of steps have the ENTIRE conditioning blanked" in line for line in lines)


def test_proof_names_every_conditioned_body_and_every_masked_off_body():
    """The block must be readable as the deployment spec, without the source."""
    cfg = _cfg(fixed_conditioning=_fixed(TELEOP5), visible_target_pose_prob=1.0)
    text = "\n".join(format_conditioning_proof(cfg, CONDITIONABLE, "RESUME"))
    for body, _ in TELEOP5:
        assert body in text
    # head_aux is the one conditionable body teleop5 excludes; it must be
    # reported as masked off rather than simply absent.
    assert "MASKED OFF: ['head_aux']" in text
    assert "pos=True rot=True" in text


def test_proof_marks_sampler_knobs_inert_under_a_fixed_spec():
    """repeat_mask_probability et al. are dead once fixed_conditioning is set."""
    cfg = _cfg(fixed_conditioning=_fixed(TELEOP5), visible_target_pose_prob=1.0)
    text = "\n".join(format_conditioning_proof(cfg, CONDITIONABLE, "RESUME"))
    assert "repeat_mask_probability" in text
    assert "INERT under a fixed spec" in text
