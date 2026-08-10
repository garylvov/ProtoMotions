# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard for the reference-vs-penalty-band consistency check.

The check exists because `limits_dof_pos` taxed a band the tracking terms
command the policy into, on `deep_hinge_crouch`, unnoticed for six teacher
generations. These tests use the REAL numbers from that finding so the guard
fails if the check ever stops detecting the case it was built for.
"""

import pytest

from protomotions.envs.utils.reference_penalty_overlap import (
    Overlap,
    find_reference_penalty_overlaps,
    format_report,
)

# The measured case: ankle_pitch range 1.421 rad, lower limit -0.897, and a
# reference minimum of -0.898 -- i.e. PAST the limit, because the retargeter
# hard-clipped deep_hinge_crouch and only that category.
ANKLE_LOWER = -0.897
ANKLE_UPPER = ANKLE_LOWER + 1.421
ANKLE_REF_MIN = -0.898


def _one_joint(ref_min, ref_max=0.0, margin=0.05, **kw):
    return find_reference_penalty_overlaps(
        dof_names=["left_ankle_pitch_joint"],
        ref_min=[ref_min],
        ref_max=[ref_max],
        limits_lower=[ANKLE_LOWER],
        limits_upper=[ANKLE_UPPER],
        soft_margin_frac=margin,
        **kw,
    )


def test_detects_the_deep_hinge_crouch_contradiction():
    """The case that cost six generations must be flagged, and flagged as pinned."""
    findings = _one_joint(ANKLE_REF_MIN)
    assert len(findings) == 1, findings
    f = findings[0]
    assert f.dof_name == "left_ankle_pitch_joint"
    assert f.side == "lower"
    assert f.at_or_past_limit, "reference is past the limit; must be flagged as pinned"
    # weight -10 * proximity_scale 0.1 * prox 1.0
    assert f.cost_per_step == pytest.approx(1.0)


def test_narrowing_the_band_does_not_help_a_pinned_reference():
    """prox == 1.0 at dist == 0 for ANY margin -- the whole point of the entry.

    A tenfold narrower band must produce the SAME cost for a reference sitting
    at the limit. If this ever changes, the kernel changed and the DAWN2 entry's
    central argument needs revisiting.
    """
    wide = _one_joint(ANKLE_REF_MIN, margin=0.05)[0]
    narrow = _one_joint(ANKLE_REF_MIN, margin=0.005)[0]
    assert wide.cost_per_step == narrow.cost_per_step == pytest.approx(1.0)
    assert wide.at_or_past_limit and narrow.at_or_past_limit


def test_narrowing_the_band_DOES_help_a_merely_near_reference():
    """The contrast case: a reference inside but not at the band gets cheaper.

    Without this, the test above would pass for a trivially broken checker that
    always returned the same cost.
    """
    near = ANKLE_LOWER + 0.03  # inside a 0.0710 band, outside a 0.0071 one
    wide = _one_joint(near, margin=0.05)
    narrow = _one_joint(near, margin=0.005)
    assert len(wide) == 1 and not wide[0].at_or_past_limit
    assert narrow == [], "a 0.005 margin should no longer reach this reference"
    assert 0.0 < wide[0].cost_per_step < 1.0


def test_clean_config_reports_nothing():
    """Zero false positives on a reference that stays well inside its limits."""
    assert _one_joint(ref_min=0.0, ref_max=0.1) == []


def test_margin_zero_disables_the_check_entirely():
    """soft_margin_frac == 0.0 means no proximity term, so no band, so no finding."""
    assert _one_joint(ANKLE_REF_MIN, margin=0.0) == []
    lines = format_report([], soft_margin_frac=0.0, n_dofs=27)
    assert any("proximity term OFF" in ln for ln in lines)


def test_fixed_joints_are_skipped_not_divided_by_zero():
    """upper == lower gives margin 0; the kernel masks these and so must we."""
    findings = find_reference_penalty_overlaps(
        dof_names=["welded"],
        ref_min=[0.0],
        ref_max=[0.0],
        limits_lower=[0.0],
        limits_upper=[0.0],
        soft_margin_frac=0.05,
    )
    assert findings == []


def test_upper_side_is_checked_too():
    """The original finding was a lower-limit case; the upper side must work."""
    findings = _one_joint(ref_min=0.0, ref_max=ANKLE_UPPER)
    assert len(findings) == 1 and findings[0].side == "upper"
    assert findings[0].at_or_past_limit


def test_mismatched_sequence_lengths_raise_rather_than_misalign():
    """Silently zipping mismatched per-DOF arrays would report the wrong joint.

    That is the failure mode of the metric this campaign already renamed a doc
    after ("the joint a metric is named after is not the joint that governs
    it"), so it must be loud.
    """
    with pytest.raises(ValueError, match="one entry per DOF"):
        find_reference_penalty_overlaps(
            dof_names=["a", "b"],
            ref_min=[0.0],
            ref_max=[0.0],
            limits_lower=[-1.0, -1.0],
            limits_upper=[1.0, 1.0],
            soft_margin_frac=0.05,
        )


def test_findings_are_sorted_worst_first():
    """Pinned joints must lead the report; a 3am reader gets the fix-me first."""
    findings = find_reference_penalty_overlaps(
        dof_names=["merely_near", "pinned"],
        ref_min=[ANKLE_LOWER + 0.03, ANKLE_REF_MIN],
        ref_max=[0.0, 0.0],
        limits_lower=[ANKLE_LOWER, ANKLE_LOWER],
        limits_upper=[ANKLE_UPPER, ANKLE_UPPER],
        soft_margin_frac=0.05,
    )
    assert [f.dof_name for f in findings] == ["pinned", "merely_near"]


def test_report_names_the_no_band_width_fixes_this_conclusion():
    """The report must carry the actionable half, not just the numbers.

    A boot line that says "3 joints overlap" sends the reader to narrow the
    band, which provably does nothing for a pinned joint. The line has to say
    so at the point of the alarm.
    """
    lines = format_report(_one_joint(ANKLE_REF_MIN), soft_margin_frac=0.05, n_dofs=27)
    joined = "\n".join(lines)
    assert "CONTRADICTION" in joined
    assert "NARROWING" in joined and "CANNOT fix" in joined
    assert "0.0" in joined  # names the only setting that does fix it


def test_clean_report_is_one_line():
    """The healthy path must not add noise to a boot log nobody reads twice."""
    lines = format_report([], soft_margin_frac=0.05, n_dofs=27)
    assert len(lines) == 1 and "OK:" in lines[0]
