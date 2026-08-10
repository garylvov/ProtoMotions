# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard for the launcher default-provenance report.

The report exists because no gate can distinguish "an operator set this" from
"the launcher defaulted it" -- `os.environ.get` returns the same string either
way -- which is how the gain-DR stage ramp became dead code for six teacher
generations. These tests pin the case it was built to catch.
"""

from pathlib import Path

import pytest

from protomotions.utils.default_provenance import (
    ENV_VAR,
    SHADOWS,
    format_provenance,
    split_provenance,
)


def test_splits_operator_choices_from_launcher_defaults():
    env = {
        ENV_VAR: "PM_GAIN_DR_LOW PM_GAIN_DR_HIGH",
        "PM_GAIN_DR_LOW": "0.7",
        "PM_GAIN_DR_HIGH": "1.3",
        "PM_BODY_WEIGHTS": "pelvis=0.0",  # operator typed this one
        "HOME": "/root",  # non-PM_ vars are irrelevant
    }
    defaulted, operator = split_provenance(env)
    assert defaulted == ["PM_GAIN_DR_HIGH", "PM_GAIN_DR_LOW"]
    assert operator == ["PM_BODY_WEIGHTS"]


def test_the_gain_dr_case_produces_a_shadow_warning(monkeypatch):
    """The exact contradiction that motivated this, on the exact vars."""
    monkeypatch.setenv("PM_GAIN_DR_LOW", "0.7")
    monkeypatch.setenv("PM_GAIN_DR_HIGH", "1.3")
    lines = format_provenance(
        defaulted=["PM_GAIN_DR_HIGH", "PM_GAIN_DR_LOW"],
        operator=[],
        published=True,
    )
    joined = "\n".join(lines)
    assert "SHADOW" in joined
    assert "_GAIN_RANGE_BY_STAGE" in joined
    assert "NOT running" in joined
    # the verdict must be on the FIRST line; a warning buried under a var dump
    # is a warning nobody reads
    assert "from LAUNCHER DEFAULTS" in lines[0]


def test_operator_set_shadowing_var_is_not_warned_about():
    """If a human typed PM_GAIN_DR_LOW, overriding the ramp was their choice.

    This is the whole distinction the report exists to draw. Warning here would
    train readers to ignore the warning.
    """
    lines = format_provenance(
        defaulted=[], operator=["PM_GAIN_DR_LOW"], published=True
    )
    joined = "\n".join(lines)
    assert "SHADOW" not in joined
    assert "PM_GAIN_DR_LOW" in joined  # still listed, just not flagged


def test_unpublished_is_reported_as_unknown_not_as_all_operator():
    """An older launcher must not silently read as 'operator chose everything'.

    Claiming operator provenance we cannot support is worse than admitting the
    gap -- that false confidence is the failure this whole entry is about.
    """
    lines = format_provenance(defaulted=[], operator=["PM_X"], published=False)
    joined = "\n".join(lines)
    assert "cannot tell" in joined
    assert "SHADOW" not in joined


def test_clean_run_says_so_in_one_line():
    lines = format_provenance(defaulted=["PM_STACK"], operator=[], published=True)
    assert len(lines) == 1
    assert "1 from LAUNCHER DEFAULTS" in lines[0]


def _launcher():
    here = Path(__file__).resolve()
    for depth in (2, 3, 4, 5):
        c = here.parents[depth] / "launch_protomotions_ddp.sh"
        if c.is_file():
            return c
    pytest.skip("launch_protomotions_ddp.sh not found next to this checkout")


def test_launcher_publishes_the_provenance_variable():
    """The report is worthless if the launcher stops publishing the list.

    Pins the mechanism, not just the name: the snapshot must be taken BEFORE
    the exports and the emit AFTER, or every var reads as operator-set.
    """
    text = _launcher().read_text()
    assert f"export {ENV_VAR}=" in text, "launcher no longer publishes " + ENV_VAR

    snapshot = text.index("_PM_PRESET=")
    emit = text.index(f"export {ENV_VAR}=")
    first_export = text.index("\nexport PM_")
    assert snapshot < first_export, (
        "the provenance snapshot must run BEFORE the first PM_ export, or "
        "launcher defaults are indistinguishable from operator choices -- "
        "which is the exact defect this reports on"
    )
    assert emit > first_export, "the emit must run AFTER the exports"


def test_every_shadowed_var_is_actually_exported_by_the_launcher():
    """A shadow registry that names a var nobody sets is decoration.

    Catches the registry drifting out of sync with the launcher -- e.g. a knob
    renamed, leaving a warning that can never fire.
    """
    text = _launcher().read_text()
    live = [
        ln.split("=", 1)[0].removeprefix("export ").strip()
        for ln in text.splitlines()
        if ln.startswith("export PM_")
    ]
    missing = [v for v in SHADOWS if v not in live]
    assert not missing, (
        f"SHADOWS names {missing}, which the launcher does not export. Either "
        "the launcher changed or the registry is stale; a warning that cannot "
        "fire is worse than none."
    )


def test_the_dof_limit_margin_instance_is_registered():
    """Third instance of the pattern, found while building this report.

    `compute_soft_pos_limit_rew` defaults `soft_margin_frac=0.0` and documents
    the proximity term as activating "only via explicit env override". The
    launcher exports PM_DOF_LIMIT_MARGIN unconditionally at 0.05, so it is
    never explicit and the documented default never runs. Registered so a plain
    boot says so.
    """
    assert "PM_DOF_LIMIT_MARGIN" in SHADOWS
    lines = format_provenance(
        defaulted=["PM_DOF_LIMIT_MARGIN"], operator=[], published=True
    )
    joined = "\n".join(lines)
    assert "SHADOW" in joined and "soft_margin_frac" in joined
