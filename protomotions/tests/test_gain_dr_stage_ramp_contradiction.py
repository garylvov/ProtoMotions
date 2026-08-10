# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gain-DR stage ramp is dead code, and this pins that fact in place.

FOUND 2026-08-10, live on v63 and v64. Three things are individually
defensible and jointly contradictory:

1. ``stages_night13.py::_GAIN_RANGE_BY_STAGE`` defines a gain-DR curriculum:
   stage 0 ``(0.9, 1.1)``, stage 1 ``(0.8, 1.2)``, stage 2 ``(0.7, 1.3)``.
2. ``gain_dr_env_gates.apply_gain_dr_env_overrides`` fires only when an env var
   is *explicitly present* -- a guard written so an unset environment leaves a
   resumed config byte-identical. Sound in isolation.
3. ``launch_protomotions_ddp.sh`` exports ``PM_GAIN_DR_LOW=0.7`` and
   ``PM_GAIN_DR_HIGH=1.3`` **unconditionally**.

An unconditional export means the var is ALWAYS present, so the gate ALWAYS
fires, so the stage table is ALWAYS overwritten with ``(0.7, 1.3)``. Every
stage runs at stage-2 width from epoch 1. The curriculum never happens.

THIS TEST DOES NOT TAKE A SIDE. It asserts the contradiction is ACKNOWLEDGED,
not that either half is correct -- because the fix is an owner decision (make
the export conditional / delete the ramp / teach the gate to tell an operator
override from a launcher default) and v63/v64 are training off this lineage
right now. What the test prevents is the contradiction being quietly resolved
in either direction, or quietly deepened, by someone who does not know it is
there. Any such edit fails here and is pointed at the entry that explains it:

    imprint docs/curric-dawn/DAWN2.md,
    "ACTIVE CONFIGURATION DEFECTS" ->
    "CONTRADICTION: the gain-DR stage ramp is dead code, voided by an
     unconditional launcher export"

Sibling instance of the same class, same day: "READER/WRITER-LAW VIOLATION:
``limits_dof_pos`` contradicts the tracking terms on ``deep_hinge_crouch``" --
a law that another law silently voids, each defensible alone, nothing checking
the pair.
"""

from pathlib import Path

import pytest

from protomotions.simulator.base_simulator.config import (
    ActuatorGainDomainRandomizationConfig,
)
from protomotions.simulator.base_simulator.gain_dr_env_gates import (
    apply_gain_dr_env_overrides,
)

#: The curriculum the launcher voids. Kept here so a change to the table is a
#: change to this test, not a silent divergence.
STAGE_RAMP = {0: (0.9, 1.1), 1: (0.8, 1.2), 2: (0.7, 1.3)}

#: What the launcher pins instead, on every stage.
PINNED_BAND = (0.7, 1.3)

DOC = (
    "imprint docs/curric-dawn/DAWN2.md, ACTIVE CONFIGURATION DEFECTS -> "
    "'CONTRADICTION: the gain-DR stage ramp is dead code, voided by an "
    "unconditional launcher export'"
)


def _outside_file(*parts):
    """Locate a file in the run tree / imprint repo that vendors this package.

    Mirrors ``test_launcher_no_scratch._launcher``. Skips rather than fails in
    a bare ProtoMotions clone, where the imprint-side files are simply absent.
    """
    here = Path(__file__).resolve()
    for depth in (2, 3, 4, 5):
        candidate = here.parents[depth].joinpath(*parts)
        if candidate.is_file():
            return candidate
    pytest.skip(
        f"{'/'.join(parts)} not found next to this checkout (bare ProtoMotions "
        "clone); this guard runs where the imprint tree actually lives."
    )


def _live_lines(text):
    """Source lines with whole-line and trailing comments stripped."""
    out = []
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        out.append(line.split("#", 1)[0] if "#" in line else line)
    return out


def test_the_gate_really_does_overwrite_the_stage_table(monkeypatch):
    """Behavioral proof, not a source grep: the override wins over the ramp.

    Builds a config at the STAGE 0 band, sets only the two vars the launcher
    exports unconditionally, and shows stage 0 comes out at the stage-2 band.
    This is what happens on epoch 1 of every launcher-driven run.
    """
    monkeypatch.setenv("PM_GAIN_DR_LOW", str(PINNED_BAND[0]))
    monkeypatch.setenv("PM_GAIN_DR_HIGH", str(PINNED_BAND[1]))

    cfg = ActuatorGainDomainRandomizationConfig(
        dof_names=[".*"],
        stiffness_scale_range=STAGE_RAMP[0],
        damping_scale_range=STAGE_RAMP[0],
    )
    apply_gain_dr_env_overrides(cfg, log_fn=lambda *a, **k: None, label="TEST")

    assert tuple(cfg.stiffness_scale_range) == PINNED_BAND, (
        "the stage-0 stiffness band survived the launcher's unconditional "
        f"export; the contradiction described in {DOC} no longer holds and "
        "that entry needs updating"
    )
    assert tuple(cfg.damping_scale_range) == PINNED_BAND, (
        "stage-0 damping band survived the override; see " + DOC
    )


def test_the_gate_is_still_presence_triggered(monkeypatch):
    """With no var present the stage table must survive untouched.

    This is the half of the design that is CORRECT and load-bearing (resume
    byte-identity). It is asserted so a future 'fix' to the contradiction
    cannot be to make the gate fire unconditionally, which would break every
    resume instead.
    """
    for var in ("PM_GAIN_DR_LOW", "PM_GAIN_DR_HIGH"):
        monkeypatch.delenv(var, raising=False)

    cfg = ActuatorGainDomainRandomizationConfig(
        dof_names=[".*"],
        stiffness_scale_range=STAGE_RAMP[0],
        damping_scale_range=STAGE_RAMP[0],
    )
    apply_gain_dr_env_overrides(cfg, log_fn=lambda *a, **k: None, label="TEST")

    assert tuple(cfg.stiffness_scale_range) == STAGE_RAMP[0], (
        "the gain-DR gate mutated a config with NO env var present. That "
        "breaks resume byte-identity (Rule 10). See " + DOC
    )


def test_the_launcher_export_is_still_unconditional():
    """The export that voids the ramp, and the comment that admits it.

    Two assertions, deliberately paired: the defect must still be there AND
    still be labelled. Removing the label without removing the defect is the
    failure mode this catches -- an undocumented contradiction is how this one
    survived six teacher generations.
    """
    launcher = _outside_file("launch_protomotions_ddp.sh")
    text = launcher.read_text()

    joined = "\n".join(_live_lines(text))
    for var, value in (("PM_GAIN_DR_LOW", "0.7"), ("PM_GAIN_DR_HIGH", "1.3")):
        expected = f'export {var}="${{{var}:-{value}}}"'
        assert expected in joined, (
            f"{launcher.name} no longer exports {var} unconditionally. If this "
            "was deliberate, the gain-DR stage ramp is LIVE again for the "
            f"first time in this campaign -- report whether it changes "
            f"anything, and update {DOC}"
        )

    assert "override the stage table" in text or "overriding the stage" in text, (
        f"{launcher.name} exports the gain-DR band unconditionally but no "
        "longer says that this voids the stage ramp. The comment is the only "
        "thing standing between the next reader and the belief that a gain "
        f"curriculum is running. See {DOC}"
    )


def test_the_stage_ramp_still_exists_to_be_voided():
    """If the ramp is deleted, the contradiction is resolved -- say so here.

    Deleting ``_GAIN_RANGE_BY_STAGE`` is a LEGITIMATE fix (option 2 in the
    DAWN2 entry): the flat band becomes the honest, stated configuration. It
    just must not happen silently, because this test, that entry and the
    launcher comment would all still claim a ramp exists.
    """
    stages = _outside_file(
        "src", "imprint", "integrations", "wbc", "training", "stages_night13.py"
    )
    text = stages.read_text()

    # The ASSIGNMENT, not merely the name -- the name also appears in prose and
    # at the use site, so a bare substring check passes even after the table is
    # renamed out of existence. (It did, on the first draft of this test.)
    assert "_GAIN_RANGE_BY_STAGE = {" in text, (
        "the gain-DR stage table is gone. If that was the deliberate fix, "
        f"remove this test and close the entry in {DOC}"
    )
    # ...and that it is still WIRED. A table nothing reads is a third way for
    # the ramp to die silently, distinct from deleting it or overriding it.
    assert "stiffness_scale_range=_GAIN_RANGE_BY_STAGE[stage]" in text, (
        "the gain-DR stage table is no longer wired into the stage config. "
        f"See {DOC}"
    )
    for stage, band in sorted(STAGE_RAMP.items()):
        assert f"{stage}: ({band[0]}, {band[1]})" in text, (
            f"stage {stage} band is no longer {band}. The table is currently "
            "DEAD CODE (the launcher overrides it), so this edit changed "
            f"nothing about training -- which is exactly why it needs to be "
            f"deliberate. See {DOC}"
        )


def test_no_per_group_or_effort_knob_has_ever_been_turned_on():
    """The per-group and effort axes are unexported, on every run.

    Reported up the chain once as 'legs 0.7-1.3, arms/waist 0.9-1.1'. That
    configuration has never existed: the per-group knobs appear in the launcher
    only inside comments. This pins the fact so the claim cannot be revived by
    reading the launcher's prose as if it were its behavior.
    """
    launcher = _outside_file("launch_protomotions_ddp.sh")
    never_set = (
        "PM_GAIN_DR_LOW_LEGS",
        "PM_GAIN_DR_HIGH_LEGS",
        "PM_GAIN_DR_LOW_WAIST",
        "PM_GAIN_DR_HIGH_WAIST",
        "PM_GAIN_DR_LOW_ARMS",
        "PM_GAIN_DR_HIGH_ARMS",
        "PM_GAIN_DR_KD_",
        "PM_EFFORT_DR_",
        "PM_GAIN_DR_CONSTANT_ZETA",
        "PM_GAIN_DR_ENV_SCALE_SOURCE",
    )
    for lineno, line in enumerate(_live_lines(launcher.read_text()), 1):
        if not line.strip().startswith("export "):
            continue
        for knob in never_set:
            assert knob not in line, (
                f"{launcher.name}:{lineno} exports {knob}. No per-group or "
                "effort-limit gain knob has ever been set on any run in this "
                "campaign; turning one on changes the plant and invalidates "
                f"every published gain figure. See {DOC}"
            )
