# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boot-time PROOF (and validation) for the masked-mimic CONDITIONING SPEC.

Why this exists
---------------
The conditioning spec is the student's *deployment assumption*: which bodies it
is shown, whether it is shown their position, their orientation, or both, and
how often it is shown anything at all.  Train it under one spec and deploy it
under another and you get exactly the failure the DOF-profile run surfaced --
significantly better at ``sparse6`` (near the density it trains under) and flat
at ``sparse3`` (the teleop-shaped headline).  That is a conditioning mismatch,
not a gradient-allocation problem, and it is invisible in every loss curve.

Two mechanisms make the spec easy to get silently wrong:

1. **``train_agent.py::detect_checkpoint_mode`` does not re-execute the
   experiment file on a resume.**  ``results/<EXP>/config.yaml`` is written back
   onto ``args`` and ``resolved_configs.pt`` supplies the config objects.  So on
   a resume the recipe's ``env_config()`` never runs: the ``MM_FIXED_COND`` env
   gate in ``masked_mimic_stiffv2.py``, ``--overrides``, and every CLI flag are
   INERT.  The launcher can echo an intended conditioning set that the run never
   uses.  Rewriting the frozen pickle (``stage_resume_config.py
   --fixed-conditioning``) is the only way to change it, and reading it back off
   the object the env will actually be built with is the only way to know.

2. **``fixed_conditioning`` does NOT disable ``visible_target_pose_prob``.**
   ``MaskedMimicControl._shift_and_sample_body_masks`` applies the fixed set in
   ``_sample_body_masks``, and *then*, unconditionally, blanks the entire step
   with probability ``1 - visible_target_pose_prob``.  At the stock 0.8 that
   silently throws away ~20% of the conditioning frames of a spec whose whole
   point is that it is fixed.  ``test_control_components.py`` pairs
   ``fixed_conditioning`` with ``visible_target_pose_prob=1.0`` -- that is the
   intended pairing, and this module makes a violation of it loud.

The proof line is emitted on EVERY boot, including for the plain random sampler.
"No line in the log" must mean "this binary predates the proof", never "the spec
is stock" -- an absent line that could mean either is not a proof.
"""

from __future__ import annotations

from typing import Callable, List, Optional

#: ``constraint_state`` -> (position observed, rotation observed).
#: Mirrors ``MaskedMimicControl._sample_body_masks``:
#:     translation_mask = (constraint_states <= 1) & active_body_ids
#:     rotation_mask    = (constraint_states >= 1) & active_body_ids
CONSTRAINT_STATE_BITS = {
    0: (True, False),   # translation only
    1: (True, True),    # position AND orientation
    2: (False, True),   # rotation only
}

CONSTRAINT_STATE_NAMES = {
    0: "pos-only",
    1: "pos+rot",
    2: "rot-only",
}

#: Sampler knobs that become DEAD when ``fixed_conditioning`` is set. Printed as
#: INERT so nobody tunes one and waits for an effect that cannot arrive.
SAMPLER_ONLY_FIELDS = (
    "repeat_mask_probability",
    "force_max_conditioned_bodies_prob",
    "force_small_num_conditioned_bodies_prob",
)


def constraint_state_bits(state: int):
    """(pos_observed, rot_observed) for a ``constraint_state``. Raises if unknown."""
    if state not in CONSTRAINT_STATE_BITS:
        raise ValueError(
            f"unknown constraint_state {state!r}; expected 0 (translation only), "
            "1 (position AND orientation) or 2 (rotation only)"
        )
    return CONSTRAINT_STATE_BITS[state]


def validate_conditioning_spec(control_config, conditionable_bodies: List[str]) -> None:
    """Fail at BOOT on a conditioning spec the env would only break on later.

    ``MaskedMimicControl._sample_body_masks`` resolves fixed body names with
    ``self._all_body_names.index(name)`` and then indexes
    ``conditionable_body_ids``.  A name that is not a body, or is a body but is
    not *conditionable*, therefore dies with a bare
    ``ValueError: 'x' is not in list`` from inside the first env reset -- after
    the allocation is taken, the sim is up and the checkpoint is loaded.  Catch
    it here instead, while the message can still say what is wrong.
    """
    fixed = getattr(control_config, "fixed_conditioning", None)
    if not fixed:
        return

    seen = set()
    for entry in fixed:
        name = entry.body_name
        constraint_state_bits(entry.constraint_state)  # raises on a bad state
        if name not in conditionable_bodies:
            raise ValueError(
                f"fixed_conditioning names {name!r}, which is not a CONDITIONABLE "
                f"body. Conditionable bodies come from the robot config's "
                f"trackable_bodies_subset and are: {conditionable_bodies}. "
                "Fix the spec, or add the body to trackable_bodies_subset -- but "
                "note that changing that subset changes the observation width and "
                "therefore INVALIDATES any resume."
            )
        if name in seen:
            raise ValueError(
                f"fixed_conditioning lists {name!r} twice; the second entry would "
                "silently overwrite the first's constraint_state."
            )
        seen.add(name)


def format_conditioning_proof(
    control_config,
    conditionable_bodies: Optional[List[str]],
    label: str,
) -> list:
    """Render the boot-time proof lines. Pure, for testing."""
    fixed = getattr(control_config, "fixed_conditioning", None)
    visible = getattr(control_config, "visible_target_pose_prob", None)
    bodies = list(conditionable_bodies or [])

    lines = [
        f"[MM-COND] {label}: conditionable bodies ({len(bodies)}) = {bodies}"
    ]

    if not fixed:
        lines.append(
            f"[MM-COND] {label}: fixed_conditioning = None -> RANDOM SUBSET "
            "SAMPLER (the stock training distribution)"
        )
        for field in SAMPLER_ONLY_FIELDS:
            lines.append(
                f"[MM-COND] {label}:   {field:38s} = "
                f"{getattr(control_config, field, None)}"
            )
    else:
        lines.append(
            f"[MM-COND] {label}: fixed_conditioning = FIXED SPEC on "
            f"{len(fixed)}/{len(bodies)} conditionable bodies. Every other body "
            "is masked off on EVERY step; the random subset sampler is BYPASSED."
        )
        for entry in fixed:
            pos, rot = constraint_state_bits(entry.constraint_state)
            lines.append(
                f"[MM-COND] {label}:   {entry.body_name:26s} "
                f"constraint_state={entry.constraint_state} "
                f"({CONSTRAINT_STATE_NAMES[entry.constraint_state]}) "
                f"pos={pos} rot={rot}"
            )
        hidden = [b for b in bodies if b not in {e.body_name for e in fixed}]
        lines.append(f"[MM-COND] {label}:   MASKED OFF: {hidden or 'none'}")
        for field in SAMPLER_ONLY_FIELDS:
            lines.append(
                f"[MM-COND] {label}:   {field:38s} = "
                f"{getattr(control_config, field, None)}  (INERT under a fixed spec)"
            )

    # --- the trap ----------------------------------------------------------
    if visible is None:
        lines.append(
            f"[MM-COND] {label}: visible_target_pose_prob = <MISSING> -- config "
            "predates the field; cannot prove the full-hide rate."
        )
    else:
        hide = 1.0 - float(visible)
        lines.append(
            f"[MM-COND] {label}: visible_target_pose_prob = {float(visible):g} "
            f"-> {hide * 100:.1f}% of steps have the ENTIRE conditioning blanked "
            "(applied AFTER the mask is chosen, fixed or sampled)."
        )
        if fixed and hide > 0:
            lines.append(
                f"[MM-COND] {label}: *** WARNING: a FIXED conditioning spec is "
                f"combined with visible_target_pose_prob={float(visible):g}. "
                f"{hide * 100:.1f}% of the conditioning frames this 'fixed' spec "
                "promises are being thrown away. Set it to 1.0 "
                "(stage_resume_config.py --visible-target-pose-prob 1.0) unless "
                "the dropout is deliberate. ***"
            )

    lines.append(
        f"[MM-COND] {label}: SOURCE = the MaskedMimicControlConfig this run will "
        "build its env from (on a resume: the UNPICKLED resolved_configs.pt, "
        "which is the only thing that can be true -- the recipe, --overrides and "
        "every env var are inert there)."
    )
    return lines


def log_conditioning_proof(
    control_config,
    conditionable_bodies: Optional[List[str]],
    log_fn: Callable[[str], None],
    label: str,
) -> None:
    """Emit the boot-time proof. Never mutates the config."""
    for line in format_conditioning_proof(control_config, conditionable_bodies, label):
        log_fn(line)
