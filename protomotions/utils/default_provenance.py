# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boot-time report: which config came from an OPERATOR, which from a DEFAULT.

WHY THIS EXISTS (2026-08-10). `os.environ.get("PM_GAIN_DR_LOW")` returns
`"0.7"` whether a human typed it or the launcher defaulted it. Every env gate
in this codebase fires on PRESENCE, because presence is the only signal
available. So when `launch_protomotions_ddp.sh` began exporting
`PM_GAIN_DR_LOW/HIGH` unconditionally, the gain-DR gate started firing on every
run and overwriting `_GAIN_RANGE_BY_STAGE` -- and the three-stage gain
curriculum silently became dead code for six teacher generations. Nobody was
wrong at any single site: the table is reasonable, the presence-gate is a
correct resume guard, the export is a reasonable pin. Nothing compared them.

This is the CHEAP HALF of the fix. It does not teach the gates anything -- that
is the ~150-250 line cross-cutting change sized in DAWN2.md and deliberately
not built. It just makes the launcher publish what it defaulted
(`PM_LAUNCHER_DEFAULTED`) and prints, at boot, how much of the config is
default-provenance and which of those defaults SHADOW a documented curriculum.
That one line would have surfaced the gain-DR contradiction on the first v55
boot.

SHADOW REGISTRY. A var is only interesting here if overriding it can void
something a human would otherwise believe is running. Those are listed
explicitly rather than inferred: a guessed list would be both incomplete and
untrustworthy, and this output is only worth printing if it can be believed at
3am.

RULE 10: read-only. Reads `os.environ`, returns strings. Touches no config and
no tensor; a run with the var unset prints one line and is otherwise identical.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

#: Vars whose LAUNCHER DEFAULT silently overrides something a reader would
#: otherwise believe is in force -- a stage table, a curriculum, or an
#: opt-in-by-design kernel default. value = what it overrides, phrased so the
#: boot line explains itself without a second lookup.
SHADOWS: Dict[str, str] = {
    "PM_GAIN_DR_LOW": "stages_night13._GAIN_RANGE_BY_STAGE (0.9-1.1/0.8-1.2/0.7-1.3 gain-DR ramp)",
    "PM_GAIN_DR_HIGH": "stages_night13._GAIN_RANGE_BY_STAGE (0.9-1.1/0.8-1.2/0.7-1.3 gain-DR ramp)",
    # Third instance of the same shape, found 2026-08-10 while building this.
    # compute_soft_pos_limit_rew defaults soft_margin_frac=0.0 and its docstring
    # says the proximity term "only activates via explicit env override
    # (PM_DOF_LIMIT_MARGIN)". The launcher exports that var UNCONDITIONALLY at
    # 0.05, so the override is never explicit and the documented off-by-default
    # is never what runs.
    "PM_DOF_LIMIT_MARGIN": (
        "compute_soft_pos_limit_rew soft_margin_frac=0.0 default "
        "(docstring: proximity term activates only via EXPLICIT override)"
    ),
}
# ONLY vars the launcher actually EXPORTS belong here. A registry entry for a
# var nobody sets is a warning that can never fire, which is worse than no
# warning -- it reads as coverage. `test_default_provenance` asserts
# SHADOWS is a subset of the launcher's exports, and it caught this list
# claiming the NOISE-DR knobs (PM_ACTION_NOISE_SCALE / PM_OBS_NOISE_SCALE /
# PM_ANCHOR_ROT_NOISE_SCALE) on the first run: those shadow stages_night13's
# pinned noise magnitudes and WOULD belong here, but the launcher only
# documents them in comments and never exports them, so they can never be
# launcher-defaulted. Add them the day the launcher starts exporting them.

ENV_VAR = "PM_LAUNCHER_DEFAULTED"
TAG = "[provenance]"


def split_provenance(
    environ: Optional[Dict[str, str]] = None,
) -> tuple[List[str], List[str]]:
    """Return (defaulted, operator_set) PM_* var names, both sorted.

    `defaulted` is what the launcher published; `operator_set` is every other
    PM_* var present in the environment. With `PM_LAUNCHER_DEFAULTED` absent
    (a hand-rolled launch, or an older launcher) everything reads as
    operator-set, which is the honest answer: nothing claimed otherwise.
    """
    env = os.environ if environ is None else environ
    declared = [v for v in (env.get(ENV_VAR) or "").split() if v]
    defaulted = sorted(set(declared))
    present = {k for k in env if k.startswith("PM_")} - {ENV_VAR}
    operator = sorted(present - set(defaulted))
    return defaulted, operator


def format_provenance(
    defaulted: Sequence[str], operator: Sequence[str], published: bool
) -> List[str]:
    """The boot lines. Designed to be read at 3am: verdict first, list second."""
    if not published:
        return [
            f"{TAG} {ENV_VAR} not published by the launcher -- cannot tell "
            f"operator choices from script defaults. {len(operator)} PM_* vars "
            f"present, all treated as operator-set."
        ]
    total = len(defaulted) + len(operator)
    lines = [
        f"{TAG} {total} PM_* config values live: {len(defaulted)} from LAUNCHER "
        f"DEFAULTS, {len(operator)} chosen by the operator."
    ]
    shadowing = [v for v in defaulted if v in SHADOWS]
    if shadowing:
        lines.append(
            f"{TAG} WARNING: {len(shadowing)} launcher default(s) SHADOW a "
            f"documented curriculum -- the curriculum below is NOT running:"
        )
        for var in shadowing:
            lines.append(f"{TAG}   {var}={os.environ.get(var, '?')} shadows {SHADOWS[var]}")
        lines.append(
            f"{TAG}   These fire because the gate cannot tell a launcher default "
            f"from an operator choice. See DAWN2.md 'PATTERN: silent config "
            f"contradictions'."
        )
    if operator:
        lines.append(f"{TAG} operator-set: {' '.join(operator)}")
    return lines


def report(log_fn=print, environ: Optional[Dict[str, str]] = None) -> List[str]:
    """Emit the provenance report. Never raises into boot."""
    try:
        env = os.environ if environ is None else environ
        defaulted, operator = split_provenance(env)
        lines = format_provenance(defaulted, operator, published=ENV_VAR in env)
        for line in lines:
            log_fn(line)
        return lines
    except Exception as exc:
        log_fn(f"{TAG} report failed ({type(exc).__name__}: {exc}); skipped.")
        return []
