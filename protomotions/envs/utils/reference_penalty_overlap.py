# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boot-time check: does the REFERENCE enter a band a penalty term taxes?

WHY THIS EXISTS (2026-08-10). `limits_dof_pos` (weight -10) taxes a band that
the tracking terms command the policy INTO. On `deep_hinge_crouch` the
reference `ankle_pitch` sits AT its -0.897 limit on 10.5% of (frame, ankle)
pairs -- the retargeter hard-clipped that category -- while `dof_pos_track`,
`relative_body_pos` and `foot_relative_body_pos` all command the policy to
reach it. Two laws in the same 28-term stack command opposite things about the
same joint. It survived unchanged across six teacher generations because
nothing in the stack compares a term's TARGET against another term's PENALISED
SUPPORT.

This module is that comparison. It is the (a) half of the pair named in
DAWN2.md "PATTERN: silent config contradictions"; the (b) half
(curriculum-value vs env-override precedence) is sized and deliberately not
built.

THE NON-OBVIOUS PART, and the reason a narrower band is not a fix:

    prox = ((margin - dist) / margin).clamp(0, 1)

equals **1.0 whenever dist == 0, for any margin > 0**. A joint exactly at its
limit costs `proximity_scale` -- i.e. `proximity_scale * |weight|` per step per
joint -- no matter how tight the band is. So an overlap where the reference
sits AT or PAST a limit is categorically worse than one merely inside the band:
only `soft_margin_frac = 0.0` removes it. The findings below carry that
distinction explicitly (`at_or_past_limit`) instead of reporting one blended
severity, because the two have different fixes.

RULE 10: log-only. This module reads config and reference statistics, returns
findings, and never touches a tensor that feeds training. A run with no overlap
prints one clean line; nothing else about the run changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class Overlap:
    """One joint whose reference trajectory enters a penalised band."""

    dof_name: str
    side: str  # "lower" | "upper"
    ref_extreme: float  # the reference value closest to (or past) the limit
    limit: float
    band_edge: float  # inner edge of the penalty band
    at_or_past_limit: bool  # prox == 1.0 regardless of band width
    penetration_frac: float  # 0 at band edge, 1 at the limit, >1 past it
    cost_per_step: float  # |weight| * proximity_scale * prox, at the extreme

    def describe(self) -> str:
        where = "AT/PAST LIMIT" if self.at_or_past_limit else "inside band"
        return (
            f"{self.dof_name} [{self.side}] {where}: reference reaches "
            f"{self.ref_extreme:+.4f}, limit {self.limit:+.4f}, band edge "
            f"{self.band_edge:+.4f} ({self.penetration_frac * 100:.0f}% into "
            f"the band) -> {self.cost_per_step:.4f}/step/joint"
        )


def find_reference_penalty_overlaps(
    dof_names: Sequence[str],
    ref_min: Sequence[float],
    ref_max: Sequence[float],
    limits_lower: Sequence[float],
    limits_upper: Sequence[float],
    soft_margin_frac: float,
    proximity_scale: float = 0.1,
    weight: float = -10.0,
) -> List[Overlap]:
    """Joints whose reference range enters `limits_dof_pos`'s penalty band.

    All sequences are per-DOF and must be the same length and order. Returns
    findings sorted worst-first (at/past-limit before merely-inside-band, then
    by penetration depth). An empty list means no contradiction.

    `soft_margin_frac <= 0` disables the proximity term entirely, so there is
    no band and nothing to report -- the base `out_of_limits` term still
    charges genuine violations, but that is the policy exceeding a limit, not
    the config commanding it to.
    """
    n = len(dof_names)
    if not (len(ref_min) == len(ref_max) == len(limits_lower) == len(limits_upper) == n):
        raise ValueError(
            "reference/limit sequences must all have one entry per DOF "
            f"(got {n} names, {len(ref_min)}/{len(ref_max)} ref, "
            f"{len(limits_lower)}/{len(limits_upper)} limits)"
        )
    if soft_margin_frac <= 0.0:
        return []

    findings: List[Overlap] = []
    scale = abs(weight) * proximity_scale
    for i in range(n):
        lo, hi = float(limits_lower[i]), float(limits_upper[i])
        joint_range = hi - lo
        if joint_range <= 0.0:  # fixed joint; the kernel masks these out too
            continue
        margin = soft_margin_frac * joint_range
        for side, extreme, limit, dist in (
            ("lower", float(ref_min[i]), lo, float(ref_min[i]) - lo),
            ("upper", float(ref_max[i]), hi, hi - float(ref_max[i])),
        ):
            if dist >= margin:
                continue  # reference never enters this band
            prox = min(max((margin - dist) / margin, 0.0), 1.0)
            findings.append(
                Overlap(
                    dof_name=dof_names[i],
                    side=side,
                    ref_extreme=extreme,
                    limit=limit,
                    band_edge=limit + margin if side == "lower" else limit - margin,
                    at_or_past_limit=dist <= 0.0,
                    penetration_frac=(margin - dist) / margin,
                    cost_per_step=scale * prox,
                )
            )
    findings.sort(key=lambda f: (not f.at_or_past_limit, -f.penetration_frac))
    return findings


def format_report(
    findings: Sequence[Overlap], soft_margin_frac: float, n_dofs: int
) -> List[str]:
    """Boot-log lines. One clean line when there is nothing to report."""
    tag = "[ref-penalty]"
    if soft_margin_frac <= 0.0:
        return [
            f"{tag} limits_dof_pos soft margin is 0.0 -- proximity term OFF, "
            f"no reference/penalty overlap possible."
        ]
    if not findings:
        return [
            f"{tag} OK: no reference DOF enters the limits_dof_pos penalty band "
            f"(margin {soft_margin_frac:.3f} of range, {n_dofs} DOFs checked)."
        ]
    pinned = [f for f in findings if f.at_or_past_limit]
    lines = [
        f"{tag} CONTRADICTION: {len(findings)} of {n_dofs} DOFs have a "
        f"REFERENCE that enters a band limits_dof_pos PENALISES "
        f"(margin {soft_margin_frac:.3f} of range).",
    ]
    lines += [f"{tag}   {f.describe()}" for f in findings]
    if pinned:
        names = ", ".join(sorted({f.dof_name for f in pinned}))
        lines += [
            f"{tag} {len(pinned)} of those sit AT or PAST the limit ({names}). "
            f"prox == 1.0 at dist == 0 for ANY band width, so NARROWING "
            f"soft_margin_frac CANNOT fix these -- only setting it to 0.0 "
            f"(or moving the limit) removes the charge.",
        ]
    lines += [
        f"{tag} The tracking terms command these values; limits_dof_pos taxes "
        f"arriving at them. See DAWN2.md 'READER/WRITER-LAW VIOLATION'.",
    ]
    return lines


def check_from_env(env, log_fn) -> Optional[List[str]]:
    """Wire-up: pull reference stats + penalty config off a built env.

    Returns the emitted lines (for tests), or None when the check cannot run --
    no motion corpus, no `limits_dof_pos` component, or no reference DOF cache.
    Never raises into boot: a diagnostic that can break a training run is worse
    than no diagnostic.
    """
    try:
        component = (getattr(env.config, "reward_components", None) or {}).get(
            "limits_dof_pos"
        )
        if component is None:
            return None
        params = getattr(component, "static_params", None) or {}
        soft_margin_frac = float(params.get("soft_margin_frac", 0.0) or 0.0)

        dps = getattr(env.motion_lib, "dps", None)
        if dps is None or dps.ndim != 2 or dps.shape[0] == 0:
            return None

        kin = env.robot_config.kinematic_info
        dof_names = list(getattr(kin, "dof_names", []) or [])
        if len(dof_names) != dps.shape[1]:
            # A representation mismatch (e.g. exp-map DOFs) means the columns
            # are not the joints these limits describe. Say so; do not guess.
            log_fn(
                f"[ref-penalty] SKIPPED: reference DOF width {dps.shape[1]} != "
                f"{len(dof_names)} named DOFs; cannot align columns to limits."
            )
            return None

        findings = find_reference_penalty_overlaps(
            dof_names=dof_names,
            ref_min=dps.min(dim=0).values.tolist(),
            ref_max=dps.max(dim=0).values.tolist(),
            limits_lower=kin.dof_limits_lower.tolist(),
            limits_upper=kin.dof_limits_upper.tolist(),
            soft_margin_frac=soft_margin_frac,
            proximity_scale=float(params.get("proximity_scale", 0.1) or 0.1),
            weight=float(params.get("weight", -10.0) or -10.0),
        )
        lines = format_report(findings, soft_margin_frac, len(dof_names))
        for line in lines:
            log_fn(line)
        return lines
    except Exception as exc:  # never break boot for a diagnostic
        log_fn(f"[ref-penalty] check failed ({type(exc).__name__}: {exc}); skipped.")
        return None
