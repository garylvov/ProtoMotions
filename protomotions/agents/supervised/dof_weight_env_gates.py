# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Env-var gate for PER-DOF WEIGHTING of the distillation (BC) MSE.

Why this exists
---------------
The masked-mimic STUDENT captures only ~10% of the wrist motion it is
conditioned on; the TEACHER it distils from captures ~88%. In 3 of 7 eval
categories the student scores WORSE than a frozen-wrist null policy that simply
parks its hands, and doubling the conditioning density improves every category
EXCEPT ``reaching`` -- the one regime that is purely about hand placement.
Three independently trained lineages with three different conditioning
interfaces all show the same signature. That is causal confusion: the student
solves the task from proprioception and the hand channel never earns gradient.

The mechanism is in the loss. ``compute_supervision_loss`` takes a FLAT
``F.mse_loss`` over all 27 H1-2 action DOFs, so every DOF's share of the
gradient is its share of the squared error. On H1-2 the six wrist DOFs are a
low-variance minority (6/27 = 22% of the dims, and far less than that of the
error energy), so the channel we actually care about is a rounding error in the
objective.

This module is the name-addressed knob that fixes that, plus the reader that
makes its effect visible.

Interface
---------
``PM_MM_DOF_WEIGHTS`` -- comma-separated ``glob=weight`` pairs, e.g.::

    PM_MM_DOF_WEIGHTS='*_wrist_*=4.0,*_elbow_*=2.0,*_shoulder_*=1.5'

The literal value ``default`` expands to :data:`DEFAULT_DOF_WEIGHT_SPEC`, which
is exactly the profile above. Any DOF matched by no pattern keeps weight 1.0.
When patterns overlap, the LAST matching pattern wins (so a broad rule can be
written first and narrowed afterwards); the resolver logs every override.

Weights resolve from DOF **NAMES** via :mod:`fnmatch`, never from indices.
Index lists silently rot the moment the robot config changes its DOF ordering,
and this campaign has already been bitten by exactly that class of bug. A glob
that matches NOTHING is a hard ``ValueError``: a typo'd pattern would otherwise
degrade to "everything stays 1.0", i.e. it would look like the change shipped
while training exactly the unweighted objective this change exists to replace.
That silent-no-op failure mode is the whole point, so it is fatal here.

The resolved vector is a plain length-``number_of_actions`` list stored on
``SupervisedAgentConfig.action_dim_weights``. The loss module itself never sees
DOF names -- resolution happens once at config-build time, on both wiring
paths, through this one shared implementation:

* **fresh build** -- ``train_agent.py`` calls
  :func:`apply_dof_weight_env_overrides` right after the experiment file's
  ``agent_config()`` returns, so the vector is pickled into
  ``resolved_configs.pt`` for free.
* **resume** -- ``train_agent.py::detect_checkpoint_mode`` never re-executes
  the experiment file: ``results/<EXP>/config.yaml`` is written back onto
  ``args`` and ``resolved_configs.pt`` supplies the real config objects, so
  environment knobs and CLI flags are otherwise INERT on a resume. The SAME
  call site therefore also runs on the resume branch's unpickled
  ``agent_config``, using the FROZEN robot config's ``dof_names`` so the
  resolution is against the DOF ordering the run was actually built with.
  ``stage_resume_config.py --dof-weights`` can additionally bake the vector
  into the frozen pickle permanently.

Both paths share this module, so they cannot drift.

Numerical contract
------------------
The weighted MSE is

    ``(((pred - target) ** 2) * w).sum(-1) / w.sum()``  then batch-meaned

which is normalized by ``w.sum()`` **so that a uniform ``w`` reproduces the
current ``F.mse_loss`` exactly**. Loss scale, learning-rate meaning and resume
dynamics are therefore unchanged by the mere presence of the knob;
``test_dof_weight_env_gates.py`` proves the all-ones case against
``F.mse_loss`` directly.

RULE-10 RESUME SAFETY: the gate is a hard no-op -- not one field written, not
one line logged -- when ``PM_MM_DOF_WEIGHTS`` is absent. An unset environment
leaves a frozen config byte-identical.

The reader
----------
A weighting knob with no way to see its effect is debt: at epoch 50 you could
not tell a working fix from a silently inert one, which is precisely the hole
this campaign already fell into. So :func:`resolve_dof_groups` splits the DOFs
into named anatomical groups and the agent logs the **unweighted** per-group
MSE for each of them every epoch, ALWAYS -- weighted run or not, so the two are
directly comparable across runs. Watch ``masked_mimic/mse_group/wrists``.
"""

import fnmatch
import math
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Env var carrying the ``glob=weight`` spec.
DOF_WEIGHT_SPEC_VAR = "PM_MM_DOF_WEIGHTS"

#: Recommended starting profile, also what ``PM_MM_DOF_WEIGHTS=default`` means.
#:
#: Sizing (H1-2, 27 DOFs: 12 leg, 1 waist, 6 shoulder, 2 elbow, 6 wrist):
#: sum(w) = 6*4.0 + 2*2.0 + 6*1.5 + 13*1.0 = 50.0, so the normalized gradient
#: share moves wrists 22.2% -> 48.0%, elbows 7.4% -> 8.0%, shoulders 22.2% ->
#: 18.0%, and legs+waist 48.1% -> 26.0%. That roughly inverts the DOF-count and
#: variance imbalance behind the 10%-vs-88% wrist-capture gap without starving
#: locomotion, which currently WORKS (backward_locomotion 7.71 cm, static_hold
#: 8.25 cm) and must not regress -- the lower body keeps a quarter of the
#: gradient, and this lands on a RESUME from an already-competent policy rather
#: than on a cold start. A more aggressive wrist weight (8.0 puts wrists at 65%
#: and legs at 18%) is deliberately NOT the default: the per-group readout
#: below is the instrument that makes the next step data-driven instead of
#: another guess.
#:
#: ``torso_joint`` stays at 1.0 on purpose. The wrist reaches partly THROUGH
#: the waist, so upweighting it is tempting, but it is also the single joint
#: whose posture the locomotion and static-hold categories are most sensitive
#: to, and it is one DOF against six wrists -- poor return for the regression
#: risk.
DEFAULT_DOF_WEIGHT_SPEC = "*_wrist_*=4.0,*_elbow_*=2.0,*_shoulder_*=1.5"

#: Weight given to a DOF that no pattern matches.
BASE_DOF_WEIGHT = 1.0

#: Anatomical groups for the per-group MSE readout, FIRST match wins. Ordered
#: most-specific first so ``*_wrist_*`` cannot be swallowed by a broader arm
#: pattern. Name-addressed for the same reason the weights are.
DOF_GROUP_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("wrists", ("*_wrist_*",)),
    ("elbows", ("*_elbow_*",)),
    ("shoulders", ("*_shoulder_*",)),
    ("waist", ("torso_joint", "*waist*", "*_torso_*")),
    ("legs", ("*_hip_*", "*_knee_*", "*_ankle_*", "*_thigh_*", "*_calf_*")),
)

#: Coarse roll-ups logged alongside the fine groups. The brief's minimum bar is
#: "arms/wrists vs legs/waist"; these two are that bar, and they stay stable
#: across robots whose fine group membership differs.
DOF_GROUP_ROLLUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("arms", ("wrists", "elbows", "shoulders")),
    ("lower", ("legs", "waist")),
)

#: Bucket for DOFs matched by no group pattern. Never silently dropped.
UNGROUPED_NAME = "other"


def dof_weight_env_gate_requested(env: Optional[Dict[str, str]] = None) -> bool:
    """True when ``PM_MM_DOF_WEIGHTS`` is explicitly present."""
    env = os.environ if env is None else env
    return env.get(DOF_WEIGHT_SPEC_VAR) is not None


def parse_dof_weight_spec(spec: str) -> List[Tuple[str, float]]:
    """Parse ``glob=weight[,glob=weight...]`` into ordered pairs.

    ``default`` (case-insensitive, whitespace-stripped) expands to
    :data:`DEFAULT_DOF_WEIGHT_SPEC`.

    Raises:
        ValueError: on an empty spec, a malformed entry, or a weight that is
            not a finite float >= 0. A negative weight is rejected rather than
            clamped: it would flip the sign of that DOF's gradient, which is
            never what anybody means and would quietly destabilize training.
    """
    if spec is None:
        raise ValueError(f"{DOF_WEIGHT_SPEC_VAR} spec is None")
    text = spec.strip()
    if text.lower() == "default":
        text = DEFAULT_DOF_WEIGHT_SPEC
    if not text:
        raise ValueError(
            f"{DOF_WEIGHT_SPEC_VAR} is empty. Use 'default' for the recommended "
            f"profile ({DEFAULT_DOF_WEIGHT_SPEC!r}), or unset the variable "
            "entirely for stock uniform MSE."
        )

    pairs: List[Tuple[str, float]] = []
    for entry in text.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"{DOF_WEIGHT_SPEC_VAR} entry {entry!r} is not 'glob=weight'. "
                f"Example: {DEFAULT_DOF_WEIGHT_SPEC!r}"
            )
        pattern, _, raw_weight = entry.partition("=")
        pattern = pattern.strip()
        raw_weight = raw_weight.strip()
        if not pattern:
            raise ValueError(
                f"{DOF_WEIGHT_SPEC_VAR} entry {entry!r} has an empty glob pattern"
            )
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            raise ValueError(
                f"{DOF_WEIGHT_SPEC_VAR} entry {entry!r}: weight {raw_weight!r} "
                "is not a float"
            )
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"{DOF_WEIGHT_SPEC_VAR} entry {entry!r}: weight must be a finite "
                f"value >= 0 (1.0 = unchanged, 0.0 = this DOF contributes no "
                f"gradient), got {raw_weight!r}"
            )
        pairs.append((pattern, weight))

    if not pairs:
        raise ValueError(
            f"{DOF_WEIGHT_SPEC_VAR}={spec!r} parsed to zero 'glob=weight' entries"
        )
    return pairs


def resolve_dof_weights(
    spec: str,
    dof_names: Sequence[str],
    log_fn: Optional[Callable[[str], None]] = None,
) -> List[float]:
    """Resolve a ``glob=weight`` spec against ordered DOF names.

    Args:
        spec: the ``PM_MM_DOF_WEIGHTS`` string (``default`` accepted).
        dof_names: the robot's ORDERED DOF names,
            ``robot_config.kinematic_info.dof_names``. This is the same
            ordering the action vector uses, which is why resolution must
            happen where the robot config is in scope and not inside the loss.
        log_fn: optional single-string logger for per-pattern override lines.

    Returns:
        A list of floats, one per DOF, in DOF order.

    Raises:
        ValueError: if ``dof_names`` is empty/None, or if any pattern matches
            no DOF name. The latter is fatal by design -- see the module
            docstring; a silently-unmatched pattern is indistinguishable from
            the unweighted loss we are replacing.
    """
    if not dof_names:
        raise ValueError(
            f"{DOF_WEIGHT_SPEC_VAR} cannot be resolved: dof_names is empty or "
            "None. Weights resolve from DOF NAMES (robot_config.kinematic_info"
            ".dof_names); index-addressed weights silently rot when the robot "
            "config changes and are not supported."
        )

    names = list(dof_names)
    pairs = parse_dof_weight_spec(spec)
    weights = [float(BASE_DOF_WEIGHT)] * len(names)
    assigned_by: List[Optional[str]] = [None] * len(names)

    for pattern, weight in pairs:
        matched = [i for i, name in enumerate(names) if fnmatch.fnmatchcase(name, pattern)]
        if not matched:
            raise ValueError(
                f"{DOF_WEIGHT_SPEC_VAR} pattern {pattern!r} matched NONE of the "
                f"robot's {len(names)} DOF names. This is fatal, not a warning: "
                "an unmatched pattern would leave the loss uniform while the run "
                "logs as if it were weighted. DOF names are: "
                f"{names}"
            )
        for i in matched:
            if assigned_by[i] is not None and log_fn is not None:
                log_fn(
                    f"[DOF-WEIGHTS] {names[i]}: {weights[i]} (from "
                    f"{assigned_by[i]!r}) OVERRIDDEN to {weight} by later "
                    f"pattern {pattern!r}"
                )
            weights[i] = float(weight)
            assigned_by[i] = pattern

    return weights


def validate_dof_weights(weights: Sequence[float], number_of_actions: int) -> None:
    """Hard length check of a resolved weight vector against the action dim.

    Raises:
        ValueError: on a length mismatch. Loud is the point: a vector of the
            wrong length either silently broadcasts or errors deep inside the
            loss on the first optimizer step, hours after the run started.
    """
    if weights is None:
        raise ValueError("resolved DOF weight vector is None")
    if len(weights) != int(number_of_actions):
        raise ValueError(
            f"resolved DOF weight vector has length {len(weights)} but the "
            f"robot has number_of_actions={number_of_actions}. The weight "
            "vector must have exactly one entry per action DOF, in DOF order."
        )


def resolve_dof_groups(dof_names: Sequence[str]) -> "OrderedDict[str, List[int]]":
    """Split ordered DOF names into anatomical groups for the MSE readout.

    First matching entry of :data:`DOF_GROUP_PATTERNS` wins. DOFs matched by
    nothing land in :data:`UNGROUPED_NAME` rather than disappearing, so the
    groups always partition the action vector and the readout can never hide a
    channel. Empty groups are omitted. Roll-ups from
    :data:`DOF_GROUP_ROLLUPS` are appended when at least one member group is
    non-empty.
    """
    if not dof_names:
        raise ValueError("resolve_dof_groups requires a non-empty dof_names list")

    names = list(dof_names)
    groups: "OrderedDict[str, List[int]]" = OrderedDict(
        (group, []) for group, _ in DOF_GROUP_PATTERNS
    )
    groups[UNGROUPED_NAME] = []

    for i, name in enumerate(names):
        for group, patterns in DOF_GROUP_PATTERNS:
            if any(fnmatch.fnmatchcase(name, p) for p in patterns):
                groups[group].append(i)
                break
        else:
            groups[UNGROUPED_NAME].append(i)

    resolved: "OrderedDict[str, List[int]]" = OrderedDict(
        (group, idx) for group, idx in groups.items() if idx
    )
    for rollup, members in DOF_GROUP_ROLLUPS:
        idx = sorted(i for m in members for i in groups.get(m, ()))
        if idx:
            resolved[rollup] = idx
    return resolved


def format_dof_weight_proof(
    weights: Sequence[float],
    dof_names: Sequence[str],
    label: str,
    source: str,
) -> List[str]:
    """Build the loud startup proof lines showing the weights actually in force.

    Reports, per group: the weight(s), the DOF count, and the group's share of
    total gradient BEFORE and AFTER weighting. The share is the number that
    matters -- it is what the optimizer sees -- and printing the uniform
    baseline next to it makes an inert gate obvious at a glance (every
    ``share`` would equal its ``uniform``).
    """
    names = list(dof_names)
    values = [float(w) for w in weights]
    groups = resolve_dof_groups(names)
    rollup_names = {r for r, _ in DOF_GROUP_ROLLUPS}
    total = sum(values)
    n = len(values)

    lines = [
        f"[DOF-WEIGHTS] {label}: per-DOF supervision weights ACTIVE "
        f"(source: {source})",
    ]
    for group, idx in groups.items():
        gw = [values[i] for i in idx]
        distinct = sorted(set(gw))
        shown = (
            f"{distinct[0]:g}" if len(distinct) == 1
            else "/".join(f"{v:g}" for v in distinct)
        )
        share = 100.0 * sum(gw) / total if total > 0 else 0.0
        uniform = 100.0 * len(idx) / n
        kind = "rollup" if group in rollup_names else "group "
        lines.append(
            f"[DOF-WEIGHTS] {label}:   {kind} {group:<10s} n={len(idx):<3d} "
            f"w={shown:<12s} gradient share {share:6.2f}%  "
            f"(uniform would be {uniform:6.2f}%)"
        )
    lines.append(
        f"[DOF-WEIGHTS] {label}: sum(w)={total:.4f} mean(w)={total / n:.4f} "
        f"over {n} DOFs; loss is (((pred-target)**2)*w).sum(-1)/w.sum(), "
        "batch-meaned -- uniform w reproduces F.mse_loss exactly, so loss "
        "scale and LR meaning are unchanged."
    )
    lines.append(
        f"[DOF-WEIGHTS] {label}: per-DOF vector = "
        + ", ".join(f"{name}={value:g}" for name, value in zip(names, values))
    )
    lines.append(
        f"[DOF-WEIGHTS] {label}: READER -- per-group UNWEIGHTED supervision MSE "
        "is logged every epoch as masked_mimic/mse_group/<group> (wrists, "
        "elbows, shoulders, waist, legs, arms, lower). If this change is "
        "working, mse_group/wrists falls relative to its own pre-change "
        "trajectory while mse_group/lower stays flat. If the gate were inert "
        "those curves would be indistinguishable from the unweighted run."
    )
    return lines


def apply_dof_weight_env_overrides(
    agent_config: Any,
    dof_names: Optional[Sequence[str]],
    log_fn: Callable[[str], None],
    label: str,
    env: Optional[Dict[str, str]] = None,
    number_of_actions: Optional[int] = None,
) -> bool:
    """Resolve ``PM_MM_DOF_WEIGHTS`` onto ``agent_config.action_dim_weights``.

    ONE implementation shared by the fresh-build and resume wiring rows in
    ``train_agent.py``, so the two can never drift apart.

    Args:
        agent_config: the agent config to mutate. Must expose
            ``action_dim_weights`` (``SupervisedAgentConfig`` and subclasses);
            anything else is a hard error rather than a silent skip, because a
            set env var that does nothing is the exact failure this module
            exists to prevent.
        dof_names: the robot's ordered DOF names. On a resume this MUST come
            from the FROZEN robot config so resolution matches the DOF ordering
            the run was built with.
        log_fn: single-string logger (``log.warning``) for the proof lines.
        label: proof-line prefix identifying the wiring path, ``"FRESH-BUILD"``
            or ``"RESUME"``.
        env: environment mapping override (tests).
        number_of_actions: optional action-dim cross-check. When given, the
            resolved vector length is validated against it.

    Returns:
        True when ``action_dim_weights`` was written.
    """
    env = os.environ if env is None else env
    if not dof_weight_env_gate_requested(env):
        return False

    spec = env[DOF_WEIGHT_SPEC_VAR]

    if agent_config is None or not hasattr(agent_config, "action_dim_weights"):
        raise ValueError(
            f"{DOF_WEIGHT_SPEC_VAR}={spec!r} is set but the agent config "
            f"({type(agent_config).__name__}) has no 'action_dim_weights' "
            "field, so the weights would be silently inert. Per-DOF "
            "supervision weighting is only supported for SupervisedAgentConfig "
            "and its subclasses (e.g. MaskedMimicSupervisedAgentConfig). Unset "
            f"{DOF_WEIGHT_SPEC_VAR} or use a supervised/distillation recipe."
        )

    if not dof_names:
        raise ValueError(
            f"{DOF_WEIGHT_SPEC_VAR}={spec!r} is set but the robot config "
            "carries no kinematic_info.dof_names, so name-addressed weights "
            "cannot be resolved. Refusing to fall back to uniform weights: "
            "that would train the unweighted objective while logging as if "
            "weighted."
        )

    weights = resolve_dof_weights(spec, dof_names, log_fn=log_fn)
    if number_of_actions is not None:
        validate_dof_weights(weights, number_of_actions)

    previous = getattr(agent_config, "action_dim_weights", None)
    if previous is not None and list(previous) != weights:
        log_fn(
            f"[DOF-WEIGHTS] {label}: REPLACING frozen action_dim_weights "
            f"{list(previous)} with the vector resolved from "
            f"{DOF_WEIGHT_SPEC_VAR}"
        )
    agent_config.action_dim_weights = weights

    for line in format_dof_weight_proof(
        weights, dof_names, label, f"{DOF_WEIGHT_SPEC_VAR}={spec!r}"
    ):
        log_fn(line)
    return True
