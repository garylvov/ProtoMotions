# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Env-var gate for PER-BODY WEIGHTS on the all-body tracking rewards.

Why this exists
---------------
``relative_body_pos`` (weight 1.0, sigma 0.3) reduces over ALL ~29 H1-2 bodies
with UNIFORM weights -- ``body_weights`` is supported by the kernel and the
factory but is NOT present in the live v59 config, because the ``pickup`` preset
leaves ``use_weighted_tracking=False``. The reduction is::

    E = sum_i w_i * ||e_i||^2 / sum_i w_i        (uniform: w_i = 1)
    r = exp(-E / sigma^2) [+ fine companion]

so with 29 uniform bodies each wrist receives 1/29 = 3.4% of the term's
attention, and the gradient the wrist sees through this term is

    dr/de_wrist = r' * 2 * w_wrist * e_wrist / sum(w)

i.e. LINEAR in ``w_wrist / sum(w)``. Concentrating weight on the wrist/hand
chain focuses capacity exactly where the sub-centimetre objective is measured.

Because the reduction is NORMALIZED by ``sum(w)``, the term's value range is
unchanged (max stays 1.0 -> the component weight is still the term's ceiling):
this knob RESHAPES attention, it does not rescale the reward. The price is
dilution elsewhere -- every unweighted body's gradient share is multiplied by
``N / sum(w)`` -- which is why the recommended weighting keeps the ratio modest
and why this gate refuses to guess: the weights are stated explicitly.

ANCHOR NOTE. ``relative_body_pos`` scores in the ANCHOR-RELATIVE frame, where
the anchor body's own error is IDENTICALLY zero on both sides. The pelvis
therefore contributes 0 to the numerator and 1 to the denominator -- it is pure
dilution. Zeroing its weight is safe and free, but it is left at 1.0 by default
so this gate changes exactly what the operator asked it to change.

Knobs
-----
======================================  ====================================
env var                                 effect
======================================  ====================================
``PM_BODY_WEIGHTS``                     ``pattern=weight`` pairs, comma
                                        separated. ``pattern`` is an exact
                                        body name or an ``fnmatch`` glob
                                        (``*_wrist_yaw_link=4``). Later
                                        tokens win over earlier ones, so a
                                        broad glob can be refined by a
                                        specific name. A pattern matching NO
                                        body is a hard ValueError -- a
                                        mistyped body name must never be a
                                        silent uniform run.
``PM_BODY_WEIGHTS_DEFAULT``             weight for bodies no pattern
                                        matched. Default 1.0.
``PM_BODY_WEIGHTS_COMPONENTS``          comma-separated reward components to
                                        apply the weights to. Default
                                        ``relative_body_pos``. A named
                                        component absent from the config
                                        logs a loud SKIPPED warning.
======================================  ====================================

RULE 10 / RESUME SAFETY
-----------------------
* ``PM_BODY_WEIGHTS`` unset => hard no-op: not one ``static_params`` key
  written, not one line logged. A frozen config is byte-identical.
* A spec that resolves to UNIFORM weights (every body equal) also writes
  nothing and warns, because uniform-via-weights and uniform-via-None are the
  same reduction and writing the keys would perturb the pickle for no effect.

IDEMPOTENCE
-----------
The gate writes an ABSOLUTE weight vector derived only from the env spec and the
robot's body-name list, never a multiplier against the current value. Applying
it N times is therefore identical to applying it once -- the same property the
``weight``/``sigma`` resume rows have, and the property the NOISE-DR scale gates
had to be retrofitted with (see ``noise_scale_env_gates.NOISE_SCALE_BASELINE_ATTR``).

ONE implementation, BOTH wiring paths, so they can never drift:

* fresh build -- ``imprint``'s ``teacher.py::env_config`` calls it on the
  freshly built ``reward_components`` dict;
* resume      -- ``protomotions/train_agent.py`` calls it on the components
  unpickled from ``resolved_configs.pt`` (teacher.py never runs on a resume).
"""

import fnmatch
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

BODY_WEIGHTS_VAR = "PM_BODY_WEIGHTS"
BODY_WEIGHTS_DEFAULT_VAR = "PM_BODY_WEIGHTS_DEFAULT"
BODY_WEIGHTS_COMPONENTS_VAR = "PM_BODY_WEIGHTS_COMPONENTS"

BODY_WEIGHT_ENV_VARS: Tuple[str, ...] = (
    BODY_WEIGHTS_VAR,
    BODY_WEIGHTS_DEFAULT_VAR,
    BODY_WEIGHTS_COMPONENTS_VAR,
)

#: Components the weights land on when ``PM_BODY_WEIGHTS_COMPONENTS`` is unset.
#: ``relative_body_pos`` only: it is the all-body position term whose ~29-way
#: uniform mean is the dilution this gate exists to fix. ``relative_body_ori``
#: takes the same kwargs but orientation is a different objective with a
#: different sigma, and silently reshaping it too would confound the experiment.
DEFAULT_BODY_WEIGHT_COMPONENTS: Tuple[str, ...] = ("relative_body_pos",)


def body_weight_env_gate_requested(env: Optional[Dict[str, str]] = None) -> bool:
    """True when the per-body weighting spec is explicitly present.

    Only ``PM_BODY_WEIGHTS`` arms the gate. The other two vars are modifiers:
    setting them alone is a configuration mistake and is reported as such by
    :func:`apply_body_weight_env_overrides`.
    """
    env = os.environ if env is None else env
    return env.get(BODY_WEIGHTS_VAR) not in (None, "")


def _read_weight(raw: str, where: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: weight must be a float, got {raw!r}")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"{where}: weight must be a finite value >= 0 (relative per-body "
            f"attention in a normalized weighted mean), got {raw!r}"
        )
    return value


def parse_body_weight_spec(
    spec: str,
    body_names: Sequence[str],
    default_weight: float = 1.0,
) -> List[float]:
    """Resolve a ``pattern=weight,...`` spec to a full per-body weight vector.

    Args:
        spec: comma-separated ``pattern=weight`` tokens. ``pattern`` is an exact
            body name or an ``fnmatch`` glob. Later tokens override earlier ones.
        body_names: the robot's ORDERED body name list. The returned vector is
            aligned to it index-for-index.
        default_weight: weight for bodies no pattern matched.

    Raises:
        ValueError: on an empty spec, a malformed token, a non-finite/negative
            weight, or a pattern that matches NO body. That last one is the
            important case: a mistyped body name must be a hard failure, never a
            silently-uniform run that looks like it worked.
    """
    if not body_names:
        raise ValueError(
            f"{BODY_WEIGHTS_VAR} needs the robot's ordered body name list "
            "(robot_config.kinematic_info.body_names) to resolve patterns; got "
            "an empty/absent list. Refusing to guess an alignment."
        )
    if not spec or not spec.strip():
        raise ValueError(f"{BODY_WEIGHTS_VAR} is set but empty.")

    weights = [float(default_weight)] * len(body_names)
    seen: Dict[str, float] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"{BODY_WEIGHTS_VAR} token {token!r} is not 'pattern=weight'."
            )
        pattern, _, raw = token.partition("=")
        pattern = pattern.strip()
        value = _read_weight(raw.strip(), f"{BODY_WEIGHTS_VAR} token {token!r}")
        matched = [
            i for i, name in enumerate(body_names) if fnmatch.fnmatchcase(name, pattern)
        ]
        if not matched:
            raise ValueError(
                f"{BODY_WEIGHTS_VAR} pattern {pattern!r} matched NO body. Known "
                f"bodies: {list(body_names)}. A pattern that matches nothing is "
                "a hard error, not a silent uniform run -- that is exactly the "
                "'looks like it worked' failure class this gate must not have."
            )
        for i in matched:
            weights[i] = value
            seen[body_names[i]] = value
    return weights


def _aligned_weights(
    weights: Sequence[float], body_indices: Optional[Sequence[int]]
) -> List[float]:
    """Project a full-body weight vector onto a component's body subset."""
    if body_indices is None:
        return [float(w) for w in weights]
    return [float(weights[int(i)]) for i in body_indices]


def apply_body_weight_env_overrides(
    reward_components: Optional[Dict[str, Any]],
    body_names: Optional[Sequence[str]],
    log_fn: Callable[[str], None],
    label: str,
    env: Optional[Dict[str, str]] = None,
) -> bool:
    """Apply ``PM_BODY_WEIGHTS`` onto the targeted reward components.

    Args:
        reward_components: name -> ``MdpComponent`` mapping to mutate. ``None``
            or empty logs a loud SKIPPED warning when the gate is armed.
        body_names: the robot's ordered body name list.
        log_fn: single-string logger (``log.warning``) for the proof lines.
        label: proof-line prefix, ``"FRESH-BUILD"`` or ``"RESUME"``.
        env: environment mapping override (tests).

    Returns:
        True when at least one component was mutated.
    """
    env = os.environ if env is None else env

    if not body_weight_env_gate_requested(env):
        # Modifiers without the spec do nothing -- say so rather than let a
        # half-configured launcher look armed.
        stray = [
            v
            for v in (BODY_WEIGHTS_DEFAULT_VAR, BODY_WEIGHTS_COMPONENTS_VAR)
            if env.get(v) is not None
        ]
        if stray:
            log_fn(
                f"{label} BODY-WEIGHTS gate SKIPPED: {'/'.join(stray)} set but "
                f"{BODY_WEIGHTS_VAR} is unset/empty -- per-body weighting is OFF "
                "and those vars have NO effect."
            )
        return False

    spec = env[BODY_WEIGHTS_VAR]
    default_weight = (
        _read_weight(env[BODY_WEIGHTS_DEFAULT_VAR], BODY_WEIGHTS_DEFAULT_VAR)
        if env.get(BODY_WEIGHTS_DEFAULT_VAR) is not None
        else 1.0
    )
    targets = [
        t.strip()
        for t in (
            env.get(BODY_WEIGHTS_COMPONENTS_VAR)
            or ",".join(DEFAULT_BODY_WEIGHT_COMPONENTS)
        ).split(",")
        if t.strip()
    ]
    if not targets:
        raise ValueError(f"{BODY_WEIGHTS_COMPONENTS_VAR} is set but names no component.")

    weights = parse_body_weight_spec(spec, body_names or [], default_weight)

    if len(set(weights)) <= 1:
        log_fn(
            f"{label} BODY-WEIGHTS gate wrote NOTHING: {BODY_WEIGHTS_VAR}={spec!r} "
            f"resolves to UNIFORM weights ({weights[0] if weights else 'n/a'} on "
            "every body), which is the same reduction as the default uniform "
            "mean. The config is byte-identical to unset."
        )
        return False

    if not reward_components:
        log_fn(
            f"{label} BODY-WEIGHTS gate SKIPPED: {BODY_WEIGHTS_VAR} is set but "
            "the config has no reward_components (env var has NO effect)."
        )
        return False

    total = sum(weights)
    n = len(weights)
    changed = False
    for name in targets:
        component = reward_components.get(name)
        if component is None:
            log_fn(
                f"{label} BODY-WEIGHTS gate SKIPPED for '{name}': "
                f"{BODY_WEIGHTS_VAR} is set but that reward component is not in "
                "the config (this component gets NO weighting)."
            )
            continue
        static_params = component.static_params
        existing_indices = static_params.get("body_indices")
        aligned = _aligned_weights(weights, existing_indices)
        if len(set(aligned)) <= 1:
            log_fn(
                f"{label} BODY-WEIGHTS gate SKIPPED for '{name}': the spec is "
                f"uniform ({aligned[0]}) across that component's body subset "
                f"{list(existing_indices)}, so it would not change the reduction."
            )
            continue
        indices = (
            list(range(n)) if existing_indices is None else list(existing_indices)
        )
        previous = static_params.get("body_weights")
        if previous is not None and [float(p) for p in previous] == aligned:
            log_fn(
                f"{label} BODY-WEIGHTS '{name}' ALREADY at the requested weights "
                "-- no write (this gate is idempotent by construction: it stores "
                "an ABSOLUTE weight vector, never a multiplier)."
            )
            continue
        static_params["body_indices"] = indices
        static_params["body_weights"] = aligned
        changed = True
        subset_total = sum(aligned)
        named = ", ".join(
            f"{body_names[i]}={w:g}"
            for i, w in zip(indices, aligned)
            if w != default_weight
        )
        log_fn(
            f"{label} BODY-WEIGHTS override {name}.body_weights set from "
            f"{BODY_WEIGHTS_VAR} (was {previous!r}, default {default_weight:g}): "
            f"{named or '(all at default)'}. Reduction is the NORMALIZED "
            f"weighted mean sum(w*e^2)/sum(w), so the term's value range is "
            f"unchanged; sum(w)={subset_total:g} over {len(aligned)} bodies "
            f"(uniform would be {len(aligned)}). Per-body gradient share scales "
            f"by w_i * {len(aligned)} / {subset_total:g}, i.e. a body at weight "
            f"{default_weight:g} now sees "
            f"{default_weight * len(aligned) / subset_total:.3f}x its uniform "
            "share."
        )
    return changed
