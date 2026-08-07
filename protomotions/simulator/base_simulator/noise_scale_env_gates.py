# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Env-var gate for TRAINING OBSERVATION / ACTION NOISE magnitudes (NOISE-DR).

Why this exists
---------------
``stages_night13.py:17-20`` documents ``action_noise`` and ``observation_noise``
as **NON-RAMPED**: they sit at the heavy-DR ceiling for the whole run no matter
what ``DR_START_STAGE`` says. The 2026-08-07 sub-centimetre wrist analysis
(``docs/SUBCM_WRIST_PLAN.md`` §2b) propagated those two settings through the
real H1-2 wrist Jacobian and measured:

* ``action_noise_range = (-0.05, 0.05)``  -> **26.4 mm rms** wrist-equivalent
  (a *persistent* per-DOF PD-target bias, not jitter -- a memoryless policy
  cannot integrate it out),
* ``observation_noise.dof_pos_noise = 0.02`` -> **10.6 mm rms**
  (``torso_joint`` alone is 7.27 mm of it),
* combined -> **28.4 mm rms**, against v57's best-ever clip of **21.2 mm**.

i.e. the policy is sitting at its own observability floor, and until now there
was **no env gate that could move it**. These knobs are that gate.

ONE implementation shared by BOTH wiring paths, so the fresh-build gate and the
resume re-apply row can never drift apart:

* fresh build  -- ``imprint``'s ``teacher.py::configure_robot_and_simulator``
  mutates the stage table's ``DomainRandomizationConfig`` before the simulator
  is constructed. The mutated object is what ``train_agent.py`` pickles into
  ``resolved_configs.pt``, so the new magnitudes are stamped for free.
* resume       -- ``protomotions/train_agent.py`` mutates the FROZEN
  ``simulator_config`` unpickled from the checkpoint (teacher.py never runs on
  a resume). Action noise is resampled at every simulator build
  (``simulator.py::_process_action_noise_domain_randomization``) and
  observation noise is read live off ``simulator.config.domain_randomization``
  every step (``env.py``), so mutating the config is sufficient in both cases.

RULE-10 RESUME SAFETY
---------------------
Every knob below is read with ``os.environ.get`` and the gate is a hard no-op
-- not one field written, not one line logged -- when NONE of them is present.
An unset environment therefore leaves a frozen config **byte-identical**.

Stronger than that: a scale that resolves to **exactly 1.0** also writes
nothing. The OFF path is the *literal original object*, never
``original * 1.0``, so there is no route by which a float round-trip could
perturb a stored magnitude by one ulp. ``test_noise_scale_env_gates.py`` proves
this by pickling the config before and after and comparing the bytes.

Knobs
-----
==============================  ==========================================
env var                         effect (multiplicative, default 1.0)
==============================  ==========================================
``PM_ACTION_NOISE_SCALE``       scales BOTH ends of
                                ``domain_randomization.action_noise
                                .action_noise_range``. ``0`` removes the
                                block entirely (``action_noise = None``)
                                rather than leaving a degenerate
                                ``(0.0, 0.0)`` range, which
                                ``ActionNoiseDomainRandomizationConfig
                                .__post_init__`` would reject if the config
                                were ever rebuilt.
``PM_OBS_NOISE_SCALE``          scales EVERY magnitude field of
                                ``domain_randomization.observation_noise``
                                (a ``RobotNoiseConfig``): dof/root/anchor/
                                whole-body/ground-height. Scalars and
                                per-axis lists both handled. At ``0`` every
                                field is 0.0, so ``RobotNoiseConfig
                                .has_noise()`` goes False and ``env.py``
                                skips the noise path altogether.
``PM_ANCHOR_ROT_NOISE_SCALE``   an EXTRA multiplier applied to
                                ``anchor_rot_noise`` only, composed on top
                                of ``PM_OBS_NOISE_SCALE``. See below for
                                why this one earns its own knob.
==============================  ==========================================

Only the observation-noise config is touched. The **reset** noise
(``stages_night13.apply_reset_noise`` -> ``robot_cfg``, consumed by
``observation_noise.py:283``) is a different ``RobotNoiseConfig`` on a
different object and is deliberately left alone: it is RSI initial-condition
spread, not an observability limit, and shrinking it would narrow the state
distribution the policy is trained over without buying any wrist precision.

Why ``anchor_rot_noise`` gets its own knob
------------------------------------------
It is the one observation-noise field whose magnitude is both **large** and
**separately suspect**. ``anchor_rot_noise = 0.1`` on each quaternion
component, renormalized, measures as **0.190 rad = 10.9 deg** of pelvis
attitude error -- an implausible state-estimator error model for a precision
task, and worth 156 mm of apparent wrist displacement at the 0.82 m mean wrist
radius. The analysis is explicit that the 156 mm is an *upper bound*: the plant
low-passes the i.i.d. component down to the measured ~2.1 mm tremor, while its
bias/conservatism path is real but unmeasured. That is exactly the shape of a
term that needs to be moved **independently** of the rest so the next run can
attribute the change, instead of being confounded inside a single
``PM_OBS_NOISE_SCALE``.

Validation
----------
Every scale must parse as a finite float ``>= 0``. Negative scales are a hard
``ValueError`` (a sign flip on a symmetric uniform band is a silent no-op, so
accepting one would hide a typo); NaN/inf likewise.
"""

import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

#: Every magnitude field on ``RobotNoiseConfig``. Enumerated explicitly rather
#: than derived from ``dataclasses.fields`` so that a future non-magnitude
#: field (a mode flag, a name list) cannot silently start being multiplied.
OBS_NOISE_FIELDS: Tuple[str, ...] = (
    "dof_pos_noise",
    "dof_vel_noise",
    "root_pos_noise",
    "root_rot_noise",
    "root_vel_noise",
    "root_ang_vel_noise",
    "anchor_rot_noise",
    "anchor_ang_vel_noise",
    "body_pos_noise",
    "body_rot_noise",
    "body_vel_noise",
    "body_ang_vel_noise",
    "ground_height_noise",
)

ACTION_NOISE_SCALE_VAR = "PM_ACTION_NOISE_SCALE"
OBS_NOISE_SCALE_VAR = "PM_OBS_NOISE_SCALE"
ANCHOR_ROT_NOISE_SCALE_VAR = "PM_ANCHOR_ROT_NOISE_SCALE"

NOISE_SCALE_ENV_VARS: Tuple[str, ...] = (
    ACTION_NOISE_SCALE_VAR,
    OBS_NOISE_SCALE_VAR,
    ANCHOR_ROT_NOISE_SCALE_VAR,
)


def noise_scale_env_gate_requested(env: Optional[Dict[str, str]] = None) -> bool:
    """True when at least one NOISE-DR env knob is explicitly present."""
    env = os.environ if env is None else env
    return any(env.get(var) is not None for var in NOISE_SCALE_ENV_VARS)


def _read_scale(env: Dict[str, str], var: str) -> Optional[float]:
    """Parse and validate one scale knob. ``None`` when unset."""
    raw = env.get(var)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{var} must be a float, got {raw!r}")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"{var} must be a finite scale >= 0 (multiplicative on the existing "
            f"noise magnitude; 1.0 = unchanged, 0.0 = noise off), got {raw!r}"
        )
    return value


def _scaled(
    value: Union[float, List[float], Tuple[float, ...], None], scale: float
) -> Union[float, List[float], Tuple[float, ...], None]:
    """Multiply a scalar or a per-axis sequence. ``None`` passes through."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        scaled = [float(v) * scale for v in value]
        return type(value)(scaled) if isinstance(value, tuple) else scaled
    return float(value) * scale


def _is_zero(value: Union[float, List[float], Tuple[float, ...], None]) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return all(float(v) == 0.0 for v in value)
    return float(value) == 0.0


def apply_noise_scale_env_overrides(
    domain_randomization: Any,
    log_fn: Callable[[str], None],
    label: str,
    env: Optional[Dict[str, str]] = None,
) -> bool:
    """Apply the NOISE-DR env knobs onto a ``DomainRandomizationConfig``.

    Args:
        domain_randomization: the DR config object to mutate, or ``None`` when
            DR is stripped/absent (then the gate logs a loud SKIPPED warning so
            a set knob is never silently inert).
        log_fn: single-string logger (``log.warning``) for the proof lines.
        label: proof-line prefix identifying the wiring path, e.g.
            ``"FRESH-BUILD"`` or ``"RESUME"``.
        env: environment mapping override (tests).

    Returns:
        True when the config was mutated.
    """
    env = os.environ if env is None else env
    if not noise_scale_env_gate_requested(env):
        return False

    present = [v for v in NOISE_SCALE_ENV_VARS if env.get(v) is not None]
    if domain_randomization is None:
        log_fn(
            f"{label} NOISE-DR gate SKIPPED: {'/'.join(present)} set but "
            "domain_randomization is stripped/absent (env vars have NO effect)"
        )
        return False

    action_scale = _read_scale(env, ACTION_NOISE_SCALE_VAR)
    obs_scale = _read_scale(env, OBS_NOISE_SCALE_VAR)
    anchor_scale = _read_scale(env, ANCHOR_ROT_NOISE_SCALE_VAR)

    changed = False

    # ------------------------------------------------------------------ action
    if action_scale is not None:
        action_noise = getattr(domain_randomization, "action_noise", None)
        if action_noise is None:
            log_fn(
                f"{label} NOISE-DR gate SKIPPED for actions: "
                f"{ACTION_NOISE_SCALE_VAR}={action_scale} set but "
                "domain_randomization.action_noise is stripped/absent "
                "(this env var has NO effect)"
            )
        elif action_scale == 1.0:
            # Rule 10: leave the ORIGINAL tuple object untouched. Writing
            # `old * 1.0` back would be numerically exact but is still a write,
            # and the whole contract here is byte-identity of the frozen config.
            log_fn(
                f"{label} NOISE-DR action_noise_range UNCHANGED "
                f"({ACTION_NOISE_SCALE_VAR}=1.0 -> literal original value "
                f"{tuple(action_noise.action_noise_range)} kept, no write)"
            )
        elif action_scale == 0.0:
            old = tuple(action_noise.action_noise_range)
            domain_randomization.action_noise = None
            changed = True
            log_fn(
                f"{label} NOISE-DR action_noise REMOVED "
                f"(domain_randomization.action_noise: range {old} -> None, from "
                f"{ACTION_NOISE_SCALE_VAR}=0). The whole persistent PD-target "
                "bias block is off; a degenerate (0.0, 0.0) range is NOT used "
                "because ActionNoiseDomainRandomizationConfig.__post_init__ "
                "rejects low >= high on any rebuild."
            )
        else:
            old = tuple(action_noise.action_noise_range)
            new = tuple(float(v) * action_scale for v in old)
            action_noise.action_noise_range = new
            changed = True
            log_fn(
                f"{label} NOISE-DR override action_noise.action_noise_range = "
                f"{new} (was {old}, from {ACTION_NOISE_SCALE_VAR}="
                f"{action_scale}). This is a PERSISTENT per-DOF PD-target bias; "
                "at (-0.05, 0.05) it measures 26.4 mm rms of wrist-position "
                "equivalent (SUBCM_WRIST_PLAN.md 2b)."
            )

    # ------------------------------------------------------------------ obs
    if obs_scale is not None or anchor_scale is not None:
        obs_noise = getattr(domain_randomization, "observation_noise", None)
        obs_vars = "/".join(
            v
            for v in (OBS_NOISE_SCALE_VAR, ANCHOR_ROT_NOISE_SCALE_VAR)
            if env.get(v) is not None
        )
        if obs_noise is None:
            log_fn(
                f"{label} NOISE-DR gate SKIPPED for observations: {obs_vars} set "
                "but domain_randomization.observation_noise is stripped/absent "
                "(these env vars have NO effect)"
            )
        else:
            base = 1.0 if obs_scale is None else obs_scale
            for field in OBS_NOISE_FIELDS:
                if not hasattr(obs_noise, field):
                    # Field absent on this (older//newer) config class -- say so
                    # rather than silently skipping a magnitude we meant to move.
                    log_fn(
                        f"{label} NOISE-DR observation_noise.{field} SKIPPED: "
                        "field absent on this RobotNoiseConfig"
                    )
                    continue
                scale = base
                if field == "anchor_rot_noise" and anchor_scale is not None:
                    scale = base * anchor_scale
                old = getattr(obs_noise, field)
                if scale == 1.0:
                    # Literal original value kept -- see the action branch.
                    continue
                if _is_zero(old):
                    # 0 * anything is 0; skip the write so an all-zero frozen
                    # config stays byte-identical even under a set knob.
                    continue
                new = _scaled(old, scale)
                setattr(obs_noise, field, new)
                changed = True
                extra = ""
                if field == "anchor_rot_noise" and anchor_scale is not None:
                    extra = (
                        f" [{OBS_NOISE_SCALE_VAR}={base} x "
                        f"{ANCHOR_ROT_NOISE_SCALE_VAR}={anchor_scale}]"
                    )
                log_fn(
                    f"{label} NOISE-DR override observation_noise.{field} = "
                    f"{new} (was {old}, scale {scale}){extra}"
                )
            if changed and not obs_noise.has_noise():
                log_fn(
                    f"{label} NOISE-DR observation_noise.has_noise() is now "
                    "False -- env.py will skip the observation-noise path "
                    "entirely. The policy trains on CLEAN observations."
                )

    if not changed:
        log_fn(
            f"{label} NOISE-DR gate present ({'/'.join(present)}) but wrote "
            "NOTHING (every resolved scale was 1.0 or its target magnitude was "
            "already 0). The config is byte-identical to unset."
        )
    return changed
