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

IDEMPOTENCE ACROSS RESUMES (2026-08-07 defect fix)
--------------------------------------------------
The knobs are MULTIPLIERS and the gate runs on EVERY resume, so as originally
written they compounded: v59's relaunch turned ``action_noise_range``
``(-0.05, 0.05) -> (-0.01, 0.01)`` on the first launch and
``(-0.01, 0.01) -> (-0.002, 0.002)`` on the resume, i.e. 25x below nominal
instead of the intended 5x, with log lines indistinguishable from a correct
first application. The scale is now always applied to the NOMINAL magnitude
recorded once in ``NOISE_SCALE_BASELINE_ATTR``, so ``apply(apply(x)) ==
apply(x)`` by construction and a changed scale lands on ``nominal * new_scale``
exactly. See that attribute's docstring for the alternatives weighed.

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


#: Attribute stamped on the ``DomainRandomizationConfig`` the FIRST time this
#: gate writes anything, recording the NOMINAL (pre-gate) magnitudes.
#:
#: WHY (2026-08-07 resume-compounding defect). The knobs are MULTIPLIERS, and
#: the gate runs on EVERY resume against whatever is in the config it is handed.
#: On a resume that config is the ALREADY-SCALED one the previous launch
#: pickled, so ``PM_ACTION_NOISE_SCALE=0.2`` compounded::
#:
#:     launch : (-0.05, 0.05)   -> (-0.01,   0.01)      x0.2   intended
#:     resume : (-0.01, 0.01)   -> (-0.002,  0.002)     x0.04  NOT intended
#:     resume : (-0.002, 0.002) -> (-0.0004, 0.0004)    x0.008 ...
#:
#: v59 spent its second leg at 25x below nominal action noise instead of 5x,
#: and nothing in the log said so -- every line read "override ... (was ...)"
#: exactly as it does on a correct first application. That is the same failure
#: class as v57: a gate that looks like it worked and quietly did something
#: else.
#:
#: The fix is to make the multiplier's REFERENT explicit and permanent. The
#: scale is defined against the NOMINAL magnitude -- what the stage table built
#: -- so the gate records that nominal once and always computes
#: ``nominal * scale``. Applying the same environment twice is then a no-op by
#: construction, and CHANGING the scale between resumes lands on
#: ``nominal * new_scale`` (exactly, from the stored original -- no lossy
#: divide-out of the previous factor), which is what a multiplier against
#: nominal must mean.
#:
#: Alternatives weighed and rejected:
#:   * ABSOLUTE-target env vars. ``PM_OBS_NOISE_SCALE`` fans out over 13
#:     magnitude fields with different units (rad, m, rad/s, quaternion
#:     components); there is no single absolute number to express, and forcing
#:     13 new absolute knobs would be a far larger contract change than the
#:     defect warrants. ``PM_ACTION_NOISE_SCALE`` could go absolute, but then
#:     the three knobs would no longer share one semantics.
#:   * A "scale already applied" MARKER that skips when it matches. Idempotent
#:     for an UNCHANGED scale, but a CHANGED scale would then have to be
#:     applied as ``current * (new/old)`` -- a float divide whose round-trip is
#:     not exact, so the run's magnitudes would drift by ulps across resumes and
#:     the Rule-10 byte-identity proof would no longer hold. Recording the
#:     nominal costs the same one attribute and is exact.
#:
#: Contents (plain picklable primitives + the detached config object):
#:     {"version": 1,
#:      "action_noise": {"present": bool, "range": (lo, hi) | None, "obj": cfg},
#:      "observation_noise": {field_name: nominal_value, ...}}
NOISE_SCALE_BASELINE_ATTR = "_noise_scale_env_gate_nominal"

#: Stamp format version. Bump if the layout changes so an old pickle is
#: rejected loudly instead of silently misread.
NOISE_SCALE_BASELINE_VERSION = 1


def noise_scale_nominal_baseline(domain_randomization: Any) -> Optional[Dict[str, Any]]:
    """The recorded NOMINAL magnitudes, or ``None`` if the gate never fired.

    Public so tests, launchers and post-mortems can read what the multipliers
    are actually multiplying, without importing the private attribute name.
    """
    if domain_randomization is None:
        return None
    return getattr(domain_randomization, NOISE_SCALE_BASELINE_ATTR, None)


def _copy_value(
    value: Union[float, List[float], Tuple[float, ...], None]
) -> Union[float, List[float], Tuple[float, ...], None]:
    """Snapshot a magnitude BY VALUE so a later in-place edit cannot corrupt it."""
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, tuple):
        return tuple(float(v) for v in value)
    return value


def _read_baseline(domain_randomization: Any) -> Optional[Dict[str, Any]]:
    stamp = getattr(domain_randomization, NOISE_SCALE_BASELINE_ATTR, None)
    if stamp is None:
        return None
    if not isinstance(stamp, dict) or stamp.get("version") != NOISE_SCALE_BASELINE_VERSION:
        raise ValueError(
            f"{NOISE_SCALE_BASELINE_ATTR} on the frozen domain_randomization is "
            f"not a version-{NOISE_SCALE_BASELINE_VERSION} NOISE-DR baseline "
            f"stamp (got {stamp!r}). Refusing to guess what the multipliers are "
            "multiplying -- that guess is exactly the compounding bug this "
            "stamp exists to prevent."
        )
    return stamp


def apply_noise_scale_env_overrides(
    domain_randomization: Any,
    log_fn: Callable[[str], None],
    label: str,
    env: Optional[Dict[str, str]] = None,
) -> bool:
    """Apply the NOISE-DR env knobs onto a ``DomainRandomizationConfig``.

    IDEMPOTENT (2026-08-07). Every scale is applied to the NOMINAL magnitude
    recorded in ``NOISE_SCALE_BASELINE_ATTR`` on first use, never to whatever
    happens to be in the config right now, so running this gate N times with
    the same environment gives the same result as running it once. See the
    ``NOISE_SCALE_BASELINE_ATTR`` docstring for the defect this closes and the
    alternatives that were rejected.

    Args:
        domain_randomization: the DR config object to mutate, or ``None`` when
            DR is stripped/absent (then the gate logs a loud SKIPPED warning so
            a set knob is never silently inert).
        log_fn: single-string logger (``log.warning``) for the proof lines.
        label: proof-line prefix identifying the wiring path, e.g.
            ``"FRESH-BUILD"`` or ``"RESUME"``.
        env: environment mapping override (tests).

    Returns:
        True when the config was mutated by THIS call. A second call with the
        same environment returns False and writes nothing -- that return value
        is the idempotency signal, and ``test_noise_scale_env_gates.py`` asserts
        the pickled bytes are unchanged across it.
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

    stamp = _read_baseline(domain_randomization)
    if stamp is not None:
        log_fn(
            f"{label} NOISE-DR gate IDEMPOTENT RE-APPLY: a NOMINAL baseline is "
            f"already stamped on this config, so every scale below is computed "
            f"as nominal x scale, NOT as current x scale. Re-applying the same "
            f"{'/'.join(present)} therefore writes nothing and changes nothing."
        )
    elif label != "FRESH-BUILD":
        # A resume with NO stamp is either (a) the first resume after this fix
        # landed, whose frozen config was ALREADY scaled by the pre-fix gate, or
        # (b) a genuinely un-gated config. We cannot tell them apart -- the
        # pre-fix gate left no record, which is the whole defect -- so say so
        # rather than let the operator assume the multiplier is against nominal.
        log_fn(
            f"{label} NOISE-DR gate: NO nominal baseline is stamped on this "
            f"frozen config, so the CURRENT magnitudes are being adopted as "
            f"nominal. If this run was launched BEFORE 2026-08-07 with "
            f"{'/'.join(present)} set, its config is already scaled and this "
            f"call will scale it ONCE more (the pre-fix compounding defect). To "
            f"hold a pre-fix run where it is, set every NOISE-DR knob to 1.0 on "
            f"this resume; to restart the ladder from nominal, relaunch fresh."
        )
    new_stamp = (
        {"version": NOISE_SCALE_BASELINE_VERSION, "action_noise": None, "observation_noise": {}}
        if stamp is None
        else stamp
    )

    changed = False

    # ------------------------------------------------------------------ action
    if action_scale is not None:
        action_noise = getattr(domain_randomization, "action_noise", None)
        recorded = new_stamp.get("action_noise")
        if recorded is None:
            # First application: today's value IS the nominal.
            nominal_obj = action_noise
            nominal_range = (
                _copy_value(tuple(action_noise.action_noise_range))
                if action_noise is not None
                else None
            )
            nominal_present = action_noise is not None
        else:
            nominal_obj = recorded["obj"]
            nominal_range = recorded["range"]
            nominal_present = recorded["present"]

        if not nominal_present:
            log_fn(
                f"{label} NOISE-DR gate SKIPPED for actions: "
                f"{ACTION_NOISE_SCALE_VAR}={action_scale} set but "
                "domain_randomization.action_noise is stripped/absent "
                "(this env var has NO effect)"
            )
        else:
            record = {
                "present": True,
                "range": nominal_range,
                "obj": nominal_obj,
            }
            if action_scale == 1.0 and recorded is None:
                # Rule 10: never gated before and the scale is a no-op -- leave
                # the ORIGINAL object untouched. Writing `old * 1.0` back would
                # be numerically exact but is still a write, and the contract
                # here is byte-identity of the frozen config.
                log_fn(
                    f"{label} NOISE-DR action_noise_range UNCHANGED "
                    f"({ACTION_NOISE_SCALE_VAR}=1.0 -> literal original value "
                    f"{tuple(nominal_range)} kept, no write)"
                )
            elif action_scale == 0.0:
                if action_noise is None:
                    log_fn(
                        f"{label} NOISE-DR action_noise ALREADY REMOVED by an "
                        f"earlier application of {ACTION_NOISE_SCALE_VAR}=0 "
                        f"(nominal range {tuple(nominal_range)} is preserved in "
                        f"{NOISE_SCALE_BASELINE_ATTR} and would be restored by a "
                        "non-zero scale). Nothing written."
                    )
                else:
                    domain_randomization.action_noise = None
                    new_stamp["action_noise"] = record
                    changed = True
                    log_fn(
                        f"{label} NOISE-DR action_noise REMOVED "
                        f"(domain_randomization.action_noise: nominal range "
                        f"{tuple(nominal_range)} -> None, from "
                        f"{ACTION_NOISE_SCALE_VAR}=0). The whole persistent "
                        "PD-target bias block is off; a degenerate (0.0, 0.0) "
                        "range is NOT used because "
                        "ActionNoiseDomainRandomizationConfig.__post_init__ "
                        "rejects low >= high on any rebuild."
                    )
            else:
                target = (
                    tuple(nominal_range)
                    if action_scale == 1.0
                    else tuple(float(v) * action_scale for v in nominal_range)
                )
                current = (
                    tuple(action_noise.action_noise_range)
                    if action_noise is not None
                    else None
                )
                if action_noise is None:
                    # A previous run removed the block with scale 0; this run
                    # wants it back. Restore the ORIGINAL object, then scale.
                    nominal_obj.action_noise_range = target
                    domain_randomization.action_noise = nominal_obj
                    new_stamp["action_noise"] = record
                    changed = True
                    log_fn(
                        f"{label} NOISE-DR action_noise RESTORED and set to "
                        f"{target} (nominal {tuple(nominal_range)} x "
                        f"{ACTION_NOISE_SCALE_VAR}={action_scale}); an earlier "
                        "application had removed the block entirely."
                    )
                elif current == target:
                    log_fn(
                        f"{label} NOISE-DR action_noise.action_noise_range "
                        f"ALREADY {target} (nominal {tuple(nominal_range)} x "
                        f"{ACTION_NOISE_SCALE_VAR}={action_scale}) -- no write. "
                        "This is the idempotent path; before 2026-08-07 this "
                        "call would have compounded the scale a second time."
                    )
                    # Keep the stamp if it exists; if it does not, this branch is
                    # unreachable for scale != 1.0 (a fresh config cannot already
                    # equal nominal x scale unless scale == 1.0, handled above).
                    if recorded is None:
                        new_stamp["action_noise"] = record
                else:
                    action_noise.action_noise_range = target
                    new_stamp["action_noise"] = record
                    changed = True
                    log_fn(
                        f"{label} NOISE-DR override "
                        f"action_noise.action_noise_range = {target} "
                        f"(nominal {tuple(nominal_range)}, current was "
                        f"{current}, from {ACTION_NOISE_SCALE_VAR}="
                        f"{action_scale}). This is a PERSISTENT per-DOF "
                        "PD-target bias; at (-0.05, 0.05) it measures 26.4 mm "
                        "rms of wrist-position equivalent "
                        "(SUBCM_WRIST_PLAN.md 2b)."
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
            recorded_obs = new_stamp["observation_noise"]
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
                seen_before = field in recorded_obs
                nominal = (
                    recorded_obs[field] if seen_before else getattr(obs_noise, field)
                )
                if scale == 1.0 and not seen_before:
                    # Literal original value kept -- see the action branch.
                    continue
                if _is_zero(nominal):
                    # 0 * anything is 0; skip the write so an all-zero frozen
                    # config stays byte-identical even under a set knob.
                    continue
                target = (
                    _copy_value(nominal) if scale == 1.0 else _scaled(nominal, scale)
                )
                current = getattr(obs_noise, field)
                if not seen_before:
                    recorded_obs[field] = _copy_value(nominal)
                if current == target:
                    # Idempotent re-apply: already at nominal x scale.
                    continue
                setattr(obs_noise, field, target)
                changed = True
                extra = ""
                if field == "anchor_rot_noise" and anchor_scale is not None:
                    extra = (
                        f" [{OBS_NOISE_SCALE_VAR}={base} x "
                        f"{ANCHOR_ROT_NOISE_SCALE_VAR}={anchor_scale}]"
                    )
                log_fn(
                    f"{label} NOISE-DR override observation_noise.{field} = "
                    f"{target} (nominal {nominal}, current was {current}, "
                    f"scale {scale}){extra}"
                )
            if changed and not obs_noise.has_noise():
                log_fn(
                    f"{label} NOISE-DR observation_noise.has_noise() is now "
                    "False -- env.py will skip the observation-noise path "
                    "entirely. The policy trains on CLEAN observations."
                )

    if changed:
        # Stamp ONLY when something was actually written, so an unset (or
        # all-1.0) environment never adds an attribute and the frozen config
        # stays byte-identical.
        setattr(domain_randomization, NOISE_SCALE_BASELINE_ATTR, new_stamp)
    else:
        log_fn(
            f"{label} NOISE-DR gate present ({'/'.join(present)}) but wrote "
            "NOTHING. Either every resolved scale was 1.0 / its target "
            "magnitude was already 0 (config byte-identical to unset), or the "
            "config is ALREADY at nominal x scale from an earlier application "
            "(idempotent re-apply -- the intended state, reached once)."
        )
    return changed
