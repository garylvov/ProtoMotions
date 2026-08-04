# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Env-var gate for MARIONETTE actuator-gain / effort-limit DR (GAIN-DR).

ONE implementation shared by BOTH wiring paths, so the fresh-build gate and
the resume re-apply row can never drift apart:

* fresh build  -- ``imprint``'s ``teacher.py::configure_robot_and_simulator``
  mutates the stage table's ``DomainRandomizationConfig`` before the simulator
  is constructed.
* resume       -- ``protomotions/train_agent.py`` mutates the FROZEN
  ``simulator_config`` unpickled from the checkpoint (teacher.py never runs on
  a resume). GAIN-DR resamples at every simulator build, so mutating the
  ranges is sufficient.

RULE-10 RESUME SAFETY: every knob below is read with ``os.environ.get`` and
the gate is a hard no-op -- not one field written, not one line logged --
when NONE of them is present. An unset environment therefore leaves a frozen
config byte-identical.

Three independent axes
----------------------
=========  ===============================  ====================================
axis       env prefix                       config field
=========  ===============================  ====================================
stiffness  ``PM_GAIN_DR_{LOW,HIGH}``        ``stiffness_scale_range`` /
                                            ``group_stiffness_scale_ranges``
damping    ``PM_GAIN_DR_KD_{LOW,HIGH}``     ``damping_scale_range`` /
           (falls back to ``PM_GAIN_DR_*``) ``group_damping_scale_ranges``
effort     ``PM_EFFORT_DR_{LOW,HIGH}``      ``effort_limit_scale_range`` /
                                            ``group_effort_limit_scale_ranges``
=========  ===============================  ====================================

Each prefix takes an optional ``_LEGS`` / ``_WAIST`` / ``_ARMS`` suffix for a
per-joint-group band, e.g. ``PM_GAIN_DR_KD_LOW_ARMS=0.1``. Resolution order
for any (axis, group) band, first present wins::

    <axis>_<BOUND>_<GROUP>  ->  PM_GAIN_DR_<BOUND>_<GROUP>  (damping only)
      ->  <axis>_<BOUND>    ->  PM_GAIN_DR_<BOUND> (damping only)
      ->  the config's current global range for that axis

so ``PM_GAIN_DR_LOW_ARMS`` alone still moves BOTH arm stiffness and arm
damping (the briefed "one knob pair per group" behavior), while
``PM_GAIN_DR_KD_LOW_ARMS`` lets damping be softened INDEPENDENTLY -- which is
what kinesthetic teaching actually needs, since a human dragging a hand fights
``kd * v`` directly.

Presence of ANY group-suffixed knob switches the sampler into per-group mode
(joint-group partition resolved from ``H1_2_GAIN_DR_GROUP_PATTERNS``, loud
failure on an unmatched or double-matched DOF).

Extra knobs
-----------
``PM_GAIN_DR_CONSTANT_ZETA``
    ``1`` derives each DOF's damping scale as ``sqrt(stiffness_scale)``
    instead of sampling it, holding zeta = d / (2*sqrt(k*m)) constant.
    Default OFF = independent damping sample = today's behavior. Mutually
    exclusive with the damping knobs above (constant-zeta wins and says so).
``PM_GAIN_DR_ENV_SCALE_SOURCE``
    ``all`` | ``legs`` | ``waist`` | ``arms`` -- which group's geometric-mean
    stiffness scale feeds the MARIONETTE perturbation coupling's per-env
    ``env_gain_scale``. Unset = AUTO (``legs`` when per-group mode is on,
    ``all`` otherwise = today's behavior).

EFFORT-LIMIT NOTE (verified 2026-08-04): the effort axis populates the
long-existing-but-never-enabled ``effort_limit_scale_range`` field. Its apply
path (``isaaclab/simulator.py`` -> ``write_joint_effort_limit_to_sim`` ->
``root_physx_view.set_dof_max_forces``) is the REAL torque saturation for
``ControlType.BUILT_IN_PD`` / ``ImplicitActuatorCfg`` robots, which is what
this fleet runs (``robot_configs/h1_2.py``: ``control_type=BUILT_IN_PD``,
``effort_limit_sim=control_info.effort_limit``). IsaacLab's docstring warns
"this function isn't setting the values for actuator models (#128)": harmless
here because an implicit actuator computes no torque in python, but a robot
switched to ``IdealPDActuatorCfg`` would get only HALF the clip (PhysX yes,
python actuator model no) -- re-verify before reusing this axis off
BUILT_IN_PD.
"""

import os
from typing import Any, Callable, Dict, Optional, Tuple

GAIN_DR_GROUPS: Tuple[str, ...] = ("legs", "waist", "arms")

#: axis -> (env prefix, fallback env prefix or None, range field, group field)
GAIN_DR_AXES: Tuple[Tuple[str, str, Optional[str], str, str], ...] = (
    (
        "stiffness",
        "PM_GAIN_DR",
        None,
        "stiffness_scale_range",
        "group_stiffness_scale_ranges",
    ),
    (
        "damping",
        "PM_GAIN_DR_KD",
        "PM_GAIN_DR",
        "damping_scale_range",
        "group_damping_scale_ranges",
    ),
    (
        "effort_limit",
        "PM_EFFORT_DR",
        None,
        "effort_limit_scale_range",
        "group_effort_limit_scale_ranges",
    ),
)

_EXTRA_VARS: Tuple[str, ...] = (
    "PM_GAIN_DR_CONSTANT_ZETA",
    "PM_GAIN_DR_ENV_SCALE_SOURCE",
)


def _axis_vars(prefix: str) -> Tuple[str, ...]:
    return (f"{prefix}_LOW", f"{prefix}_HIGH") + tuple(
        f"{prefix}_{bound}_{group.upper()}"
        for group in GAIN_DR_GROUPS
        for bound in ("LOW", "HIGH")
    )


def _group_vars(prefix: str) -> Tuple[str, ...]:
    return tuple(
        f"{prefix}_{bound}_{group.upper()}"
        for group in GAIN_DR_GROUPS
        for bound in ("LOW", "HIGH")
    )


GAIN_DR_ENV_VARS: Tuple[str, ...] = tuple(
    dict.fromkeys(
        [v for _, prefix, _, _, _ in GAIN_DR_AXES for v in _axis_vars(prefix)]
        + list(_EXTRA_VARS)
    )
)


def gain_dr_env_gate_requested(env: Optional[Dict[str, str]] = None) -> bool:
    """True when at least one GAIN-DR env knob is explicitly present."""
    env = os.environ if env is None else env
    return any(env.get(var) is not None for var in GAIN_DR_ENV_VARS)


def _truthy(value: Optional[str]) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def apply_gain_dr_env_overrides(
    actuator_gain: Any,
    log_fn: Callable[[str], None],
    label: str,
    env: Optional[Dict[str, str]] = None,
) -> bool:
    """Apply the GAIN-DR env knobs onto an ``ActuatorGainDomainRandomizationConfig``.

    Args:
        actuator_gain: the config object to mutate, or ``None`` when GAIN-DR is
            stripped/absent (then the gate logs a loud SKIPPED warning so a set
            knob is never silently inert).
        log_fn: single-string logger (``log.warning``) for the proof lines.
        label: proof-line prefix identifying the wiring path, e.g.
            ``"FRESH-BUILD"`` or ``"RESUME"``.
        env: environment mapping override (tests).

    Returns:
        True when the config was mutated.
    """
    env = os.environ if env is None else env
    if not gain_dr_env_gate_requested(env):
        return False

    present = [v for v in GAIN_DR_ENV_VARS if env.get(v) is not None]
    if actuator_gain is None:
        log_fn(
            f"{label} MARIONETTE gain gate SKIPPED: {'/'.join(present)} set but "
            "actuator_gain DR is stripped/absent (env vars have NO effect)"
        )
        return False

    def _bound(*names: str) -> Optional[float]:
        """First explicitly-present var among ``names``, as a float."""
        for name in names:
            raw = env.get(name)
            if raw is not None:
                return float(raw)
        return None

    changed = False
    group_requested = any(
        env.get(v) is not None
        for _, prefix, _, _, _ in GAIN_DR_AXES
        for v in _group_vars(prefix)
    )

    for axis, prefix, fallback, range_field, group_field in GAIN_DR_AXES:
        prefixes = (prefix,) if fallback is None else (prefix, fallback)

        # --- global range for this axis ------------------------------------
        g_lo = _bound(*(f"{p}_LOW" for p in prefixes))
        g_hi = _bound(*(f"{p}_HIGH" for p in prefixes))
        old = getattr(actuator_gain, range_field, None)
        # effort_limit_scale_range defaults to None (= "leave effort limits
        # alone"); once ANY effort knob is present the axis turns on, and the
        # implicit base band is the no-op (1.0, 1.0) so unmentioned groups
        # keep their nominal limits exactly.
        base = tuple(old) if old is not None else (1.0, 1.0)
        if g_lo is not None or g_hi is not None:
            new = (
                g_lo if g_lo is not None else float(base[0]),
                g_hi if g_hi is not None else float(base[1]),
            )
            if not (0.0 < new[0] <= new[1]):
                raise ValueError(
                    f"{prefix}_LOW/{prefix}_HIGH must satisfy 0 < low <= high, "
                    f"got {new}"
                )
            setattr(actuator_gain, range_field, new)
            base = new
            changed = True
            log_fn(
                f"{label} override actuator_gain.{range_field} = {new} "
                f"(was {tuple(old) if old is not None else None}, from "
                f"{'/'.join(f'{p}_LOW,{p}_HIGH' for p in prefixes)})"
            )

        # --- per-group ranges for this axis ---------------------------------
        axis_group_requested = any(
            env.get(v) is not None for v in _group_vars(prefix)
        ) or (
            fallback is not None
            and any(env.get(v) is not None for v in _group_vars(fallback))
        )
        if not axis_group_requested:
            # An axis only enters per-group mode when a group knob for IT (or,
            # for damping, the shared PM_GAIN_DR_* pair) is present -- so a
            # stiffness-only marionette config never starts writing effort
            # limits, and an effort-only config never touches the gains.
            continue

        groups: Dict[str, Tuple[float, float]] = dict(
            getattr(actuator_gain, group_field, None) or {}
        )
        for group in GAIN_DR_GROUPS:
            up = group.upper()
            lo = _bound(*(f"{p}_LOW_{up}" for p in prefixes))
            hi = _bound(*(f"{p}_HIGH_{up}" for p in prefixes))
            new = (
                lo if lo is not None else float(base[0]),
                hi if hi is not None else float(base[1]),
            )
            if not (0.0 < new[0] <= new[1]):
                raise ValueError(
                    f"{prefix}_LOW_{up}/{prefix}_HIGH_{up} must satisfy "
                    f"0 < low <= high, got {new}"
                )
            old_group = groups.get(group, tuple(base))
            groups[group] = new
            changed = True
            log_fn(
                f"{label} override actuator_gain GROUP '{group}' {axis} range "
                f"= {new} (was {tuple(old_group)}, from "
                f"{'/'.join(f'{p}_LOW_{up},{p}_HIGH_{up}' for p in prefixes)} "
                f"defaulting to the global {axis} band {tuple(base)})"
            )
        setattr(actuator_gain, group_field, groups)
        if axis == "effort_limit" and getattr(actuator_gain, range_field, None) is None:
            # Turn the axis on with a no-op global band so the sampler emits an
            # effort-scale matrix at all (None would skip it entirely).
            setattr(actuator_gain, range_field, (float(base[0]), float(base[1])))
            log_fn(
                f"{label} actuator_gain.effort_limit_scale_range turned ON at "
                f"the no-op band {tuple(base)} because per-group effort knobs "
                "are set (this axis has NEVER run in production before -- "
                "check the '[gain-dr] applied: ... effort_limit' proof line)"
            )

    # --- constant damping ratio --------------------------------------------
    zeta_raw = env.get("PM_GAIN_DR_CONSTANT_ZETA")
    if zeta_raw is not None:
        old_zeta = bool(getattr(actuator_gain, "constant_damping_ratio", False))
        new_zeta = _truthy(zeta_raw)
        actuator_gain.constant_damping_ratio = new_zeta
        changed = True
        effect = (
            "damping scale is DERIVED as sqrt(stiffness scale), holding "
            "zeta = d/(2*sqrt(k*m)) constant -- this OVERRIDES every damping "
            "range/knob (PM_GAIN_DR_KD_*)"
            if new_zeta
            else "damping sampled independently from its own range"
        )
        log_fn(
            f"{label} override actuator_gain.constant_damping_ratio = "
            f"{new_zeta} (was {old_zeta}, from PM_GAIN_DR_CONSTANT_ZETA="
            f"{zeta_raw!r}); {effect}"
        )

    # --- env_gain_scale source ---------------------------------------------
    src_raw = env.get("PM_GAIN_DR_ENV_SCALE_SOURCE")
    if src_raw is not None:
        src = src_raw.strip().lower()
        if src not in ("all",) + GAIN_DR_GROUPS:
            raise ValueError(
                f"PM_GAIN_DR_ENV_SCALE_SOURCE must be one of "
                f"{('all',) + GAIN_DR_GROUPS}, got {src_raw!r}"
            )
        old_src = getattr(actuator_gain, "env_gain_scale_group", None)
        actuator_gain.env_gain_scale_group = src
        changed = True
        log_fn(
            f"{label} override actuator_gain.env_gain_scale_group = {src!r} "
            f"(was {old_src!r}, from PM_GAIN_DR_ENV_SCALE_SOURCE); this "
            "selects which group's geometric-mean stiffness scale drives the "
            "MARIONETTE perturbation coupling"
        )
    elif group_requested:
        log_fn(
            f"{label} actuator_gain.env_gain_scale_group left AUTO -> resolves "
            "to 'legs' because per-group ranges are active (balance authority "
            "lives in the legs; set PM_GAIN_DR_ENV_SCALE_SOURCE=all to keep "
            "the all-DOF geometric mean)"
        )

    return changed
