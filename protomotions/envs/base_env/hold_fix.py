# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HOLD-FIX: hold-conditioning balance spec (Gary, 2026-07-10).

During holds (reference still) the policy should BALANCE — small corrective
actuation allowed/expected, zero steps, zero rigidity. Never reward literal
zero actuation. Three independent, env-gated interventions (all OFF by
default; one env var each so they can deploy independently):

1. HOLD_ACTION_GRACE=1  (hold grace)
   Compute a per-env reference-stillness mask (reference root pos/rot + dof
   deltas below epsilon over a small window, OR an explicit reference-freeze
   from the mimic motion manager) and OR it into
   ``EnvContext.perturbation_grace_mask`` so the graced action_rate penalty
   untaxes joint-space corrections during stills. Steps stay taxed: the
   always-on ``in_the_air`` / ``max_feet_height`` components do NOT consume
   the grace mask (verified 2026-07-10 — their dynamic_vars bind only
   contacts/positions/progress).

2. FALL_TERM_PENALTY=<float>  (fall penalty, e.g. 5.0)
   Inject the dormant ``fall_penalty_factory`` reward component at weight
   ``-abs(FALL_TERM_PENALTY)``. It fires on the SAME condition as the
   anchor-height fall termination (which is pure early-termination today), on
   the terminating step — a negative terminal reward for falling.

3. HOLD_BALANCE_BONUS=<float>  (balance bonus, e.g. 0.2)
   Inject a small positive reward paid ONLY during reference-still windows
   for quiet stance: upright pelvis x both feet planted x low pelvis
   velocity. Pays for balancing, not for rigidity (no actuation term in it).

Tunables (all env vars, sane defaults):
   HOLD_STILL_ROOT_EPS   reference root linear speed epsilon, m/s   (0.05)
   HOLD_STILL_ROT_EPS    reference root angular speed epsilon, rad/s (0.30)
   HOLD_STILL_DOF_EPS    reference max |dof| speed epsilon, rad/s   (0.20)
   HOLD_STILL_WINDOW     consecutive still steps before mask fires  (5)
   HOLD_BALANCE_VEL_COEF exp coefficient on |root_vel|^2            (-4.0)
   HOLD_BALANCE_UP_COEF  exp coefficient on (1 - up_z)^2            (-10.0)

Rollback = unset the env vars (behavior is bit-identical to stock when all
three gates are off: no tracker allocated, no components injected, grace mask
untouched).

Launch-time log lines are prefixed ``[hold-fix]`` so dump-verify can prove
the flags live. Per-step TB signals (via env extras): ``hold_fix/ref_still_mask``
(stillness-mask fraction), ``hold_fix/grace_mask`` (post-OR grace fraction),
plus the standard ``raw_r/ scaled_r/`` tags of the injected components
(``hold_fix_fall_penalty``, ``hold_balance``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor


# =============================================================================
# Env-var gates
# =============================================================================


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


@dataclass
class HoldFixSettings:
    """Parsed HOLD-FIX env-var gates and tunables."""

    hold_action_grace: bool = False
    fall_term_penalty: float = 0.0  # magnitude; applied at negative weight
    hold_balance_bonus: float = 0.0  # positive weight
    # Stillness detection tunables
    still_root_eps: float = 0.05  # m/s
    still_rot_eps: float = 0.30  # rad/s
    still_dof_eps: float = 0.20  # rad/s (max over dofs)
    still_window: int = 5  # consecutive env steps
    # Balance-bonus kernel tunables
    balance_vel_coef: float = -4.0
    balance_up_coef: float = -10.0
    # Wrist direction-agreement reward (WRIST-DIR, Gary 2026-07-10: targets
    # the measured dir_cos 0.61 weak axis). weight 0.0 = off.
    # PRIMARY VARIANT = WINDOWED DISPLACEMENT (Gary refinement: step-to-step
    # velocity direction is too noisy): cos(x(t)-x(t-K), x_ref(t)-x_ref(t-K))
    # over a ~0.3 s window. Instantaneous-velocity variant stays behind
    # WRIST_DIR_INSTANT=1 for comparison.
    wrist_dir_weight: float = 0.0
    wrist_dir_window_s: float = 0.3  # displacement window (s); K = round(/dt)
    wrist_dir_subsample: int = 3  # store every M-th frame (ring = ceil(K/M) slots)
    wrist_dir_disp_eps: float = 0.02  # m over window; ref wrist actually traveled
    wrist_dir_instant: bool = False  # sub-flag: instantaneous-velocity variant
    wrist_dir_eps: float = 0.1  # m/s; instantaneous-variant activity gate
    wrist_dir_vmax: float = 0.0  # >0: ref-speed-proportional weighting cap (m/s)
    # Root displacement-gain reward (ROOT-GAIN, Gary 2026-07-10: teacher
    # fwd_gain 0.464 = structural displacement undershoot, regime-invariant).
    # Same windowed-ring machinery as wrist-dir with gain_proj shaping:
    # pay for MATCHING the reference's progress in its direction, clamp at
    # gain 1.0 (no overshoot bonus), negative projection = 0. weight 0 = off.
    root_gain_weight: float = 0.0
    root_gain_window_s: float = 0.3
    root_gain_subsample: int = 3
    root_gain_disp_eps: float = 0.03  # m over window (~0.1 m/s ref speed)
    # XY-drift early termination (XY-TERM, component [8], Gary 2026-07-10:
    # root drift is a reward negotiation the disc always outbids; termination
    # is the one lever class never outbid tonight). Fires ONLY on
    # moving-reference frames: ref anchor xy-speed > xy_move_eps AND NOT
    # reference-still (delta-based still mask guards against frozen-clock
    # velocity garbage). 0.0 = off.
    xy_drift_term_m: float = 0.0
    xy_move_eps: float = 0.3  # m/s ref anchor horizontal speed
    # XY-drift terminal penalty (component [8b], Gary 2026-07-10: the drift
    # fence must carry the SAME terminal cost as falling — without it
    # "committing xy suicide is better than falling": drift-out costs
    # episode-only while a fall costs episode + FALL_TERM_PENALTY, so the
    # fence is a discount exit. Magnitude; applied at NEGATIVE weight on the
    # exact xy_drift termination condition. Defaults to FALL_TERM_PENALTY so
    # the two exits are never priced apart. Active only when the xy_drift
    # termination itself is armed.
    xy_drift_term_penalty: float = 0.0
    # YAW-drift termination + terminal penalty (component [8c], Gary
    # 2026-07-10: "penalize theta disagreement" — heading twin of [8]+[8b].
    # Measured motive: turn_gain ~0.44-0.49 on every turning-rich val pack
    # (the angular twin of fwd_gain 0.464 — a commanded circle produces a
    # spiral; heldout real teleop 0.06) and W3 gait heading drift 500 deg.
    # Fires when |root yaw error vs reference| > yaw_drift_term_rad, ONLY
    # while the delta-based still mask says the reference is LIVE (frozen
    # clocks and stills never fire it; rot_eps 0.30 rad/s in the mask keeps
    # in-place turns live). NO ref yaw-rate gate ON PURPOSE: straight-walk
    # heading drift (the dominant observed failure) must stay inside the
    # fence. Error must PERSIST yaw_persist_steps consecutive env steps
    # (default 15 = 0.3 s @ 50 Hz) so a single noisy frame cannot kill an
    # episode. 0.0 = off.
    yaw_drift_term_rad: float = 0.0
    yaw_persist_steps: int = 15
    # Terminal penalty for the yaw fence; DEFAULT = FALL_TERM_PENALTY
    # (equal-pricing law across all three exits — fall, xy, theta: none may
    # ever be a discount). Active only when the yaw termination is armed.
    yaw_drift_term_penalty: float = 0.0
    # HOLD-BALANCE V2 (component 3 v2, 2026-07-10 21:19 ruling): posture-
    # RELATIVE + LIVING-BAND quiet-stance bonus. Pays ONLY during ref-stills,
    # ONLY when actuation is inside [ad_min, ad_max] (ZERO at zero motion BY
    # CONSTRUCTION — the only term that pays for aliveness); uprightness is
    # measured RELATIVE to the reference pelvis orientation (bent-waist holds
    # get paid). v1 (absolute-upright exp(-v^2)) experimentally indicted:
    # rigid pole at every weight; substrate = still-economics. weight 0 = off.
    hold_balance_v2: float = 0.0
    # v2.1 NOISE-CANCELLING BAND (2026-07-11 noise-loophole ruling): alive is
    # the windowed action-MEAN displacement (halves of a 2K ring), so
    # exploration noise cancels ~1/sqrt(K). Floor at sigma 0.103, K=15:
    # ~0.030; ad_min = 1.5x floor. (Old per-step band [0.005,0.15] sat BELOW
    # the 0.116 noise floor — farmed by noise, rigid at eval.)
    v2_ad_min: float = 0.045  # living-band floor (windowed |d action-mean|)
    v2_ad_max: float = 0.30  # band top; smoothly decaying pay above
    v2_alive_window: int = 15  # K; halves = 0.3 s each @ 50 Hz
    v2_posture_coef: float = -10.0  # exp coef on squared rel-tilt (rad^2)
    v2_dofvel_min: float = 0.05  # fallback band (rad/s) if no action history
    v2_dofvel_max: float = 2.0

    @property
    def needs_still_mask(self) -> bool:
        return (
            self.hold_action_grace
            or self.hold_balance_bonus > 0.0
            or self.xy_drift_term_m > 0.0
            or self.hold_balance_v2 > 0.0
            or self.yaw_drift_term_rad > 0.0
        )

    @property
    def any_enabled(self) -> bool:
        return (
            self.needs_still_mask
            or self.fall_term_penalty > 0.0
            or self.wrist_dir_weight > 0.0
            or self.root_gain_weight > 0.0
            or self.xy_drift_term_m > 0.0
            or self.hold_balance_v2 > 0.0
            or self.yaw_drift_term_rad > 0.0
        )


def load_hold_fix_settings() -> HoldFixSettings:
    """Parse the HOLD-FIX env-var gates. All off by default."""
    return HoldFixSettings(
        hold_action_grace=_env_flag("HOLD_ACTION_GRACE"),
        fall_term_penalty=abs(_env_float("FALL_TERM_PENALTY", 0.0)),
        hold_balance_bonus=_env_float("HOLD_BALANCE_BONUS", 0.0),
        still_root_eps=_env_float("HOLD_STILL_ROOT_EPS", 0.05),
        still_rot_eps=_env_float("HOLD_STILL_ROT_EPS", 0.30),
        still_dof_eps=_env_float("HOLD_STILL_DOF_EPS", 0.20),
        still_window=int(_env_float("HOLD_STILL_WINDOW", 5)),
        balance_vel_coef=_env_float("HOLD_BALANCE_VEL_COEF", -4.0),
        balance_up_coef=_env_float("HOLD_BALANCE_UP_COEF", -10.0),
        wrist_dir_weight=_env_float("WRIST_DIR_REWARD", 0.0),
        wrist_dir_window_s=_env_float("WRIST_DIR_WINDOW_S", 0.3),
        wrist_dir_subsample=int(_env_float("WRIST_DIR_SUBSAMPLE", 3)),
        wrist_dir_disp_eps=_env_float("WRIST_DIR_DISP_EPS", 0.02),
        wrist_dir_instant=_env_flag("WRIST_DIR_INSTANT"),
        wrist_dir_eps=_env_float("WRIST_DIR_EPS", 0.1),
        wrist_dir_vmax=_env_float("WRIST_DIR_VMAX", 0.0),
        root_gain_weight=_env_float("ROOT_GAIN_REWARD", 0.0),
        root_gain_window_s=_env_float("ROOT_GAIN_WINDOW_S", 0.3),
        root_gain_subsample=int(_env_float("ROOT_GAIN_SUBSAMPLE", 3)),
        root_gain_disp_eps=_env_float("ROOT_GAIN_DISP_EPS", 0.03),
        xy_drift_term_m=(
            _env_float("XY_DRIFT_TERM_M", 0.55)
            if _env_flag("XY_DRIFT_TERM")
            else 0.0
        ),
        xy_move_eps=_env_float("XY_DRIFT_MOVE_EPS", 0.3),
        xy_drift_term_penalty=abs(
            _env_float(
                "XY_DRIFT_TERM_PENALTY",
                abs(_env_float("FALL_TERM_PENALTY", 0.0)),
            )
        ),
        yaw_drift_term_rad=(
            _env_float("YAW_DRIFT_TERM_RAD", 1.0)
            if _env_flag("YAW_DRIFT_TERM")
            else 0.0
        ),
        yaw_persist_steps=int(_env_float("YAW_DRIFT_PERSIST_STEPS", 15)),
        yaw_drift_term_penalty=abs(
            _env_float(
                "YAW_DRIFT_TERM_PENALTY",
                abs(_env_float("FALL_TERM_PENALTY", 0.0)),
            )
        ),
        hold_balance_v2=_env_float("HOLD_BALANCE_V2", 0.0),
        v2_ad_min=_env_float("HOLD_BALANCE_AD_MIN", 0.045),
        v2_ad_max=_env_float("HOLD_BALANCE_AD_MAX", 0.30),
        v2_alive_window=int(_env_float("V2_ALIVE_WINDOW", 15)),
        v2_posture_coef=_env_float("HOLD_BALANCE_V2_POSTURE_COEF", -10.0),
        v2_dofvel_min=_env_float("HOLD_BALANCE_V2_DOFVEL_MIN", 0.05),
        v2_dofvel_max=_env_float("HOLD_BALANCE_V2_DOFVEL_MAX", 2.0),
    )


# =============================================================================
# Partial-exposure action-delay DR (DELAY-DR, Gary 2026-07-10)
# =============================================================================


@dataclass
class ActionDelaySettings:
    """Env-var gates for PARTIAL-EXPOSURE action-delay DR.

    History (reconciled 2026-07-10): full delay machinery EXISTS and is wired
    (DelayDomainRandomizationConfig + per-env ring buffers in base_env/env.py,
    night2 2026-07-03/05) — the superdr docstring's "unavailable" note simply
    predates it. Gary's global-delay run (h1_2_laneb_delaydr_scratch v1)
    regressed (reward 85->36, EV 0.79->0.63): full-range (0,2) action+obs
    delay on ALL envs from epoch 0 broke early credit assignment before a
    competent gait existed (STATUS.md lane B; worklog: combined formulation,
    not "delay is hard"). ramp_epochs was the first fix; THIS is the
    complementary lever: only a FRACTION of envs get delayed (bootstrap on
    clean envs / transfer-robustness pattern), assignment fixed per env per
    episode, resampled at reset.

    Gates: ACTION_DELAY_DR=1 (off default), ACTION_DELAY_FRAC (0.25),
    ACTION_DELAY_STEPS_MAX (2; delayed envs sample uniform {1..max}).
    Observation delay is untouched. Instrumentation (the conviction tool):
    per-env delay assignment + per-delay-bin reward / termination-rate /
    action-delta extras -> answers "is delay what blows things up" with data.
    """

    enabled: bool = False
    frac: float = 0.25
    steps_max: int = 2


def load_action_delay_settings() -> ActionDelaySettings:
    """Parse the DELAY-DR env-var gates. Off by default."""
    return ActionDelaySettings(
        enabled=_env_flag("ACTION_DELAY_DR"),
        frac=min(1.0, max(0.0, _env_float("ACTION_DELAY_FRAC", 0.25))),
        steps_max=max(1, int(_env_float("ACTION_DELAY_STEPS_MAX", 2))),
    )


# =============================================================================
# Reference-stillness tracker
# =============================================================================


class RefStillnessTracker:
    """Per-env reference-stillness mask from reference-state deltas.

    Delta-based on purpose (NOT the motion-lib stored velocities): a frozen
    reference clock (training freeze augmentation / eval hold segments)
    replays the SAME frame each step — its stored instantaneous velocities
    can be arbitrarily large (frozen mid-jog), but the frame-to-frame delta
    is exactly zero. Naturally-still clip segments (pauses class) also have
    near-zero deltas. Both hold flavors are caught by one signal.

    An env counts as still on a step when reference root linear speed, root
    angular speed and max |dof| speed (all measured as per-step deltas / dt)
    are below their epsilons, or when an explicit freeze mask says the
    reference clock is held. The output mask fires after ``window``
    consecutive still steps (debounces instantaneously-slow frames, e.g.
    gait turnaround points) and drops immediately when the reference moves.

    Episode resets are handled implicitly: a resampled motion jumps the
    reference discontinuously, so the streak resets to zero on the first
    post-reset step. The first update after construction returns all-False.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        root_eps: float = 0.05,
        rot_eps: float = 0.30,
        dof_eps: float = 0.20,
        window: int = 5,
    ):
        self.root_eps = root_eps
        self.rot_eps = rot_eps
        self.dof_eps = dof_eps
        self.window = max(1, int(window))
        self._streak = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._prev_root_pos: Optional[Tensor] = None
        self._prev_root_rot: Optional[Tensor] = None
        self._prev_dof_pos: Optional[Tensor] = None

    def update(
        self,
        ref_root_pos: Tensor,
        ref_root_rot: Tensor,
        ref_dof_pos: Tensor,
        dt: float,
        freeze_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Advance one env step; returns Bool[num_envs] stillness mask.

        Args:
            ref_root_pos: Reference root position [num_envs, 3].
            ref_root_rot: Reference root quaternion (xyzw) [num_envs, 4].
            ref_dof_pos: Reference dof positions [num_envs, num_dofs].
            dt: Env timestep (s).
            freeze_mask: Optional Bool[num_envs] — explicit reference-clock
                freeze (counts as still regardless of deltas).
        """
        if self._prev_root_pos is None:
            self._prev_root_pos = ref_root_pos.clone()
            self._prev_root_rot = ref_root_rot.clone()
            self._prev_dof_pos = ref_dof_pos.clone()
            return torch.zeros_like(self._streak, dtype=torch.bool)

        inv_dt = 1.0 / dt
        root_speed = torch.norm(ref_root_pos - self._prev_root_pos, dim=-1) * inv_dt
        # Quaternion geodesic angle between consecutive reference root frames.
        quat_dot = (ref_root_rot * self._prev_root_rot).sum(dim=-1).abs().clamp(max=1.0)
        ang_speed = 2.0 * torch.acos(quat_dot) * inv_dt
        dof_speed = (
            (ref_dof_pos - self._prev_dof_pos).abs().amax(dim=-1) * inv_dt
        )

        inst_still = (
            (root_speed < self.root_eps)
            & (ang_speed < self.rot_eps)
            & (dof_speed < self.dof_eps)
        )
        if freeze_mask is not None:
            inst_still = inst_still | freeze_mask.bool()

        self._streak = torch.where(
            inst_still, self._streak + 1, torch.zeros_like(self._streak)
        )
        self._prev_root_pos.copy_(ref_root_pos)
        self._prev_root_rot.copy_(ref_root_rot)
        self._prev_dof_pos.copy_(ref_dof_pos)

        return self._streak >= self.window


# =============================================================================
# Balance-bonus reward kernel
# =============================================================================


def compute_hold_balance_v2(
    reference_still_mask: Tensor,
    root_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    anchor_idx: int,
    sim_contacts: Tensor,
    historical_actions: Tensor,
    dof_vel: Tensor,
    left_foot_body_ids: Tensor,
    right_foot_body_ids: Tensor,
    ad_min: float = 0.005,
    ad_max: float = 0.15,
    posture_coefficient: float = -10.0,
    dofvel_min: float = 0.05,
    dofvel_max: float = 2.0,
) -> Tensor:
    """HOLD-BALANCE V2: posture-relative + living-band quiet-stance bonus.

    bonus = still * feet_planted * exp(posture_coef * rel_tilt^2) * band(alive)

    - rel_tilt: quaternion angle between CURRENT pelvis rot and the
      REFERENCE pelvis rot (bent-waist holds pay when matching the bent ref;
      fixes the v1 absolute-upright blindspot, 18:26 entry).
    - band(alive): LIVING-BAND on actuation — 0 below ``ad_min`` (ZERO PAY AT
      ZERO MOTION, BY CONSTRUCTION: kills the rigid pole), 1 inside
      [ad_min, ad_max], smooth exponential decay above (jitter pays less).
      alive = mean |action_t - action_{t-1}| from the state-history buffer;
      FALLBACK when history is unavailable: mean |dof_vel| against the
      [dofvel_min, dofvel_max] band (same shape, different units).
    - still / feet_planted: as v1 (frozen-clock-safe delta mask; both feet).

    Returns bonus in [0, 1] per env [num_envs].
    """
    n = root_rot.shape[0]
    dev = root_rot.device
    if reference_still_mask is None:
        return torch.zeros(n, device=dev)

    contacts = sim_contacts.bool()
    planted = (
        contacts[:, left_foot_body_ids].any(dim=-1)
        & contacts[:, right_foot_body_ids].any(dim=-1)
    )

    # Posture-relative tilt: angle between current and reference pelvis quats.
    q_ref = ref_rigid_body_rot[:, anchor_idx, :]
    dot = (root_rot * q_ref).sum(dim=-1).abs().clamp(max=1.0)
    rel_tilt = 2.0 * torch.acos(dot)  # [0, pi]
    posture = torch.exp(posture_coefficient * rel_tilt * rel_tilt)

    # Living band on actuation.
    alive = None
    if historical_actions is not None and historical_actions.shape[1] >= 2:
        alive = (
            historical_actions[:, -1] - historical_actions[:, -2]
        ).abs().mean(dim=-1)
        lo, hi = ad_min, ad_max
    if alive is None:
        alive = dof_vel.abs().mean(dim=-1)
        lo, hi = dofvel_min, dofvel_max
    above = torch.exp(-4.0 * ((alive - hi).clamp(min=0.0) / max(hi, 1e-6)) ** 2)
    band = torch.where(alive < lo, torch.zeros_like(alive), above)

    return reference_still_mask.float() * planted.float() * posture * band


def make_hold_balance_v2_kernel():
    """Stateful HOLD-BALANCE V2 kernel with a NOISE-CANCELLING aliveness
    tracker (v2.1, Gary 2026-07-11 ruling on the stochastic-noise loophole).

    THE LOOPHOLE (measured): policy sigma = exp(logstd) ~ 0.103, so the pure
    exploration-noise per-step action delta floor is 1.128*sigma ~ 0.116 —
    INSIDE the original living band [0.005, 0.15]. Training-time noise alone
    farmed full band pay (TB: raw_r/hold_balance_v2 ~ 0.95x disc) while the
    deterministic eval mean stayed rigid (holds at ad 0.0005). With history
    depth 1 the deployed fallback channel was noisy dof_vel — same loophole.

    THE FIX: aliveness = WINDOWED MEAN DISPLACEMENT of the action stream:
        alive = mean_dofs | mean(a[t-K+1..t]) - mean(a[t-2K+1..t-K]) |
    Zero-mean noise cancels as 1/sqrt(K): floor = sigma * 2/sqrt(pi*K)
    (~0.030 at K=15, sigma 0.103), while genuine slow posture adjustment
    (a moving action MEAN) passes through untouched. Band defaults are
    calibrated to the K=15 floor: ad_min 0.045 = 1.5x floor
    (belt-and-suspenders per ruling track (a)), ad_max 0.30.

    Ring buffer [num_envs, 2K, num_dofs] lives in the closure (same pattern
    as make_yaw_drift_kernel; training-side only, never ONNX-exported).
    WARMUP: until 2K pushes have accumulated (startup or buffer realloc)
    aliveness is 0 => zero pay — a fresh buffer can never mint free money.
    """
    state = {"ring": None, "idx": 0, "filled": 0}

    def compute(
        reference_still_mask: Optional[Tensor],
        root_rot: Tensor,
        ref_rigid_body_rot: Tensor,
        anchor_idx: int,
        sim_contacts: Tensor,
        historical_actions: Optional[Tensor],
        dof_vel: Tensor,
        left_foot_body_ids: Tensor,
        right_foot_body_ids: Tensor,
        ad_min: float = 0.045,
        ad_max: float = 0.30,
        posture_coefficient: float = -10.0,
        alive_window: int = 15,
        dofvel_min: float = 0.05,
        dofvel_max: float = 2.0,
    ) -> Tensor:
        n = root_rot.shape[0]
        dev = root_rot.device
        if reference_still_mask is None:
            return torch.zeros(n, device=dev)

        # --- noise-cancelling aliveness from the action stream ------------
        last_action = None
        if historical_actions is not None and historical_actions.shape[1] >= 1:
            last_action = historical_actions[:, -1]
        if last_action is None:
            # No action stream at all: fall back to the legacy dof_vel band
            # (noisy channel — better than nothing, but flagged in the arm
            # line; not expected in production).
            return compute_hold_balance_v2(
                reference_still_mask, root_rot, ref_rigid_body_rot,
                anchor_idx, sim_contacts, None, dof_vel,
                left_foot_body_ids, right_foot_body_ids,
                ad_min=ad_min, ad_max=ad_max,
                posture_coefficient=posture_coefficient,
                dofvel_min=dofvel_min, dofvel_max=dofvel_max,
            )

        K = int(alive_window)
        ring = state["ring"]
        if (
            ring is None
            or ring.shape[0] != n
            or ring.shape[1] != 2 * K
            or ring.shape[2] != last_action.shape[-1]
        ):
            ring = torch.zeros(
                n, 2 * K, last_action.shape[-1], device=dev,
                dtype=last_action.dtype,
            )
            state["ring"] = ring
            state["idx"] = 0
            state["filled"] = 0
        ring[:, state["idx"] % (2 * K)] = last_action
        state["idx"] += 1
        state["filled"] = min(state["filled"] + 1, 2 * K)

        if state["filled"] < 2 * K:
            alive = torch.zeros(n, device=dev)
        else:
            # Ring order does not matter for the two-half split as long as
            # halves are contiguous in TIME: reconstruct chronological order.
            pos = state["idx"] % (2 * K)  # oldest slot
            chron = torch.cat([ring[:, pos:], ring[:, :pos]], dim=1)
            older = chron[:, :K].mean(dim=1)
            newer = chron[:, K:].mean(dim=1)
            alive = (newer - older).abs().mean(dim=-1)

        contacts = sim_contacts.bool()
        planted = (
            contacts[:, left_foot_body_ids].any(dim=-1)
            & contacts[:, right_foot_body_ids].any(dim=-1)
        )
        q_ref = ref_rigid_body_rot[:, anchor_idx, :]
        dot = (root_rot * q_ref).sum(dim=-1).abs().clamp(max=1.0)
        rel_tilt = 2.0 * torch.acos(dot)
        posture = torch.exp(posture_coefficient * rel_tilt * rel_tilt)

        above = torch.exp(
            -4.0 * ((alive - ad_max).clamp(min=0.0) / max(ad_max, 1e-6)) ** 2
        )
        band = torch.where(alive < ad_min, torch.zeros_like(alive), above)

        return reference_still_mask.float() * planted.float() * posture * band

    return compute


def compute_hold_balance_bonus(
    reference_still_mask: Tensor,
    root_rot: Tensor,
    root_vel: Tensor,
    sim_contacts: Tensor,
    left_foot_body_ids: Tensor,
    right_foot_body_ids: Tensor,
    vel_coefficient: float = -4.0,
    upright_coefficient: float = -10.0,
) -> Tensor:
    """Quiet-stance bonus, paid ONLY during reference-still windows.

    bonus = still * feet_planted * exp(up_coef * (1 - up_z)^2)
                                 * exp(vel_coef * |root_vel|^2)

    - ``still``: the reference-stillness mask (zero income outside holds).
    - ``feet_planted``: at least one collision body of EACH foot in contact
      (single-support stepping-in-place earns nothing).
    - upright factor: exp-kernel on pelvis-up tilt (up_z = world-z component
      of the root frame's up axis).
    - velocity factor: exp-kernel on pelvis speed — small corrective motion
      is barely taxed, drifting/jogging pays ~nothing.

    Deliberately contains NO actuation term: it never pays for rigidity.

    Args:
        reference_still_mask: Bool[num_envs] or None (no mask source -> 0).
        root_rot: Root quaternion (xyzw) [num_envs, 4].
        root_vel: Root linear velocity [num_envs, 3].
        sim_contacts: Per-body contact flags/forces bool-able [num_envs, num_bodies].
        left_foot_body_ids: Long tensor of left-foot body indices.
        right_foot_body_ids: Long tensor of right-foot body indices.
        vel_coefficient: Exp coefficient on squared root speed (negative).
        upright_coefficient: Exp coefficient on squared tilt (negative).

    Returns:
        Bonus in [0, 1] per env [num_envs].
    """
    if reference_still_mask is None:
        return torch.zeros(root_vel.shape[0], device=root_vel.device)

    contacts = sim_contacts.bool()
    planted = (
        contacts[:, left_foot_body_ids].any(dim=-1)
        & contacts[:, right_foot_body_ids].any(dim=-1)
    )

    # up_z: world-z component of the body-frame up axis rotated by root_rot
    # (xyzw). For q=(x,y,z,w): R(q)e_z . e_z = 1 - 2*(x^2 + y^2).
    qx, qy = root_rot[:, 0], root_rot[:, 1]
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    tilt = (1.0 - up_z).clamp(min=0.0)
    upright = torch.exp(upright_coefficient * tilt * tilt)

    speed_sq = (root_vel * root_vel).sum(dim=-1)
    quiet = torch.exp(vel_coefficient * speed_sq)

    return reference_still_mask.float() * planted.float() * upright * quiet


# =============================================================================
# Wrist direction-agreement reward kernel (WRIST-DIR)
# =============================================================================


def _wrist_dir_cos_and_active(
    rigid_body_vel: Tensor,
    ref_rigid_body_vel: Tensor,
    wrist_body_ids: Tensor,
    ref_speed_eps: float,
):
    """Shared core: raw cosine + active mask + ref speed per wrist.

    Returns (cos [envs, W], active Bool [envs, W], ref_speed [envs, W]).
    cos is 0 where the POLICY wrist is ~still (norm clamped), raw signed
    elsewhere; ``active`` gates on REFERENCE wrist speed only — direction is
    undefined on still reference wrists, so holds are inert by construction
    (composes with, never fights, the HOLD-FIX still-mask).
    """
    v_pol = rigid_body_vel[:, wrist_body_ids]  # [envs, W, 3]
    v_ref = ref_rigid_body_vel[:, wrist_body_ids]
    ref_speed = torch.norm(v_ref, dim=-1)  # [envs, W]
    active = ref_speed > ref_speed_eps
    denom = (torch.norm(v_pol, dim=-1) * ref_speed).clamp(min=1e-6)
    cos = (v_pol * v_ref).sum(dim=-1) / denom
    return cos, active, ref_speed


def compute_wrist_dir_reward(
    rigid_body_vel: Tensor,
    ref_rigid_body_vel: Tensor,
    wrist_body_ids: Tensor,
    ref_speed_eps: float = 0.1,
    vel_weight_vmax: float = 0.0,
) -> Tensor:
    """Wrist direction-agreement reward (Gary 2026-07-10, dir_cos weak axis).

    reward = mean over ACTIVE wrists of relu(cos(v_wrist_policy, v_wrist_ref))
    where a wrist is active iff ||v_wrist_ref|| > ref_speed_eps (~0.1 m/s):
    direction is undefined on still reference wrists, so the term is INERT
    during holds by construction. Envs with no active wrist earn 0.

    Shaping: relu(cos) — zero floor at orthogonal-or-worse. The affine
    alternative (cos+1)/2 was considered and rejected as the default: it pays
    0.5/wrist for orthogonal (or numerically-still policy) wrist motion,
    i.e. free income during every active-reference segment regardless of
    directional correctness; relu pays only for genuine agreement.

    Optional |v_ref|-proportional weighting (``vel_weight_vmax > 0``): each
    active wrist's contribution is scaled by min(||v_ref||, vmax)/vmax while
    the normalizer stays the ACTIVE COUNT — fast reference wrist motion pays
    proportionally more (not renormalized away).

    Args:
        rigid_body_vel: Policy body velocities [num_envs, num_bodies, 3].
        ref_rigid_body_vel: Reference body velocities [num_envs, num_bodies, 3].
        wrist_body_ids: Long tensor of wrist body indices [W].
        ref_speed_eps: Reference wrist speed activity gate (m/s).
        vel_weight_vmax: 0 = off; >0 = proportional-weighting speed cap (m/s).

    Returns:
        Reward in [0, 1] per env [num_envs].
    """
    cos, active, ref_speed = _wrist_dir_cos_and_active(
        rigid_body_vel, ref_rigid_body_vel, wrist_body_ids, ref_speed_eps
    )
    shaped = cos.clamp(min=0.0)  # relu
    contrib = shaped * active.float()
    if vel_weight_vmax > 0.0:
        contrib = contrib * (ref_speed / vel_weight_vmax).clamp(max=1.0)
    return contrib.sum(dim=-1) / active.float().sum(dim=-1).clamp(min=1.0)


def compute_wrist_dir_extras(
    rigid_body_vel: Tensor,
    ref_rigid_body_vel: Tensor,
    wrist_body_ids: Tensor,
    ref_speed_eps: float = 0.1,
):
    """TB watch signals: raw (signed) dir-cos mean over active wrists + active
    fraction, per env. Raw cos (not relu'd) so the axis is watchable through
    zero — mirrors the suite's dir_cos metric (weak axis 0.61)."""
    cos, active, _ = _wrist_dir_cos_and_active(
        rigid_body_vel, ref_rigid_body_vel, wrist_body_ids, ref_speed_eps
    )
    dir_cos = (cos * active.float()).sum(dim=-1) / active.float().sum(
        dim=-1
    ).clamp(min=1.0)
    return dir_cos, active.float().mean(dim=-1)


class WristDirTracker:
    """Per-env WRIST-DIR state + per-step outputs (reward, dir_cos, active).

    PRIMARY VARIANT — WINDOWED DISPLACEMENT (Gary refinement 2026-07-10:
    step-to-step velocity direction is too noisy):
        cos( x_wrist(t) - x_wrist(t-K),  x_ref(t) - x_ref(t-K) )
    with K = round(window_s / dt) control steps (~0.3 s default). Keeps a
    circular buffer of the last K wrist positions (policy + reference,
    [envs, K, W, 3] each — only the K-back endpoint is read). A wrist is
    ACTIVE iff the REFERENCE wrist traveled more than ``disp_eps`` (0.02 m
    default) over the window AND the env has ≥K post-reset frames (warm):
    direction is undefined on still reference wrists, so holds are inert by
    construction — composes with the HOLD-FIX still-mask, never fights it.

    Episode resets are detected via ``progress_buf <= previous`` (the
    FeetApexHeightReward pattern); reset envs go cold (count=0) so stale
    pre-reset positions can never produce cross-episode displacement ghosts.

    SUB-FLAG VARIANT (``instantaneous=True``): the original per-step velocity
    cosine, gated on ||v_ref|| > speed_eps — kept for comparison.

    Outputs refreshed by ``update()`` each step:
      .reward      [envs] — mean over active wrists of relu(cos), optionally
                   ref-speed-proportionally weighted (vmax cap); 0 when no
                   wrist is active.
      .dir_cos     [envs] — RAW signed cos mean over active wrists (TB watch).
      .active_frac [envs] — fraction of wrists active.
    """

    def __init__(
        self,
        num_envs: int,
        num_wrists: int,
        device: torch.device,
        window_steps: int,
        window_s: float,
        disp_eps: float = 0.02,
        vel_weight_vmax: float = 0.0,
        instantaneous: bool = False,
        speed_eps: float = 0.1,
        subsample: int = 1,
        shaping: str = "dir_cos",
    ):
        # shaping: "dir_cos" (wrist-dir: relu(cos) direction agreement) or
        # "gain_proj" (root-gain: clamp(proj_gain, 0, 1) where proj_gain =
        # (dx_pol . dx_ref) / ||dx_ref||^2 — the policy's displacement
        # projected on the reference direction, as a fraction of the
        # reference's displacement. 1.0 = matched progress; overshoot earns
        # NO bonus (clamp), backward/negative projection earns 0).
        assert shaping in ("dir_cos", "gain_proj"), shaping
        self.shaping = shaping
        self.K = max(1, int(window_steps))
        # Subsample (Gary refinement #2): store every M-th frame into a
        # ceil(K/M)-slot ring — only the window-back endpoint is read, so
        # temporal resolution of the LOOKBACK (not the current endpoint)
        # quantizes to M steps. Bytes per env.
        self.M = max(1, int(subsample))
        self.slots = max(1, -(-self.K // self.M))  # ceil(K/M)
        self.window_s = max(window_s, 1e-6)
        self.disp_eps = disp_eps
        self.vel_weight_vmax = vel_weight_vmax
        self.instantaneous = instantaneous
        self.speed_eps = speed_eps
        self._buf_pol = torch.zeros(
            num_envs, self.slots, num_wrists, 3, device=device
        )
        self._buf_ref = torch.zeros(
            num_envs, self.slots, num_wrists, 3, device=device
        )
        self._count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._idx = 0  # global circular slot (all envs step in lockstep)
        self._step = 0  # global step counter for the M-subsample cadence
        self._prev_progress: Optional[Tensor] = None
        self.reward = torch.zeros(num_envs, device=device)
        self.dir_cos = torch.zeros(num_envs, device=device)
        self.gain = torch.zeros(num_envs, device=device)  # raw uncapped proj gain
        self.active_frac = torch.zeros(num_envs, device=device)

    def update(
        self,
        wrist_pos: Tensor,
        ref_wrist_pos: Tensor,
        wrist_vel: Tensor,
        ref_wrist_vel: Tensor,
        progress_buf: Tensor,
    ) -> Tensor:
        """Advance one env step; refreshes .reward/.dir_cos/.active_frac.

        Args:
            wrist_pos: Policy wrist positions [envs, W, 3].
            ref_wrist_pos: Reference wrist positions [envs, W, 3].
            wrist_vel: Policy wrist velocities [envs, W, 3] (instant variant).
            ref_wrist_vel: Reference wrist velocities [envs, W, 3] (instant).
            progress_buf: Episode progress counters [envs].
        """
        # Reset detection (FeetApexHeightReward pattern): progress that did
        # not strictly increase = env reset since last call -> go cold.
        if self._prev_progress is not None:
            reset_mask = progress_buf <= self._prev_progress
            if reset_mask.any():
                self._count[reset_mask] = 0
        self._prev_progress = progress_buf.clone()

        if self.instantaneous:
            cos, active, ref_speed = _wrist_dir_cos_and_active(
                wrist_vel, ref_wrist_vel,
                torch.arange(wrist_vel.shape[1], device=wrist_vel.device),
                self.speed_eps,
            )
            speed = ref_speed
            # gain_proj on velocities (instantaneous variant of the gain).
            gain_raw = (wrist_vel * ref_wrist_vel).sum(dim=-1) / (
                ref_speed * ref_speed
            ).clamp(min=1e-9)
        else:
            # Slot self._idx holds the OLDEST stored sample (~window ago for
            # warm envs); read it BEFORE any overwrite this step.
            dx_pol = wrist_pos - self._buf_pol[:, self._idx]
            dx_ref = ref_wrist_pos - self._buf_ref[:, self._idx]
            warm = (self._count >= self.slots).unsqueeze(-1)
            ref_dist = torch.norm(dx_ref, dim=-1)
            active = (ref_dist > self.disp_eps) & warm
            denom = (torch.norm(dx_pol, dim=-1) * ref_dist).clamp(min=1e-9)
            dot = (dx_pol * dx_ref).sum(dim=-1)
            cos = dot / denom
            # Projected displacement gain: fraction of the reference's
            # displacement the policy achieved ALONG the reference direction.
            gain_raw = dot / (ref_dist * ref_dist).clamp(min=1e-9)
            speed = ref_dist / self.window_s
            # Push frame t on the M-subsample cadence only.
            if self._step % self.M == 0:
                self._buf_pol[:, self._idx] = wrist_pos
                self._buf_ref[:, self._idx] = ref_wrist_pos
                self._idx = (self._idx + 1) % self.slots
                self._count += 1
            self._step += 1

        active_f = active.float()
        if self.shaping == "gain_proj":
            # clamp(gain, 0, 1): matched progress pays 1, overshoot earns no
            # bonus, backward/negative projection earns 0.
            shaped = gain_raw.clamp(min=0.0, max=1.0)
        else:
            shaped = cos.clamp(min=0.0)  # relu direction agreement
        contrib = shaped * active_f
        if self.vel_weight_vmax > 0.0:
            contrib = contrib * (speed / self.vel_weight_vmax).clamp(max=1.0)
        norm = active_f.sum(dim=-1).clamp(min=1.0)
        self.reward = contrib.sum(dim=-1) / norm
        self.dir_cos = (cos * active_f).sum(dim=-1) / norm
        self.gain = (gain_raw * active_f).sum(dim=-1) / norm
        self.active_frac = active_f.mean(dim=-1)
        return self.reward


def passthrough_root_gain_reward(root_gain_reward: Tensor) -> Tensor:
    """Identity kernel for the injected ROOT-GAIN reward component.

    The stateful windowed-displacement gain runs once per step in the env's
    ``_apply_hold_fix`` (WristDirTracker, shaping='gain_proj', body set =
    root); the result is exposed on ``EnvContext.root_gain_reward`` and this
    kernel routes it through the standard reward plumbing."""
    return root_gain_reward


def passthrough_wrist_dir_reward(wrist_dir_reward: Tensor) -> Tensor:
    """Identity kernel for the injected WRIST-DIR reward component.

    The actual computation is stateful (K-frame ring buffer) and runs once
    per step in the env's ``_apply_hold_fix`` (WristDirTracker); the result
    is exposed on ``EnvContext.wrist_dir_reward`` and this kernel routes it
    through the standard reward-combining plumbing (weight, TB raw_r/scaled_r
    logging)."""
    return wrist_dir_reward


# =============================================================================
# Env-side wiring helpers (called by BaseEnv)
# =============================================================================


def resolve_foot_body_ids(
    body_names: List[str],
    common_naming: Optional[Dict[str, List[str]]],
    device: torch.device,
) -> Optional[Dict[str, Tensor]]:
    """Resolve left/right foot body indices from the robot config.

    Uses ``common_naming_to_robot_body_names['all_{left,right}_foot_bodies']``
    when available, else falls back to a left/right name-substring match over
    bodies containing 'foot' or 'ankle'. Returns None when feet cannot be
    resolved (the balance bonus then refuses to arm).
    """
    left_names: List[str] = []
    right_names: List[str] = []
    if common_naming:
        left_names = [
            n for n in common_naming.get("all_left_foot_bodies", [])
            if n in body_names
        ]
        right_names = [
            n for n in common_naming.get("all_right_foot_bodies", [])
            if n in body_names
        ]
    if not left_names or not right_names:
        feetish = [n for n in body_names if "foot" in n.lower() or "ankle" in n.lower()]
        left_names = [n for n in feetish if "left" in n.lower() or n.lower().startswith("l_")]
        right_names = [n for n in feetish if "right" in n.lower() or n.lower().startswith("r_")]
    if not left_names or not right_names:
        return None
    return {
        "left": torch.tensor(
            [body_names.index(n) for n in left_names], dtype=torch.long, device=device
        ),
        "right": torch.tensor(
            [body_names.index(n) for n in right_names], dtype=torch.long, device=device
        ),
    }


def resolve_wrist_body_ids(
    body_names: List[str],
    common_naming: Optional[Dict[str, List[str]]],
    device: torch.device,
) -> Optional[Tensor]:
    """Resolve wrist/hand body indices from the robot config.

    Uses ``common_naming_to_robot_body_names['all_{left,right}_hand_bodies']``
    when available (h1_2: left/right_wrist_yaw_link), else falls back to a
    name-substring match on 'wrist' or 'hand'. Returns None when unresolvable
    (the wrist-dir reward then refuses to arm)."""
    names: List[str] = []
    if common_naming:
        for key in ("all_left_hand_bodies", "all_right_hand_bodies"):
            names += [n for n in common_naming.get(key, []) if n in body_names]
    if not names:
        names = [
            n for n in body_names
            if "wrist" in n.lower() or "hand" in n.lower()
        ]
    if not names:
        return None
    return torch.tensor(
        [body_names.index(n) for n in names], dtype=torch.long, device=device
    )


def compute_xy_drift_term(
    current_anchor_pos,
    ref_rigid_body_pos,
    ref_rigid_body_vel,
    reference_still_mask,
    anchor_idx,
    drift_threshold: float = 0.55,
    move_eps: float = 0.3,
):
    """XY-drift early termination (component [8], Gary 2026-07-10).

    Terminates envs whose horizontal root (anchor) error vs the reference
    exceeds ``drift_threshold`` — but ONLY on moving-reference frames:
    ref anchor horizontal speed must exceed ``move_eps`` AND the delta-based
    reference-still mask must be False (frozen clocks replay stale velocities,
    so the velocity gate alone is not freeze-safe — the still mask is).
    Holds / frozen / pauses are therefore exempt by construction and this
    termination cannot fight the hold-balance work.

    Returns bool tensor [num_envs] (True = terminate).
    """
    xy_err = (
        current_anchor_pos[:, :2] - ref_rigid_body_pos[:, anchor_idx, :2]
    ).norm(dim=-1)
    ref_speed = ref_rigid_body_vel[:, anchor_idx, :2].norm(dim=-1)
    moving = ref_speed > move_eps
    if reference_still_mask is not None:
        moving = moving & (~reference_still_mask.bool())
    return (xy_err > drift_threshold) & moving


def compute_xy_drift_penalty(
    current_anchor_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    ref_rigid_body_vel: Tensor,
    reference_still_mask: Optional[Tensor],
    anchor_idx: int,
    drift_threshold: float = 0.55,
    move_eps: float = 0.3,
) -> Tensor:
    """XY-drift terminal penalty indicator (component [8b]).

    Returns 1.0 for envs firing the EXACT ``compute_xy_drift_term`` condition
    on this step, else 0.0. Apply a NEGATIVE weight in the component config
    (framework applies weight + grace zeroing; raw_r stays the indicator, so
    the fall-penalty identity law applies verbatim:
    env/termination/xy_drift_mean == env/raw_r/hold_fix_xy_drift_penalty_mean).

    Rationale (Gary 2026-07-10): the drift fence must be priced like a fall —
    otherwise drifting out is a discount exit versus falling
    (episode-only vs episode + FALL_TERM_PENALTY).
    """
    return compute_xy_drift_term(
        current_anchor_pos=current_anchor_pos,
        ref_rigid_body_pos=ref_rigid_body_pos,
        ref_rigid_body_vel=ref_rigid_body_vel,
        reference_still_mask=reference_still_mask,
        anchor_idx=anchor_idx,
        drift_threshold=drift_threshold,
        move_eps=move_eps,
    ).float()


def make_yaw_drift_kernel(as_penalty: bool = False):
    """Build a stateful yaw-drift kernel (component [8c]).

    Returns a compute function holding a per-env consecutive-violation
    counter in a closure. The TERMINATION and the PENALTY each get their OWN
    factory instance: the update is deterministic in the step inputs, so the
    two counters stay bit-identical as long as each component is evaluated
    exactly once per env step — giving the fall/xy identity law for free:
    ``env/termination/yaw_drift_mean == env/raw_r/hold_fix_yaw_drift_penalty_mean``.

    Counter update: ``count = (count + 1) * violating`` — increments on
    violating frames, hard-zeroes otherwise (so a post-termination reset
    clears it on the first clean frame). Fires when
    ``count >= persist_steps``.

    Condition: |root yaw − ref anchor yaw| (wrapped) > ``yaw_threshold``,
    AND the delta-based still mask says the reference is LIVE. NO ref
    yaw-rate gate: straight-walk heading drift must stay inside the fence.
    Training-side only (stripped for evals); stateful closure is fine — this
    component is never ONNX-exported.
    """
    state = {"count": None}

    def compute_yaw_drift(
        reference_still_mask: Optional[Tensor],
        root_rot: Tensor,
        ref_rigid_body_rot: Tensor,
        anchor_idx: int,
        yaw_threshold: float = 1.0,
        persist_steps: int = 15,
    ) -> Tensor:
        from protomotions.utils.rotations import calc_heading

        q_ref = ref_rigid_body_rot[:, anchor_idx, :]
        yaw_err = calc_heading(root_rot, True) - calc_heading(q_ref, True)
        yaw_err = torch.remainder(yaw_err + torch.pi, 2 * torch.pi) - torch.pi
        violating = yaw_err.abs() > yaw_threshold
        if reference_still_mask is not None:
            violating = violating & (~reference_still_mask.bool())

        count = state["count"]
        if count is None or count.shape[0] != root_rot.shape[0]:
            count = torch.zeros(
                root_rot.shape[0], device=root_rot.device
            )
        count = (count + 1.0) * violating.float()
        state["count"] = count

        fired = count >= float(persist_steps)
        return fired.float() if as_penalty else fired

    return compute_yaw_drift


def setup_hold_fix_components(
    settings: HoldFixSettings,
    reward_components: Dict,
    termination_components: Dict,
    body_names: List[str],
    common_naming: Optional[Dict[str, List[str]]],
    device: torch.device,
) -> None:
    """Inject the env-gated HOLD-FIX reward components into the config.

    Mutates ``reward_components`` in place. Pure config-level function so it
    is unit-testable without a simulator. Prints ``[hold-fix]`` launch lines
    for dump-verify.
    """
    if settings.fall_term_penalty > 0.0:
        from protomotions.envs.component_factories import fall_penalty_factory

        # Mirror the fall-termination threshold when the recipe has one.
        height_threshold = 0.25
        fall_term = termination_components.get("fall")
        if fall_term is not None:
            params = (
                fall_term.get_params()
                if hasattr(fall_term, "get_params")
                else dict(fall_term)
            )
            height_threshold = float(params.get("threshold", height_threshold))
        reward_components["hold_fix_fall_penalty"] = fall_penalty_factory(
            weight=-settings.fall_term_penalty,
            height_threshold=height_threshold,
            zero_during_grace_period=True,
        )
        print(
            f"[hold-fix] FALL_TERM_PENALTY armed: reward component "
            f"'hold_fix_fall_penalty' weight={-settings.fall_term_penalty} "
            f"height_threshold={height_threshold} (fires on the fall-termination "
            f"condition; TB tags raw_r/scaled_r/hold_fix_fall_penalty)"
        )

    if settings.hold_balance_bonus > 0.0:
        from protomotions.envs.component_factories import hold_balance_bonus_factory

        foot_ids = resolve_foot_body_ids(body_names, common_naming, device)
        if foot_ids is None:
            print(
                "[hold-fix] WARNING: HOLD_BALANCE_BONUS set but left/right foot "
                "bodies could not be resolved from the robot config — balance "
                "bonus NOT armed."
            )
        else:
            reward_components["hold_balance"] = hold_balance_bonus_factory(
                weight=settings.hold_balance_bonus,
                left_foot_body_ids=foot_ids["left"],
                right_foot_body_ids=foot_ids["right"],
                vel_coefficient=settings.balance_vel_coef,
                upright_coefficient=settings.balance_up_coef,
            )
            print(
                f"[hold-fix] HOLD_BALANCE_BONUS armed: reward component "
                f"'hold_balance' weight={settings.hold_balance_bonus} "
                f"vel_coef={settings.balance_vel_coef} "
                f"up_coef={settings.balance_up_coef} "
                f"left_foot_ids={foot_ids['left'].tolist()} "
                f"right_foot_ids={foot_ids['right'].tolist()} "
                f"(TB tags raw_r/scaled_r/hold_balance)"
            )

    if settings.hold_balance_v2 > 0.0:
        from protomotions.envs.context_views import EnvContext
        from protomotions.envs.mdp_component import MdpComponent

        v2_foot_ids = resolve_foot_body_ids(body_names, common_naming, device)
        if v2_foot_ids is None:
            print(
                "[hold-fix] WARNING: HOLD_BALANCE_V2 set but foot bodies "
                "could not be resolved — v2 balance NOT armed."
            )
        else:
            reward_components["hold_balance_v2"] = MdpComponent(
                compute_func=make_hold_balance_v2_kernel(),
                dynamic_vars={
                    "reference_still_mask": EnvContext.reference_still_mask,
                    "root_rot": EnvContext.current.root_rot,
                    "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
                    "anchor_idx": EnvContext.mimic.anchor_idx,
                    "sim_contacts": EnvContext.current.rigid_body_contacts,
                    "historical_actions": EnvContext.historical.actions,
                    "dof_vel": EnvContext.current.dof_vel,
                },
                static_params={
                    "weight": settings.hold_balance_v2,
                    "left_foot_body_ids": v2_foot_ids["left"],
                    "right_foot_body_ids": v2_foot_ids["right"],
                    "ad_min": settings.v2_ad_min,
                    "ad_max": settings.v2_ad_max,
                    "posture_coefficient": settings.v2_posture_coef,
                    "alive_window": settings.v2_alive_window,
                    "dofvel_min": settings.v2_dofvel_min,
                    "dofvel_max": settings.v2_dofvel_max,
                    "zero_during_grace_period": True,
                },
            )
            print(
                f"[hold-fix] HOLD_BALANCE_V2 armed (v2.1 NOISE-CANCELLING): "
                f"reward component 'hold_balance_v2' "
                f"weight={settings.hold_balance_v2} "
                f"POSTURE-RELATIVE (coef {settings.v2_posture_coef}) + "
                f"LIVING-BAND on WINDOWED ACTION-MEAN displacement "
                f"K={settings.v2_alive_window} "
                f"ad=[{settings.v2_ad_min},{settings.v2_ad_max}] "
                f"(exploration noise cancels 1/sqrt(K); zero pay at zero "
                f"motion AND during ring warmup BY CONSTRUCTION; dof_vel "
                f"fallback [{settings.v2_dofvel_min},{settings.v2_dofvel_max}] "
                f"rad/s only if no action stream) "
                f"left_foot_ids={v2_foot_ids['left'].tolist()} "
                f"right_foot_ids={v2_foot_ids['right'].tolist()} "
                f"(TB tags raw_r/scaled_r/hold_balance_v2)"
            )

    if settings.wrist_dir_weight > 0.0:
        from protomotions.envs.component_factories import wrist_dir_rew_factory

        wrist_ids = resolve_wrist_body_ids(body_names, common_naming, device)
        if wrist_ids is None:
            print(
                "[wrist-dir] WARNING: WRIST_DIR_REWARD set but wrist bodies "
                "could not be resolved from the robot config — NOT armed."
            )
            settings.wrist_dir_weight = 0.0
        else:
            reward_components["wrist_dir"] = wrist_dir_rew_factory(
                weight=settings.wrist_dir_weight
            )
            variant = (
                "instantaneous-velocity (sub-flag)" if settings.wrist_dir_instant
                else f"windowed-displacement window={settings.wrist_dir_window_s}s "
                     f"disp_eps={settings.wrist_dir_disp_eps}m"
            )
            print(
                f"[wrist-dir] WRIST_DIR_REWARD armed: reward component "
                f"'wrist_dir' weight={settings.wrist_dir_weight} "
                f"variant={variant} vmax={settings.wrist_dir_vmax} "
                f"wrist_body_ids={wrist_ids.tolist()} "
                f"(TB tags raw_r/scaled_r/wrist_dir, wrist_dir/dir_cos_mean)"
            )

    if settings.root_gain_weight > 0.0:
        from protomotions.envs.component_factories import root_gain_rew_factory

        reward_components["root_gain"] = root_gain_rew_factory(
            weight=settings.root_gain_weight
        )
        print(
            f"[root-gain] ROOT_GAIN_REWARD armed: reward component 'root_gain' "
            f"weight={settings.root_gain_weight} shaping=clamp(proj_gain,0,1) "
            f"window={settings.root_gain_window_s}s "
            f"disp_eps={settings.root_gain_disp_eps}m "
            f"subsample={settings.root_gain_subsample} "
            f"(TB tags raw_r/scaled_r/root_gain, root_gain/gain_mean; targets "
            f"the fwd_gain 0.464 displacement-undershoot axis)"
        )

    if settings.xy_drift_term_m > 0.0:
        from protomotions.envs.context_views import EnvContext
        from protomotions.envs.mdp_component import MdpComponent

        termination_components["xy_drift"] = MdpComponent(
            compute_func=compute_xy_drift_term,
            dynamic_vars={
                "current_anchor_pos": EnvContext.current.anchor_pos,
                "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
                "ref_rigid_body_vel": EnvContext.mimic.ref_state.rigid_body_vel,
                "reference_still_mask": EnvContext.reference_still_mask,
                "anchor_idx": EnvContext.mimic.anchor_idx,
            },
            static_params={
                "drift_threshold": settings.xy_drift_term_m,
                "move_eps": settings.xy_move_eps,
            },
        )
        print(
            f"[xy-term] XY_DRIFT_TERM ACTIVE: termination component 'xy_drift' "
            f"thresh={settings.xy_drift_term_m} m, MOVING-REFERENCE ONLY "
            f"(ref anchor xy-speed > {settings.xy_move_eps} m/s AND not "
            f"reference-still; holds/frozen/pauses exempt via the delta-based "
            f"still mask). Fired-count: TB env/termination/xy_drift_mean "
            f"(x num_envs = count/step; same auto-tag family as fall)."
        )

        if settings.xy_drift_term_penalty > 0.0:
            reward_components["hold_fix_xy_drift_penalty"] = MdpComponent(
                compute_func=compute_xy_drift_penalty,
                dynamic_vars={
                    "current_anchor_pos": EnvContext.current.anchor_pos,
                    "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
                    "ref_rigid_body_vel": EnvContext.mimic.ref_state.rigid_body_vel,
                    "reference_still_mask": EnvContext.reference_still_mask,
                    "anchor_idx": EnvContext.mimic.anchor_idx,
                },
                static_params={
                    "weight": -settings.xy_drift_term_penalty,
                    "drift_threshold": settings.xy_drift_term_m,
                    "move_eps": settings.xy_move_eps,
                    "zero_during_grace_period": True,
                },
            )
            print(
                f"[xy-term] penalty={settings.xy_drift_term_penalty} — "
                f"XY_DRIFT_TERM_PENALTY armed (component [8b]): reward "
                f"component 'hold_fix_xy_drift_penalty' "
                f"weight={-settings.xy_drift_term_penalty} fires on the EXACT "
                f"xy_drift termination condition (same kernel, same params); "
                f"prices the drift fence like a fall so drifting out is never "
                f"the discount exit. Identity law: "
                f"env/termination/xy_drift_mean == "
                f"env/raw_r/hold_fix_xy_drift_penalty_mean "
                f"(TB tags raw_r/scaled_r/hold_fix_xy_drift_penalty)."
            )

    if settings.yaw_drift_term_rad > 0.0:
        from protomotions.envs.context_views import EnvContext
        from protomotions.envs.mdp_component import MdpComponent

        yaw_dyn = {
            "reference_still_mask": EnvContext.reference_still_mask,
            "root_rot": EnvContext.current.root_rot,
            "ref_rigid_body_rot": EnvContext.mimic.ref_state.rigid_body_rot,
            "anchor_idx": EnvContext.mimic.anchor_idx,
        }
        termination_components["yaw_drift"] = MdpComponent(
            compute_func=make_yaw_drift_kernel(as_penalty=False),
            dynamic_vars=dict(yaw_dyn),
            static_params={
                "yaw_threshold": settings.yaw_drift_term_rad,
                "persist_steps": settings.yaw_persist_steps,
            },
        )
        print(
            f"[yaw-term] YAW_DRIFT_TERM ACTIVE: termination component "
            f"'yaw_drift' thresh={settings.yaw_drift_term_rad} rad, "
            f"persist={settings.yaw_persist_steps} steps "
            f"({settings.yaw_persist_steps * 0.02:.2f}s @50Hz — transient "
            f"single-frame spikes exempt), LIVE-REFERENCE ONLY (delta-based "
            f"still mask; frozen clocks/stills never fire; NO yaw-rate gate "
            f"so straight-walk heading drift stays fenced). Fired-count: TB "
            f"env/termination/yaw_drift_mean (same auto-tag family as fall)."
        )

        if settings.yaw_drift_term_penalty > 0.0:
            reward_components["hold_fix_yaw_drift_penalty"] = MdpComponent(
                compute_func=make_yaw_drift_kernel(as_penalty=True),
                dynamic_vars=dict(yaw_dyn),
                static_params={
                    "weight": -settings.yaw_drift_term_penalty,
                    "yaw_threshold": settings.yaw_drift_term_rad,
                    "persist_steps": settings.yaw_persist_steps,
                    "zero_during_grace_period": True,
                },
            )
            print(
                f"[yaw-term] penalty={settings.yaw_drift_term_penalty} — "
                f"YAW_DRIFT_TERM_PENALTY armed (component [8c]): reward "
                f"component 'hold_fix_yaw_drift_penalty' "
                f"weight={-settings.yaw_drift_term_penalty} fires on the "
                f"EXACT yaw_drift termination condition (twin stateful "
                f"kernel, deterministic-identical counters); equal-pricing "
                f"law across all three exits (fall/xy/theta). Identity law: "
                f"env/termination/yaw_drift_mean == "
                f"env/raw_r/hold_fix_yaw_drift_penalty_mean "
                f"(TB tags raw_r/scaled_r/hold_fix_yaw_drift_penalty)."
            )

    if settings.hold_action_grace:
        print(
            "[hold-fix] HOLD_ACTION_GRACE armed: reference-still mask is OR-ed "
            "into perturbation_grace_mask (action_rate untaxed during stills; "
            "in_the_air/max_feet_height stay always-on)"
        )
    if settings.needs_still_mask:
        print(
            f"[hold-fix] stillness detector: root_eps={settings.still_root_eps} m/s "
            f"rot_eps={settings.still_rot_eps} rad/s "
            f"dof_eps={settings.still_dof_eps} rad/s "
            f"window={settings.still_window} steps "
            f"(delta-based; catches frozen reference clocks AND still clips)"
        )
