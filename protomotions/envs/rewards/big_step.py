# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Big-step reward kernels (Track D teacher retrain, OmniH2O-style, dormant).

REWORKED 2026-07-10 after the OmniH2O code audit: the original kernels
inverted OmniH2O's economics. Their config ships a RAW −2500-weighted apex
SHORTFALL penalty (not curriculum-gated), gates the step-encouraging
feet-air-time term on reference root speed, and ships the continuous
feet-height term at weight 0 while keeping a small in_the_air penalty on the
teacher. Under our previous ``+0.5*min(apex, 0.15)`` form, a 2 cm stomp step
EARNED reward; under the shortfall form it costs heavily. Current forms:

- ``FeetApexHeightReward`` (SHORTFALL PENALTY): tracks each foot's swing apex
  between touchdowns and emits ``sum_feet max(0, apex_target_height - apex)``
  ONCE at touchdown. Weight it NEGATIVELY. Standing still emits nothing (no
  touchdowns), so it is anti-stomp by construction; low shuffle steps cost
  proportionally to their apex shortfall. UNGATED on reference speed: low
  steps are bad whenever they happen.
- ``StepDisplacementReward`` (GATED on reference motion): at touchdown,
  rewards ``min(max(0, step_length - min_step_length), reward_cap)`` — but
  ONLY when the reference root xy speed exceeds ``min_ref_speed`` at that
  moment. No step income on stationary / frozen-lower-body references.
- ``compute_in_the_air_penalty`` (continuous, OmniH2O teacher term): 1.0 per
  step while ALL feet are airborne. Weight it negatively (small).

The stateful kernels need per-foot contact-transition tracking across control
steps; state is lazily allocated and re-initialized per-env whenever
``progress_buf`` does not advance (episode reset). Contact transitions use
the same simulated foot-contact channels the existing contact rewards consume.

These kernels are dormant capabilities: their factories default to
``weight=0.0`` and no stock recipe registers them (the Track D teacher recipe
does).
"""

import torch
from torch import Tensor

try:  # pragma: no cover - depends on torch version
    from torch._dynamo import disable as _dynamo_disable
except Exception:  # pragma: no cover
    def _dynamo_disable(fn):
        return fn


def _foot_heights(
    rigid_body_pos: Tensor,
    contact_body_ids: Tensor,
    ground_heights: Tensor,
) -> Tensor:
    """Foot heights above ground [num_envs, num_feet].

    ``ground_heights`` is the terrain height under the root (as populated on
    ``EnvContext.ground_heights``); adequate for the mostly-flat Track D
    training terrain.
    """
    foot_z = rigid_body_pos[:, contact_body_ids, 2]
    if ground_heights.dim() == 2:
        ground_heights = ground_heights.squeeze(-1)
    return foot_z - ground_heights.unsqueeze(-1)


class _FootContactTransitionTracker:
    """Shared per-foot contact-transition state machine.

    Maintains previous-step contacts and detects touchdown (swing -> stance)
    and liftoff (stance -> swing) transitions.  Handles episode resets by
    watching ``progress_buf``: any env whose progress did not strictly
    increase since the previous call had its episode reset (or is being
    observed for the first time) and gets fresh state with no event emission.
    """

    def __init__(self):
        self._prev_contacts = None
        self._prev_progress = None

    def _update_transitions(self, contacts: Tensor, progress_buf: Tensor):
        """Returns (touchdown_mask, reset_mask), both handling episode resets.

        Args:
            contacts: Current foot contact flags [num_envs, num_feet] (bool).
            progress_buf: Episode progress counter [num_envs].

        Returns:
            touchdown_mask: [num_envs, num_feet] bool — feet that just landed
                (zeroed on reset envs).
            reset_mask: [num_envs] bool — envs whose per-episode state must be
                re-initialized this step.
        """
        num_envs = contacts.shape[0]
        if self._prev_contacts is None or self._prev_contacts.shape != contacts.shape:
            reset_mask = torch.ones(
                num_envs, dtype=torch.bool, device=contacts.device
            )
            self._prev_contacts = contacts.clone()
            self._prev_progress = progress_buf.clone()
            touchdown = torch.zeros_like(contacts)
            return touchdown, reset_mask

        # progress_buf increments every post_physics_step; a value that did not
        # strictly increase means the env was reset since the last call.
        reset_mask = progress_buf <= self._prev_progress

        touchdown = (~self._prev_contacts) & contacts
        touchdown = touchdown & ~reset_mask.unsqueeze(-1)

        self._prev_contacts = contacts.clone()
        self._prev_progress = progress_buf.clone()
        return touchdown, reset_mask


class FeetApexHeightReward(_FootContactTransitionTracker):
    """Per-step swing-apex reward — SHORTFALL penalty OR positive LIFT reward.

    Tracks each foot's swing apex (max height above ground since liftoff) and
    emits ONCE at touchdown, summed over feet. Two ``reward_mode`` forms share
    this identical touchdown gating (standing / stance feet never land, so they
    always emit exactly 0):

    - ``"shortfall"`` (default, back-compat): emits
      ``max(0, apex_target_height - apex)``. Weight NEGATIVELY — a shuffle step
      with a 5 cm apex against the default 0.25 m target costs 0.20 raw; a step
      at/above the target costs nothing. This is OmniH2O's anti-stomp economics
      (arXiv 2406.08858; their yaml ships the raw −2500 apex-shortfall term
      un-gated and the continuous feet-height term at weight 0) but it only ever
      REMOVES a penalty as the foot lifts — it never pays for lifting, so a
      policy already eating the penalty has no positive gradient toward a higher
      step (the ep2502 freeze-attractor; see imprint PR #119).

    - ``"lift"`` (v2, positive): emits ``min(apex, apex_target_height) /
      apex_target_height`` — a normalized reward in ``[0, 1]`` per completed
      swing that PAYS for lifting, grows monotonically with apex, saturates at
      the target, and NEVER penalizes a shortfall (the raw is always ≥ 0).
      Weight POSITIVELY. "Always reward high steps": a taller swing always earns
      strictly more, up to the cap, so the gradient points up toward the target
      instead of merely relaxing a cost. Optionally GATED by a REF-OR-SELF motion
      test (default OFF = ungated): a completed swing pays only when the REFERENCE
      root xy speed exceeds ``min_ref_speed`` OR the ROBOT's OWN (simulated) root
      xy speed exceeds ``min_self_speed``. The ref side blocks marching in place
      against a stationary / frozen-lower-body reference; the self side is a
      fall-RECOVERY escape hatch — if the robot is shoved or falls (wrench DR
      fires on static clips constantly) it MUST be able to take a big recovery
      step and still get paid, gated "relative to itself, not the ref traj". An
      in-place march keeps the root stationary (self-speed ~0) on a static ref
      (ref-speed ~0) so it still pays 0; swaying to unlock is self-defeating
      because root motion on a static ref directly costs the anchor pos/ori
      tracking rewards (imprint PR #119 step-in-place investigation). Each side
      is active only when its threshold > 0 and its velocity tensor is present;
      both 0/absent = ungated. ``"shortfall"`` is never gated.

    Nothing is emitted while the foot is in the air or in stance (continuous
    air-time / height terms cause stomping; OmniH2O ships the continuous
    feet-height term at weight 0).

    Raw units: ``"shortfall"`` = meters of apex shortfall per touchdown,
    summed over feet; ``"lift"`` = normalized apex fraction in ``[0, 1]`` per
    touchdown, summed over feet.
    """

    __name__ = "feet_apex_height_reward"

    def __init__(
        self,
        apex_target_height: float = 0.25,
        reward_mode: str = "shortfall",
        min_ref_speed: float = 0.0,
        min_self_speed: float = 0.0,
        min_swing_sec: float = 0.0,
        control_dt: float = 0.02,
    ):
        super().__init__()
        if apex_target_height <= 0.0:
            raise ValueError("apex_target_height must be positive.")
        if reward_mode not in ("shortfall", "lift"):
            raise ValueError(
                "reward_mode must be 'shortfall' (negative apex-shortfall "
                "penalty) or 'lift' (positive apex reward)."
            )
        if min_ref_speed < 0.0:
            raise ValueError("min_ref_speed must be non-negative.")
        if min_self_speed < 0.0:
            raise ValueError("min_self_speed must be non-negative.")
        if min_swing_sec < 0.0:
            raise ValueError("min_swing_sec must be non-negative.")
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive.")
        self.apex_target_height = apex_target_height
        self.reward_mode = reward_mode
        self.min_ref_speed = min_ref_speed
        self.min_self_speed = min_self_speed
        self.min_swing_sec = min_swing_sec
        self.control_dt = control_dt
        self._swing_apex = None
        self._swing_steps = None

    @_dynamo_disable
    def __call__(
        self,
        sim_contacts: Tensor,
        rigid_body_pos: Tensor,
        ground_heights: Tensor,
        contact_body_ids: Tensor,
        progress_buf: Tensor,
        ref_rigid_body_vel: Tensor = None,
        rigid_body_vel: Tensor = None,
    ) -> Tensor:
        contacts = sim_contacts[:, contact_body_ids].bool()
        touchdown, reset_mask = self._update_transitions(contacts, progress_buf)

        if self._swing_apex is None or self._swing_apex.shape != contacts.shape:
            self._swing_apex = torch.zeros(
                contacts.shape, dtype=torch.float32, device=contacts.device
            )
        self._swing_apex[reset_mask] = 0.0
        if self._swing_steps is None or self._swing_steps.shape != contacts.shape:
            self._swing_steps = torch.zeros(
                contacts.shape, dtype=torch.float32, device=contacts.device
            )
        self._swing_steps[reset_mask] = 0.0

        heights = _foot_heights(rigid_body_pos, contact_body_ids, ground_heights)

        # Update apex + swing-duration counter for airborne feet (including the
        # liftoff frame). The counter is the anti-machine-gun clock: a vibration
        # "swing" of a few control steps never accumulates min_swing_sec of air
        # time, so in "lift" mode it earns nothing (see below).
        in_air = ~contacts
        self._swing_apex = torch.where(
            in_air, torch.maximum(self._swing_apex, heights), self._swing_apex
        )
        self._swing_steps = torch.where(
            in_air, self._swing_steps + 1.0, self._swing_steps
        )

        # Emit once, at touchdown; then clear that apex. "shortfall" pays the
        # (non-negative) apex deficit to be weighted NEGATIVELY; "lift" pays the
        # normalized achieved apex, capped at the target, to be weighted
        # POSITIVELY. Both are 0 off a touchdown (standing/stance never lands).
        if self.reward_mode == "lift":
            emission = torch.where(
                touchdown,
                self._swing_apex.clamp(max=self.apex_target_height)
                / self.apex_target_height,
                torch.zeros_like(self._swing_apex),
            )
            # MIN-SWING-DURATION gate (anti-machine-gun, v5): a completed swing
            # pays ONLY if the foot was airborne >= min_swing_sec. Ultra-high-
            # frequency stepping (the frequency exploit: lift income scales with
            # touchdown COUNT, air-time dodges the slip penalty, high mini-steps
            # dodge the micro-step tax) produces ~0.1s vibration swings that earn
            # ZERO under a 0.25s gate, while any real step (>=0.25s swing) keeps
            # full pay. Disabled at 0.0 (default) = byte-identical back-compat.
            if self.min_swing_sec > 0.0:
                swing_ok = (self._swing_steps * self.control_dt) >= self.min_swing_sec
                emission = emission * swing_ok.float()
        else:  # "shortfall"
            emission = torch.where(
                touchdown,
                (self.apex_target_height - self._swing_apex).clamp(min=0.0),
                torch.zeros_like(self._swing_apex),
            )

        # REF-OR-SELF motion gate — "lift" mode ONLY. Pay the positive lift
        # income only when the REFERENCE root is moving (blocks marching in place
        # against a stationary / frozen-lower-body reference) OR the ROBOT's OWN
        # simulated root is moving (fall-RECOVERY escape hatch — a shoved/fallen
        # robot must still be paid for a big recovery step, gated relative to
        # itself, not the ref traj; imprint PR #119 step-in-place investigation).
        # Each side mirrors StepDisplacementReward's gate idiom (reshape +
        # unsqueeze, flattened or [envs, bodies, 3] both work) and is active ONLY
        # when its own threshold > 0.0 and its tensor is present; both thresholds
        # 0.0 (the default) leave ``gate`` None => emission byte-identical to
        # ungated. "shortfall" stays UNGATED (a low step is bad whenever it
        # happens). An in-place march has self-speed ~0 on a static ref
        # (ref-speed ~0) so it still pays 0; swaying to unlock costs the anchor
        # pos/ori tracking rewards, so it is self-defeating.
        if self.reward_mode == "lift":
            gate = None
            if ref_rigid_body_vel is not None and self.min_ref_speed > 0.0:
                ref_vel = ref_rigid_body_vel.reshape(emission.shape[0], -1, 3)
                ref_speed_xy = ref_vel[:, 0, :2].norm(dim=-1)
                gate = ref_speed_xy > self.min_ref_speed
            if rigid_body_vel is not None and self.min_self_speed > 0.0:
                self_vel = rigid_body_vel.reshape(emission.shape[0], -1, 3)
                self_speed_xy = self_vel[:, 0, :2].norm(dim=-1)
                self_gate = self_speed_xy > self.min_self_speed
                gate = self_gate if gate is None else (gate | self_gate)
            if gate is not None:
                emission = emission * gate.unsqueeze(-1)

        self._swing_apex = torch.where(
            touchdown, torch.zeros_like(self._swing_apex), self._swing_apex
        )
        self._swing_steps = torch.where(
            touchdown, torch.zeros_like(self._swing_steps), self._swing_steps
        )

        return emission.sum(dim=-1)


class StepDisplacementReward(_FootContactTransitionTracker):
    """Displacement-per-step reward, GATED on reference root motion.

    At each foot touchdown, rewards
    ``min(max(0, step_length - min_step_length), reward_cap)`` where
    ``step_length`` is the xy distance from that foot's previous touchdown
    position — but only when the REFERENCE root xy speed exceeds
    ``min_ref_speed`` at the touchdown step (OmniH2O gates their
    step-encouraging feet-air-time term on reference speed the same way).
    Stationary or frozen-lower-body references pay no step income, so this
    term cannot fund stepping-in-place. The dead-zone below
    ``min_step_length`` makes micro/shuffle steps worthless; the cap keeps a
    single lunge from dominating the budget.

    NOTE: touchdown positions keep being anchored even while gated, so a step
    taken across a gate boundary is measured from its true previous touchdown.

    Raw units: meters of (thresholded, capped) step length per touchdown
    event, summed over feet.  Scale with a positive ``weight``.
    """

    __name__ = "step_displacement_reward"

    def __init__(
        self,
        min_step_length: float = 0.1,
        reward_cap: float = 0.5,
        min_ref_speed: float = 0.1,
    ):
        super().__init__()
        if min_step_length < 0.0:
            raise ValueError("min_step_length must be non-negative.")
        if reward_cap <= 0.0:
            raise ValueError("reward_cap must be positive.")
        if min_ref_speed < 0.0:
            raise ValueError("min_ref_speed must be non-negative.")
        self.min_step_length = min_step_length
        self.reward_cap = reward_cap
        self.min_ref_speed = min_ref_speed
        self._last_touchdown_xy = None

    @_dynamo_disable
    def __call__(
        self,
        sim_contacts: Tensor,
        rigid_body_pos: Tensor,
        contact_body_ids: Tensor,
        progress_buf: Tensor,
        ref_rigid_body_vel: Tensor = None,
    ) -> Tensor:
        contacts = sim_contacts[:, contact_body_ids].bool()
        touchdown, reset_mask = self._update_transitions(contacts, progress_buf)

        foot_xy = rigid_body_pos[:, contact_body_ids, :2]

        if (
            self._last_touchdown_xy is None
            or self._last_touchdown_xy.shape != foot_xy.shape
        ):
            self._last_touchdown_xy = foot_xy.clone()
        self._last_touchdown_xy[reset_mask] = foot_xy[reset_mask]

        step_length = torch.norm(foot_xy - self._last_touchdown_xy, dim=-1)
        step_reward = torch.where(
            touchdown,
            (step_length - self.min_step_length).clamp(min=0.0, max=self.reward_cap),
            torch.zeros_like(step_length),
        )

        # Reference-motion gate: pay only while the ref root is moving.
        if ref_rigid_body_vel is not None and self.min_ref_speed > 0.0:
            ref_vel = ref_rigid_body_vel.reshape(step_reward.shape[0], -1, 3)
            ref_speed_xy = ref_vel[:, 0, :2].norm(dim=-1)
            gate = (ref_speed_xy > self.min_ref_speed).unsqueeze(-1)
            step_reward = step_reward * gate

        # Anchor the next step measurement at this touchdown position
        # (ALWAYS, gated or not — see class docstring).
        self._last_touchdown_xy = torch.where(
            touchdown.unsqueeze(-1), foot_xy, self._last_touchdown_xy
        )

        return step_reward.sum(dim=-1)


class MicroStepTax(_FootContactTransitionTracker):
    """Per-touchdown tax on the shuffle signature: steps that are BOTH short AND low.

    At each foot touchdown, emits ``1.0`` (summed over feet) ONLY when the
    completed step was simultaneously

    - SHORT: xy travel since that foot's previous touchdown < ``max_step_length``
      (default 0.10 m), AND
    - LOW: swing apex above ground since liftoff < ``max_apex_height``
      (default 0.06 m).

    This is exactly the shuffle: a tiny, ground-hugging shuffle-step. Weight it
    NEGATIVELY.

    What this kernel deliberately does NOT tax (the two hard constraints):

    - PLANTED / STANDING feet: a stance foot never lifts off and never lands, so
      it produces no touchdown transition -> no tax, ever. Stance = exactly 0.
    - BIG steps: any touchdown whose xy travel >= ``max_step_length`` is NOT
      taxed, regardless of how low it was. A big low-clearance stride (a genuine
      drag) is out of scope here -- that is the foot-slip penalty's job. Only the
      short-AND-low intersection is taxed; either condition alone spares the step.

    Stateful (per-foot swing apex + last-touchdown xy). Per-env state resets
    automatically with the episode via ``progress_buf``, and touchdown events on
    reset steps are suppressed by the shared transition tracker, so the first
    post-reset landing is never taxed off a stale anchor.

    Raw units: count of short-AND-low touchdown events this step, summed over
    feet (0, 1, or 2 for two feet).
    """

    __name__ = "micro_step_tax"

    def __init__(
        self,
        max_step_length: float = 0.10,
        max_apex_height: float = 0.06,
    ):
        super().__init__()
        if max_step_length < 0.0:
            raise ValueError("max_step_length must be non-negative.")
        if max_apex_height < 0.0:
            raise ValueError("max_apex_height must be non-negative.")
        self.max_step_length = max_step_length
        self.max_apex_height = max_apex_height
        self._swing_apex = None
        self._last_touchdown_xy = None

    @_dynamo_disable
    def __call__(
        self,
        sim_contacts: Tensor,
        rigid_body_pos: Tensor,
        ground_heights: Tensor,
        contact_body_ids: Tensor,
        progress_buf: Tensor,
    ) -> Tensor:
        contacts = sim_contacts[:, contact_body_ids].bool()
        touchdown, reset_mask = self._update_transitions(contacts, progress_buf)

        if self._swing_apex is None or self._swing_apex.shape != contacts.shape:
            self._swing_apex = torch.zeros(
                contacts.shape, dtype=torch.float32, device=contacts.device
            )
        self._swing_apex[reset_mask] = 0.0

        foot_xy = rigid_body_pos[:, contact_body_ids, :2]
        if (
            self._last_touchdown_xy is None
            or self._last_touchdown_xy.shape != foot_xy.shape
        ):
            self._last_touchdown_xy = foot_xy.clone()
        self._last_touchdown_xy[reset_mask] = foot_xy[reset_mask]

        heights = _foot_heights(rigid_body_pos, contact_body_ids, ground_heights)

        # Track swing apex for airborne feet (including the liftoff frame).
        in_air = ~contacts
        self._swing_apex = torch.where(
            in_air, torch.maximum(self._swing_apex, heights), self._swing_apex
        )

        step_length = torch.norm(foot_xy - self._last_touchdown_xy, dim=-1)
        is_short = step_length < self.max_step_length
        is_low = self._swing_apex < self.max_apex_height
        tax = (touchdown & is_short & is_low).float()

        # Clear the apex accumulator and re-anchor the step measurement at this
        # touchdown position (only for feet that just landed).
        self._swing_apex = torch.where(
            touchdown, torch.zeros_like(self._swing_apex), self._swing_apex
        )
        self._last_touchdown_xy = torch.where(
            touchdown.unsqueeze(-1), foot_xy, self._last_touchdown_xy
        )

        return tax.sum(dim=-1)


def compute_in_the_air_penalty(
    sim_contacts: Tensor,
    contact_body_ids: Tensor,
) -> Tensor:
    """Continuous both-feet-airborne indicator (OmniH2O ``in_the_air`` term).

    Emits 1.0 for every step in which NO configured contact body touches the
    ground.  Weight it with a small NEGATIVE weight: OmniH2O's teacher ships
    this as a modest continuous penalty alongside the large per-step apex
    shortfall.

    Raw units: 1.0 per fully-airborne step.
    """
    contacts = sim_contacts[:, contact_body_ids].bool()
    return (~contacts).all(dim=-1).float()


def compute_foot_speed_penalty(
    sim_contacts: Tensor,
    rigid_body_vel: Tensor,
    contact_body_ids: Tensor,
    max_foot_speed: float = 2.5,
    ref_rigid_body_vel: Tensor = None,
    ref_speed_scale: float = 0.0,
) -> Tensor:
    """Continuous swing-foot overspeed penalty, REFERENCE-RELATIVE (v5.1).

    For every AIRBORNE foot, penalizes speed above a per-foot, per-step
    allowance ``max(max_foot_speed, ref_speed_scale * ||ref_foot_vel||)``
    (full 3D norms), summed over feet. Weight NEGATIVELY. Stance feet emit 0
    here — contact-phase motion is the foot_slip penalty's job.

    WHY reference-relative (measured on the 120-clip LAFAN eval set, feet =
    lowest-mean-z bodies): median p95 reference foot speed is 2.54 m/s for
    WALK, 5.34 for RUN, 6.19 for SPRINT (max 11.8) — a fixed human-scale
    threshold taxes legitimate locomotion wholesale and rebuilds the freeze
    attractor. The machine-gun exploit instead snaps feet against SLOW or
    static references, where the allowance collapses to the floor: ref foot
    ~0 m/s => allowance = ``max_foot_speed`` (floor, default 2.5 — covers
    recovery steps on static refs); ref sprinting at 8 m/s =>
    allowance = 1.3*8 = 10.4 => real sprinting is free.

    ``ref_speed_scale=0`` or ``ref_rigid_body_vel=None`` disables the
    reference term (pure fixed-floor behavior).

    Raw units: m/s of over-allowance speed, summed over airborne feet/step.
    """
    contacts = sim_contacts[:, contact_body_ids].bool()
    vel = rigid_body_vel.reshape(rigid_body_vel.shape[0], -1, 3)[:, contact_body_ids]
    speed = vel.norm(dim=-1)
    allowance = torch.full_like(speed, max_foot_speed)
    if ref_rigid_body_vel is not None and ref_speed_scale > 0.0:
        ref_vel = ref_rigid_body_vel.reshape(rigid_body_vel.shape[0], -1, 3)[
            :, contact_body_ids
        ]
        ref_speed = ref_vel.norm(dim=-1)
        allowance = torch.maximum(allowance, ref_speed_scale * ref_speed)
    overspeed = (speed - allowance).clamp(min=0.0)
    return (overspeed * (~contacts).float()).sum(dim=-1)


__all__ = [
    "FeetApexHeightReward",
    "StepDisplacementReward",
    "MicroStepTax",
    "compute_in_the_air_penalty",
    "compute_foot_speed_penalty",
]
