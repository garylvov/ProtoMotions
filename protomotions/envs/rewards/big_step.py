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

from protomotions.utils.rotations import calc_heading_quat_inv, quat_rotate

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
        placement_sigma: float = 0.0,
        require_alternation: bool = False,
        recovery_pay_scale: float = 0.5,
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
        if placement_sigma < 0.0:
            raise ValueError("placement_sigma must be non-negative.")
        self.placement_sigma = placement_sigma
        if recovery_pay_scale < 0.0:
            raise ValueError("recovery_pay_scale must be non-negative.")
        # H2 hardening (adversarial review 2026-07-27): recovery-path payments
        # (self-gate-only unlock) are discounted by this factor so a policy
        # cannot fund a full-rate stepping habit through the recovery hatch.
        self.recovery_pay_scale = recovery_pay_scale
        self.require_alternation = bool(require_alternation)
        self._last_paid_foot = None
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
        ref_rigid_body_pos: Tensor = None,
        rigid_body_rot: Tensor = None,
        ref_rigid_body_rot: Tensor = None,
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
        if self.require_alternation:
            if (
                self._last_paid_foot is None
                or self._last_paid_foot.shape[0] != contacts.shape[0]
            ):
                self._last_paid_foot = torch.full(
                    (contacts.shape[0],), -1,
                    dtype=torch.long, device=contacts.device,
                )
            self._last_paid_foot[reset_mask] = -1

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
            ref_gate = None
            self_gate = None
            if ref_rigid_body_vel is not None and self.min_ref_speed > 0.0:
                ref_vel = ref_rigid_body_vel.reshape(emission.shape[0], -1, 3)
                ref_speed_xy = ref_vel[:, 0, :2].norm(dim=-1)
                ref_gate = ref_speed_xy > self.min_ref_speed
            if rigid_body_vel is not None and self.min_self_speed > 0.0:
                self_vel = rigid_body_vel.reshape(emission.shape[0], -1, 3)
                self_speed_xy = self_vel[:, 0, :2].norm(dim=-1)
                self_gate = self_speed_xy > self.min_self_speed

            # PLACEMENT-CONDITIONED lift (v5.2, anti double-step): scale the
            # payout by how well the landing foot PLACES relative to the
            # reference foot, in the ROOT-RELATIVE frame:
            #   factor = exp(-||(foot-root)_xy - (ref_foot-ref_root)_xy||^2 / sigma^2)
            # A stutter/double step lands near the stance foot while the
            # striding reference foot is half a stride away -> large error ->
            # ~0 pay; a back-and-forth step lands even farther -> ~0. A clean
            # step matching the reference stride pays in full. RECOVERY
            # EXEMPTION: when ONLY the self-speed side unlocked the payment
            # (shoved robot, static/slow ref), placement is NOT required -- a
            # recovery foot must land where balance demands, not where the
            # reference stands. placement_sigma=0 disables (back-compat).
            placement = None
            if (
                self.placement_sigma > 0.0
                and ref_rigid_body_pos is not None
            ):
                pos = rigid_body_pos.reshape(emission.shape[0], -1, 3)
                ref_pos = ref_rigid_body_pos.reshape(emission.shape[0], -1, 3)
                foot_rel = pos[:, contact_body_ids, :2] - pos[:, 0:1, :2]
                ref_foot_rel = (
                    ref_pos[:, contact_body_ids, :2] - ref_pos[:, 0:1, :2]
                )
                # M1 yaw alignment (adversarial review 2026-07-27): foot_rel /
                # ref_foot_rel above are root-relative XY offsets expressed in
                # WORLD axes, so any heading error between policy and reference
                # roots contaminated the placement comparison (a perfect local
                # stride under a 90-deg heading error scored ~0). Rotate each
                # offset into its OWN root's heading frame (policy by inverse
                # policy root yaw, ref by inverse ref root yaw) before
                # comparing; quats here are xyzw (w_last=True, matching every
                # other rewards-module call site). Falls back to the world-
                # frame comparison when rotations are unavailable (e.g. frozen
                # pre-M1 configs whose dynamic_vars lack the rot tensors).
                if rigid_body_rot is not None and ref_rigid_body_rot is not None:
                    rot = rigid_body_rot.reshape(emission.shape[0], -1, 4)
                    ref_rot = ref_rigid_body_rot.reshape(
                        emission.shape[0], -1, 4
                    )
                    heading_inv = calc_heading_quat_inv(rot[:, 0], w_last=True)
                    ref_heading_inv = calc_heading_quat_inv(
                        ref_rot[:, 0], w_last=True
                    )
                    n_feet = foot_rel.shape[1]
                    zpad = torch.zeros(
                        foot_rel.shape[0], n_feet, 1,
                        dtype=foot_rel.dtype, device=foot_rel.device,
                    )
                    foot_rel = quat_rotate(
                        heading_inv.unsqueeze(1).expand(-1, n_feet, -1),
                        torch.cat([foot_rel, zpad], dim=-1),
                        w_last=True,
                    )[..., :2]
                    ref_foot_rel = quat_rotate(
                        ref_heading_inv.unsqueeze(1).expand(-1, n_feet, -1),
                        torch.cat([ref_foot_rel, zpad], dim=-1),
                        w_last=True,
                    )[..., :2]
                perr2 = (foot_rel - ref_foot_rel).square().sum(dim=-1)
                placement = torch.exp(-perr2 / (self.placement_sigma**2))

            if ref_gate is not None or self_gate is not None:
                paid = torch.zeros_like(emission)
                if ref_gate is not None:
                    # Locomotion path: ref is moving -> placement REQUIRED.
                    ref_pay = emission * ref_gate.unsqueeze(-1)
                    if placement is not None:
                        ref_pay = ref_pay * placement
                    # ALTERNATION gate (v5.2 anti same-foot double-tap): a
                    # foot's touchdown pays only if the OTHER foot has landed
                    # since this foot's last PAID touchdown. Natural gait
                    # always alternates (walk through sprint); a same-foot
                    # tap-tap inside one reference stride is double-dipping.
                    # Simultaneous two-foot landings (jump) both pay and clear
                    # the state. Recovery path (below) is exempt.
                    if self.require_alternation and self._last_paid_foot is not None:
                        n_feet = ref_pay.shape[-1]
                        foot_idx = torch.arange(
                            n_feet, device=ref_pay.device
                        ).unsqueeze(0)
                        repeat = foot_idx == self._last_paid_foot.unsqueeze(-1)
                        both_land = touchdown.sum(dim=-1) >= 2
                        repeat = repeat & ~both_land.unsqueeze(-1)
                        ref_pay = ref_pay * (~repeat).float()
                        pays_now = (ref_pay > 0.0) & touchdown
                        single = pays_now.sum(dim=-1) == 1
                        new_last = torch.where(
                            both_land,
                            torch.full_like(self._last_paid_foot, -1),
                            self._last_paid_foot,
                        )
                        single_idx = pays_now.float().argmax(dim=-1)
                        new_last = torch.where(single, single_idx, new_last)
                        self._last_paid_foot = new_last
                    paid = torch.maximum(paid, ref_pay)
                if self_gate is not None:
                    # Recovery path: pay WITHOUT placement ONLY when the ref
                    # side did NOT unlock (self moving, ref static/slow =
                    # genuine perturbation recovery). During normal walking
                    # the robot's own root also moves, so a bare self-gate
                    # would bypass placement on every stride -- hence & ~ref.
                    # H2 hardening (adversarial review 2026-07-27): the
                    # recovery hatch used to pay FULL rate with NO alternation
                    # gate, making same-foot double-taps against static refs a
                    # funded habit. Now: (a) pay is discounted by
                    # recovery_pay_scale (default 0.5); (b) require_alternation
                    # applies here too, on the SAME _last_paid_foot machinery,
                    # and recovery payments UPDATE _last_paid_foot (they
                    # previously did not, so a recovery step never counted as
                    # "this foot was paid last"). Placement stays bypassed on
                    # recovery -- the ref is static there, so ref foot
                    # placement is meaningless; balance dictates the landing.
                    recovery = self_gate
                    if ref_gate is not None:
                        recovery = self_gate & ~ref_gate
                    rec_pay = emission * recovery.unsqueeze(-1) * float(
                        getattr(self, "recovery_pay_scale", 0.5)
                    )
                    if self.require_alternation and self._last_paid_foot is not None:
                        n_feet = rec_pay.shape[-1]
                        foot_idx = torch.arange(
                            n_feet, device=rec_pay.device
                        ).unsqueeze(0)
                        repeat = foot_idx == self._last_paid_foot.unsqueeze(-1)
                        both_land = touchdown.sum(dim=-1) >= 2
                        repeat = repeat & ~both_land.unsqueeze(-1)
                        rec_pay = rec_pay * (~repeat).float()
                        pays_now = (rec_pay > 0.0) & touchdown
                        single = pays_now.sum(dim=-1) == 1
                        # Update ONLY recovery envs; locomotion envs were
                        # already updated by the ref-path block above.
                        new_last = torch.where(
                            both_land & recovery,
                            torch.full_like(self._last_paid_foot, -1),
                            self._last_paid_foot,
                        )
                        single_idx = pays_now.float().argmax(dim=-1)
                        new_last = torch.where(
                            single & recovery, single_idx, new_last
                        )
                        self._last_paid_foot = new_last
                    paid = torch.maximum(paid, rec_pay)
                emission = paid
            elif placement is not None:
                emission = emission * placement

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
        min_swing_steps: int = 2,
    ):
        super().__init__()
        if max_step_length < 0.0:
            raise ValueError("max_step_length must be non-negative.")
        if max_apex_height < 0.0:
            raise ValueError("max_apex_height must be non-negative.")
        if min_swing_steps < 0:
            raise ValueError("min_swing_steps must be non-negative.")
        self.max_step_length = max_step_length
        self.max_apex_height = max_apex_height
        # M3 re-arm (2026-07-27): TB showed the M2 guard (min_swing_steps=3,
        # shared with StepBudgetPenalty) being exploited as a free-tap lane --
        # the policy inserts ultra-short-swing micro-taps that clear a
        # 3-control-step swing floor (0.06s @ 20ms dt) neither spending budget
        # NOR incurring tax. The counter increments once per airborne frame
        # INCLUDING the liftoff frame itself, so a true 1-frame solver-chatter
        # flicker (lift off this frame, land next frame) reads _swing_steps
        # == 1 at touchdown; anything that actually clears the ground for 2+
        # frames is a real (if tiny) step, not chatter. Lowering the tax-only
        # guard to 2 exempts ONLY that single-frame chatter case and taxes
        # every 2+-frame micro-tap again, while the credit-bank guard on
        # StepBudgetPenalty stays at 3 (unchanged -- it protects the bank,
        # not the tax).
        self.min_swing_steps = min_swing_steps
        self._swing_apex = None
        self._swing_steps = None
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
        # M2 chatter guard state (getattr: unpickled pre-M2 instances lack it).
        _swing_steps = getattr(self, "_swing_steps", None)
        if _swing_steps is None or _swing_steps.shape != contacts.shape:
            _swing_steps = torch.zeros(
                contacts.shape, dtype=torch.float32, device=contacts.device
            )
        _swing_steps[reset_mask] = 0.0

        foot_xy = rigid_body_pos[:, contact_body_ids, :2]
        if (
            self._last_touchdown_xy is None
            or self._last_touchdown_xy.shape != foot_xy.shape
        ):
            self._last_touchdown_xy = foot_xy.clone()
        self._last_touchdown_xy[reset_mask] = foot_xy[reset_mask]

        heights = _foot_heights(rigid_body_pos, contact_body_ids, ground_heights)

        # Track swing apex + duration for airborne feet (incl. liftoff frame).
        in_air = ~contacts
        self._swing_apex = torch.where(
            in_air, torch.maximum(self._swing_apex, heights), self._swing_apex
        )
        _swing_steps = torch.where(in_air, _swing_steps + 1.0, _swing_steps)

        step_length = torch.norm(foot_xy - self._last_touchdown_xy, dim=-1)
        is_short = step_length < self.max_step_length
        is_low = self._swing_apex < self.max_apex_height
        # M2 chatter guard: a touchdown whose preceding swing lasted fewer
        # than min_swing_steps control steps is contact chatter, not a step.
        swing_ok = _swing_steps >= float(getattr(self, "min_swing_steps", 2))
        tax = (touchdown & is_short & is_low & swing_ok).float()

        # Clear the apex/duration accumulators and re-anchor the step
        # measurement at this touchdown position (only feet that just landed).
        self._swing_apex = torch.where(
            touchdown, torch.zeros_like(self._swing_apex), self._swing_apex
        )
        self._swing_steps = torch.where(
            touchdown, torch.zeros_like(_swing_steps), _swing_steps
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


class StepBudgetPenalty(_FootContactTransitionTracker):
    """Excess-cadence penalty: policy touchdowns beyond the reference's (v5.3).

    Every REFERENCE touchdown grants that foot ONE policy-touchdown credit
    (bank capped at ``max_credits``). A policy touchdown consumes a credit;
    landing with an EMPTY bank emits 1.0 (weight NEGATIVELY). "4 fake steps
    per 1 reference step" = 3 uncredited landings = 3 penalties per cycle —
    this prices the extra steps themselves, which the v5.2 zero-pay gates do
    not (unpaid stutter-stepping survives as a free balance habit).

    Applies ONLY while the reference root is locomoting
    (ref_speed > min_ref_speed): a static reference grants no credits ever,
    so recovery stepping against static refs must be exempt, and is.
    Separate tracker instance => its own contact-transition state.

    Raw units: 1.0 per uncredited policy touchdown, summed over feet.
    """

    __name__ = "step_budget_penalty"

    def __init__(
        self,
        min_ref_speed: float = 0.05,
        max_credits: float = 2.0,
        ref_contact_threshold: float = 0.5,
        min_swing_steps: int = 3,
    ):
        super().__init__()
        if min_ref_speed < 0.0:
            raise ValueError("min_ref_speed must be non-negative.")
        if max_credits < 1.0:
            raise ValueError("max_credits must be >= 1.")
        if not (0.0 <= ref_contact_threshold < 1.0):
            raise ValueError("ref_contact_threshold must be in [0, 1).")
        if min_swing_steps < 0:
            raise ValueError("min_swing_steps must be non-negative.")
        # H1 (adversarial review 2026-07-27): reference contacts are SMOOTHED
        # floats in [0, 1] (moving-average window 7 + frame blending), so the
        # old ``.bool()`` (i.e. > 0) dilated ref stance by +/- ~3 frames on
        # each side -- short reference swings never showed a liftoff ->
        # touchdown edge, no credits were granted, and every policy step
        # overdrafted. Threshold at 0.5 recovers the true contact schedule
        # (mirrors compute_reference_contact_liftoff_penalty).
        self.ref_contact_threshold = ref_contact_threshold
        # M2 (adversarial review 2026-07-27): a policy "touchdown" whose
        # preceding swing lasted < min_swing_steps control steps is contact
        # chatter, not a step -- it must not spend a budget credit.
        self.min_swing_steps = min_swing_steps
        self.min_ref_speed = min_ref_speed
        self.max_credits = max_credits
        self._credits = None
        self._prev_ref_contacts = None
        self._swing_steps = None

    @_dynamo_disable
    def __call__(
        self,
        sim_contacts: Tensor,
        contact_body_ids: Tensor,
        progress_buf: Tensor,
        ref_rigid_body_contacts: Tensor = None,
        ref_rigid_body_vel: Tensor = None,
    ) -> Tensor:
        contacts = sim_contacts[:, contact_body_ids].bool()
        touchdown, reset_mask = self._update_transitions(contacts, progress_buf)

        if self._credits is None or self._credits.shape != contacts.shape:
            self._credits = torch.full_like(
                contacts, 1.0, dtype=torch.float32
            )
        self._credits[reset_mask] = 1.0

        # M2 chatter-guard swing counter (getattr: unpickled pre-M2 instances
        # lack the attribute).
        _swing_steps = getattr(self, "_swing_steps", None)
        if _swing_steps is None or _swing_steps.shape != contacts.shape:
            _swing_steps = torch.zeros(
                contacts.shape, dtype=torch.float32, device=contacts.device
            )
        _swing_steps[reset_mask] = 0.0
        in_air = ~contacts
        _swing_steps = torch.where(in_air, _swing_steps + 1.0, _swing_steps)
        swing_ok = _swing_steps >= float(getattr(self, "min_swing_steps", 3))
        self._swing_steps = torch.where(
            touchdown, torch.zeros_like(_swing_steps), _swing_steps
        )

        if ref_rigid_body_contacts is None or ref_rigid_body_vel is None:
            return torch.zeros(
                contacts.shape[0], device=contacts.device, dtype=torch.float32
            )

        # H1: ref contacts are smoothed floats in [0, 1]; threshold at 0.5
        # instead of .bool() (> 0), which dilated stance by the smoothing
        # window and starved credit grants on short reference swings.
        ref_contacts = (
            ref_rigid_body_contacts[:, contact_body_ids]
            > float(getattr(self, "ref_contact_threshold", 0.5))
        )
        if (
            self._prev_ref_contacts is None
            or self._prev_ref_contacts.shape != ref_contacts.shape
        ):
            self._prev_ref_contacts = ref_contacts.clone()
        ref_touchdown = (~self._prev_ref_contacts) & ref_contacts
        ref_touchdown = ref_touchdown & ~reset_mask.unsqueeze(-1)
        self._prev_ref_contacts = ref_contacts.clone()

        # Grant credits on reference touchdowns (capped bank).
        self._credits = torch.where(
            ref_touchdown,
            (self._credits + 1.0).clamp(max=self.max_credits),
            self._credits,
        )

        # Locomotion gate: budget applies only while the ref root moves.
        ref_vel = ref_rigid_body_vel.reshape(contacts.shape[0], -1, 3)
        loco = (ref_vel[:, 0, :2].norm(dim=-1) > self.min_ref_speed).unsqueeze(-1)

        # M2 chatter guard: a touchdown after a < min_swing_steps "swing" is
        # solver contact flicker, not a real step -- no spend, no overdraft.
        spend = touchdown & loco & swing_ok
        has_credit = self._credits >= 1.0
        overdraft = (spend & ~has_credit).float()
        self._credits = torch.where(
            spend & has_credit, self._credits - 1.0, self._credits
        )
        return overdraft.sum(dim=-1)


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
    "StepBudgetPenalty",
    "StepDisplacementReward",
    "MicroStepTax",
    "compute_in_the_air_penalty",
    "compute_foot_speed_penalty",
]
