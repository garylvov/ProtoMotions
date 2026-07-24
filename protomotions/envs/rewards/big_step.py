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
      instead of merely relaxing a cost. Optionally GATED on reference root xy
      speed via ``min_ref_speed`` (default 0.0 = ungated): when set, a completed
      swing pays only while the reference root is moving, so marching in place
      against a stationary reference earns no lift (imprint PR #119 step-in-place
      investigation). ``"shortfall"`` is never gated.

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
        self.apex_target_height = apex_target_height
        self.reward_mode = reward_mode
        self.min_ref_speed = min_ref_speed
        self._swing_apex = None

    @_dynamo_disable
    def __call__(
        self,
        sim_contacts: Tensor,
        rigid_body_pos: Tensor,
        ground_heights: Tensor,
        contact_body_ids: Tensor,
        progress_buf: Tensor,
        ref_rigid_body_vel: Tensor = None,
    ) -> Tensor:
        contacts = sim_contacts[:, contact_body_ids].bool()
        touchdown, reset_mask = self._update_transitions(contacts, progress_buf)

        if self._swing_apex is None or self._swing_apex.shape != contacts.shape:
            self._swing_apex = torch.zeros(
                contacts.shape, dtype=torch.float32, device=contacts.device
            )
        self._swing_apex[reset_mask] = 0.0

        heights = _foot_heights(rigid_body_pos, contact_body_ids, ground_heights)

        # Update apex for airborne feet (including the liftoff frame).
        in_air = ~contacts
        self._swing_apex = torch.where(
            in_air, torch.maximum(self._swing_apex, heights), self._swing_apex
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
        else:  # "shortfall"
            emission = torch.where(
                touchdown,
                (self.apex_target_height - self._swing_apex).clamp(min=0.0),
                torch.zeros_like(self._swing_apex),
            )

        # Reference-motion gate — "lift" mode ONLY: pay the positive lift income
        # only while the REFERENCE root is moving in xy above ``min_ref_speed``,
        # so a policy marching in place against a stationary / frozen-lower-body
        # reference earns no lift (imprint PR #119 step-in-place investigation).
        # Mirrors StepDisplacementReward's gate above (reshape + unsqueeze), so a
        # flattened or [envs, bodies, 3] ref-vel tensor both work. "shortfall"
        # stays UNGATED (a low step is bad whenever it happens). ``min_ref_speed
        # == 0.0`` (the default) leaves emission byte-identical to ungated.
        if (
            self.reward_mode == "lift"
            and ref_rigid_body_vel is not None
            and self.min_ref_speed > 0.0
        ):
            ref_vel = ref_rigid_body_vel.reshape(emission.shape[0], -1, 3)
            ref_speed_xy = ref_vel[:, 0, :2].norm(dim=-1)
            gate = (ref_speed_xy > self.min_ref_speed).unsqueeze(-1)
            emission = emission * gate

        self._swing_apex = torch.where(
            touchdown, torch.zeros_like(self._swing_apex), self._swing_apex
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


__all__ = [
    "FeetApexHeightReward",
    "StepDisplacementReward",
    "MicroStepTax",
    "compute_in_the_air_penalty",
]
