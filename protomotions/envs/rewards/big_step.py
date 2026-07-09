# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Big-step reward kernels (Track D teacher retrain, OmniH2O-style, dormant).

OmniH2O (He et al., arXiv 2406.08858) reports that continuous regularizers
like feet air time or feet height cause the humanoid to stomp instead of
standing still, and instead uses a "max feet height for each step" reward —
credited once per completed step at its swing apex.  These kernels implement
that per-step credit scheme:

- ``FeetApexHeightReward``: tracks each foot's swing apex (max height above
  ground) between touchdowns and emits ``min(apex, apex_height_cap)`` exactly
  once, on the touchdown transition — never continuously.
- ``StepDisplacementReward``: at each touchdown, rewards the xy distance the
  foot traveled since its previous touchdown, thresholded and capped:
  ``min(max(0, step_length - min_step_length), reward_cap)`` — discourages
  micro/shuffle steps.

Both need per-foot contact-transition tracking across control steps, so unlike
the other reward kernels they are *stateful callables* rather than pure
functions.  State is lazily allocated on first call and re-initialized per-env
whenever ``progress_buf`` does not advance (episode reset).  Contact
transitions use the same simulated foot-contact channels the existing contact
rewards consume (``EnvContext.current.rigid_body_contacts`` subset by
``EnvContext.contact_body_ids``; cf. ``compute_contact_match_rew`` and
``compute_reference_contact_liftoff_penalty`` in ``regularization.py``).

These kernels are dormant capabilities: they are not registered in any recipe
and their factories default to ``weight=0.0``.
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
    """Per-step max-feet-height reward (OmniH2O arXiv 2406.08858 style).

    Tracks each foot's swing apex (max height above ground since liftoff) and
    rewards ``min(apex, apex_height_cap)`` ONCE at touchdown.  No reward is
    emitted while the foot is in the air or in stance — continuous air-time /
    height rewards cause stomping (per the paper).

    Raw units: meters of (capped) apex height per touchdown event, summed over
    feet.  Scale with a positive ``weight`` in the factory metadata.
    """

    __name__ = "feet_apex_height_reward"

    def __init__(self, apex_height_cap: float = 0.15):
        super().__init__()
        if apex_height_cap <= 0.0:
            raise ValueError("apex_height_cap must be positive.")
        self.apex_height_cap = apex_height_cap
        self._swing_apex = None

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

        heights = _foot_heights(rigid_body_pos, contact_body_ids, ground_heights)

        # Update apex for airborne feet (including the liftoff frame).
        in_air = ~contacts
        self._swing_apex = torch.where(
            in_air, torch.maximum(self._swing_apex, heights), self._swing_apex
        )

        # Emit capped apex once, at touchdown; then clear that foot's apex.
        apex_reward = torch.where(
            touchdown,
            self._swing_apex.clamp(max=self.apex_height_cap),
            torch.zeros_like(self._swing_apex),
        )
        self._swing_apex = torch.where(
            touchdown, torch.zeros_like(self._swing_apex), self._swing_apex
        )

        return apex_reward.sum(dim=-1)


class StepDisplacementReward(_FootContactTransitionTracker):
    """Displacement-per-step reward: capped step length credited at touchdown.

    At each foot touchdown, rewards
    ``min(max(0, step_length - min_step_length), reward_cap)`` where
    ``step_length`` is the xy distance from that foot's previous touchdown
    position.  The dead-zone below ``min_step_length`` makes micro/shuffle
    steps worthless; the cap keeps a single lunge from dominating the budget.

    Raw units: meters of (thresholded, capped) step length per touchdown
    event, summed over feet.  Scale with a positive ``weight``.
    """

    __name__ = "step_displacement_reward"

    def __init__(self, min_step_length: float = 0.1, reward_cap: float = 0.5):
        super().__init__()
        if min_step_length < 0.0:
            raise ValueError("min_step_length must be non-negative.")
        if reward_cap <= 0.0:
            raise ValueError("reward_cap must be positive.")
        self.min_step_length = min_step_length
        self.reward_cap = reward_cap
        self._last_touchdown_xy = None

    @_dynamo_disable
    def __call__(
        self,
        sim_contacts: Tensor,
        rigid_body_pos: Tensor,
        contact_body_ids: Tensor,
        progress_buf: Tensor,
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

        # Anchor the next step measurement at this touchdown position.
        self._last_touchdown_xy = torch.where(
            touchdown.unsqueeze(-1), foot_xy, self._last_touchdown_xy
        )

        return step_reward.sum(dim=-1)


__all__ = [
    "FeetApexHeightReward",
    "StepDisplacementReward",
]
