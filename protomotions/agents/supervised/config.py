# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for generic supervised rollout training."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from protomotions.agents.base_agent.config import BaseAgentConfig, BaseModelConfig
from protomotions.agents.common.supervision import SupervisionLossConfig


class RolloutActor(Enum):
    """Policy source used to step the environment during supervised rollout collection."""

    STUDENT = "student"
    EXPERT = "expert"

    @classmethod
    def from_str(cls, value: str) -> "RolloutActor":
        try:
            return next(
                member for member in cls if member.value.lower() == value.lower()
            )
        except StopIteration:
            valid = [member.value for member in cls]
            raise ValueError(
                f"'{value}' is not a valid {cls.__name__}. Valid values are: {valid}"
            )


@dataclass
class SupervisedAgentConfig(BaseAgentConfig):
    """Generic supervised imitation agent configuration.

    Experiment files choose the rollout actor, optional external expert
    checkpoint, and supervised loss keys. The agent loop stays independent of
    the specific student model.
    """

    _target_: str = "protomotions.agents.supervised.agent.SupervisedAgent"

    model: BaseModelConfig = field(
        default_factory=BaseModelConfig,
        metadata={"help": "Model configuration."},
    )
    expert_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional checkpoint for an external expert policy."},
    )
    rollout_actor: RolloutActor = field(
        default=RolloutActor.STUDENT,
        metadata={
            "help": "Policy used for collecting rollout actions."
        },
    )
    loss: SupervisionLossConfig = field(
        default_factory=SupervisionLossConfig,
        metadata={"help": "Supervised loss over model outputs and labels."},
    )
    # Supervised port of PPO L2C2Config from protomotions/agents/ppo/config.py.
    # The tracker recipe examples/experiments/mimic/mlp_bm_l2c2.py enables the
    # term with lambda_l2c2=1.0 and explicit noisy->clean observation pairs.
    l2c2_weight: float = field(
        default=0.0,
        metadata={"help": "L2C2 loss coefficient for supervised distillation."},
    )
    l2c2_obs_pairs: Dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Map from noisy supervised obs key to clean counterpart key."},
    )
    # Track C (2026-07-09) additions. Both default OFF so existing recipes and
    # already-pickled resolved_configs keep stock behavior (agent.py reads them
    # via getattr for old pickles that predate these fields).
    l2c2_mse_form: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, the L2C2 term is plain MSE(pred_noisy, pred_clean) "
                "(no input-distance ratio, no clamp). Only used when "
                "l2c2_weight > 0."
            )
        },
    )
    action_rate_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Temporal smoothness penalty on the supervised prediction: "
                "weight * mean((prediction - previous_actions)^2). Requires a "
                "'previous_actions' key in the training batch (standard env "
                "obs component, history_steps=1). NOTE: intentionally NOT "
                "affected by action_dim_weights — smoothness stays uniform "
                "across joints."
            )
        },
    )
    expert_action_delta_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Action-delta matching loss: weight * mean(dimw * "
                "((pred_t - prev_action_t) - (expert_t - expert_prev_t))^2). "
                "Supervises velocity gain/direction in action space (wrist "
                "under-response fix). Uses action_dim_weights (mean-normalized) "
                "when set. Requires an external expert; when > 0 the agent "
                "stores expert_prev_actions in the rollout buffer. Samples "
                "with an all-zero previous_actions (fresh episode) are masked "
                "out. Complements (does not replace) action_rate_weight."
            )
        },
    )
    # --- FK Cartesian wrist loss (default OFF; see
    # SupervisedAgent._calculate_fk_wrist_loss for the full contract). ---
    fk_wrist_pos_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight of the FK Cartesian wrist POSITION loss. FK is applied "
                "to the predicted action (a PD joint-position target), giving "
                "the COMMANDED wrist position in the anchor-relative, "
                "heading-local frame -- the same frame the teacher's "
                "relative_body_pos_rew_factory reward uses. 0.0 (default) = "
                "term absent, zero FK cost paid."
            )
        },
    )
    fk_wrist_ori_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight of the FK Cartesian wrist ORIENTATION loss (MSE over "
                "the 6D tan-norm encoding of the heading-local wrist rotation, "
                "the same encoding the reference observation carries). Student "
                "twin of relative_body_ori_rew_factory. 0.0 (default) = off."
            )
        },
    )
    fk_wrist_body_names: Optional[list] = field(
        default=None,
        metadata={
            "help": (
                "Bodies scored by the FK wrist loss. None (default) resolves "
                "to the robot's hand bodies "
                "(common_naming_to_robot_body_names all_left/right_hand_bodies, "
                "i.e. left/right_wrist_yaw_link on H1_2). Every name must also "
                "be in the masked-mimic conditionable set, since the reference "
                "target comes from that observation."
            )
        },
    )
    fk_wrist_ref_key: str = field(
        default="masked_mimic_target_poses",
        metadata={
            "help": (
                "Batch key carrying the masked-mimic sparse target poses "
                "(build_sparse_target_poses with include_root_relative=True). "
                "Its root-relative block supplies the reference wrist pose."
            )
        },
    )
    fk_wrist_ref_mask_key: str = field(
        default="masked_mimic_target_masks",
        metadata={
            "help": (
                "Batch key carrying the per-(step, body, {pos,rot}) visibility "
                "masks aligned with fk_wrist_ref_key. Samples whose wrist "
                "target is masked out contribute nothing to the loss."
            )
        },
    )
    fk_wrist_root_rot_obs_key: Optional[str] = field(
        default="max_coords_obs",
        metadata={
            "help": (
                "Batch key of the max-coords humanoid observation, used to read "
                "the root's heading-local rotation (its roll/pitch) so the FK "
                "output can be rotated from the pelvis body frame into the "
                "heading-local frame the reference lives in. Set to None to "
                "skip that rotation (assumes an upright pelvis -- a real "
                "approximation, only for robots/tasks with no root tilt)."
            )
        },
    )
    fk_wrist_future_step: int = field(
        default=0,
        metadata={
            "help": (
                "Which masked-mimic future frame to score against (0 = the "
                "nearest sampled future frame)."
            )
        },
    )
    action_dim_weights: Optional[list] = field(
        default=None,
        metadata={
            "help": (
                "Optional per-action-dim weights (length = number_of_actions, "
                "robot dof order) for the supervised imitation MSE. "
                "Normalized by their mean inside the loss so the total loss "
                "scale stays comparable to the unweighted MSE. None (default) "
                "= stock uniform MSE. Only valid with loss_type=mse."
            )
        },
    )
