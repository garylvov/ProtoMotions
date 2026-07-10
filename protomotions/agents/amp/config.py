# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for AMP (Adversarial Motion Priors) agent.

This module defines configurations for the AMP algorithm which uses a discriminator
to learn motion priors from reference motions.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from protomotions.envs.mdp_component import MdpComponent

from protomotions.agents.common.config import ModuleContainerConfig
from protomotions.agents.ppo.config import (
    PPOModelConfig,
    PPOAgentConfig,
    OptimizerConfig,
)
from dataclasses import dataclass, field


@dataclass
class AMPParametersConfig:
    """Configuration for AMP-specific hyperparameters."""

    conditional_discriminator: bool = field(
        default=False,
        metadata={"help": "Whether to use conditional discriminator based on motion state."}
    )

    discriminator_reward_w: float = field(
        default=1.0,
        metadata={"help": "Weight for discriminator reward in total reward.", "min": 0.0}
    )

    discriminator_weight_decay: float = field(
        default=0.0001,
        metadata={"help": "L2 weight decay for discriminator parameters.", "min": 0.0}
    )
    discriminator_logit_weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay specifically for discriminator logit layer.", "min": 0.0}
    )
    discriminator_batch_size: int = field(
        default=4096,
        metadata={"help": "Batch size for discriminator training.", "min": 1}
    )
    discriminator_grad_penalty: float = field(
        default=5.0,
        metadata={"help": "Gradient penalty coefficient for discriminator stability.", "min": 0.0}
    )
    discriminator_optimization_ratio: int = field(
        default=1,
        metadata={"help": "Ratio of discriminator updates to policy updates.", "min": 1}
    )

    discriminator_replay_keep_prob: float = field(
        default=0.01,
        metadata={"help": "Probability to keep samples in replay buffer.", "min": 0.0, "max": 1.0}
    )
    discriminator_replay_size: int = field(
        default=200000,
        metadata={"help": "Maximum size of discriminator replay buffer.", "min": 1}
    )

    discriminator_reward_threshold: float = field(
        default=0.05,
        metadata={"help": "Threshold for discriminator reward termination.", "min": 0.0, "max": 1.0}
    )
    discriminator_max_cumulative_bad_transitions: int = field(
        default=10,
        metadata={"help": "Max bad transitions before termination.", "min": 1}
    )

    use_disc_critic: bool = field(
        default=True,
        metadata={"help": "Use a value baseline for disc advantages. False = raw discounted disc rewards."}
    )


@dataclass
class DiscriminatorConfig(ModuleContainerConfig):
    """Configuration for AMP Discriminator network."""

    _target_: str = "protomotions.agents.amp.model.Discriminator"
    out_keys: List[str] = field(
        default_factory=lambda: ["disc_logits"],
        metadata={"help": "Output key for discriminator logits."}
    )


@dataclass
class AMPModelConfig(PPOModelConfig):
    """Configuration for AMP Model (Actor-Critic with Discriminator)."""

    _target_: str = "protomotions.agents.amp.model.AMPModel"
    discriminator: DiscriminatorConfig = field(
        default_factory=DiscriminatorConfig,
        metadata={"help": "Discriminator network for motion prior learning."}
    )
    discriminator_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=1e-4),
        metadata={"help": "Optimizer settings for discriminator."}
    )
    disc_critic: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Critic network for discriminator reward."}
    )
    disc_critic_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=1e-4),
        metadata={"help": "Optimizer settings for discriminator critic."}
    )


@dataclass
class AMPAgentConfig(PPOAgentConfig):
    """Main configuration class for AMP Agent."""

    _target_: str = "protomotions.agents.amp.agent.AMP"

    model: AMPModelConfig = field(
        default_factory=AMPModelConfig,
        metadata={"help": "AMP model configuration including discriminator."}
    )

    amp_parameters: AMPParametersConfig = field(
        default_factory=AMPParametersConfig,
        metadata={"help": "AMP-specific training parameters."}
    )

    reference_obs_components: Dict[str, "MdpComponent"] = field(
        default_factory=dict,
        metadata={"help": "MdpComponent instances for computing reference motion features. Agent injects motion_lib params at runtime."}
    )

    add_root_displacement_features: bool = field(
        default=False,
        metadata={
            "help": (
                "MimicADD only (Track D, dormant by default; A/B option — the "
                "2026-07-10 final design carries xy+heading pressure in the "
                "task-reward channel instead): append the wrapped root heading "
                "error (1D) to the mimic_target_poses_diff discriminator input "
                "(plus heading-frame xy 2D when add_root_xy_features is set). "
                "Expert positive samples get matching zero channels."
            )
        },
    )
    add_root_xy_features: bool = field(
        default=False,
        metadata={
            "help": (
                "MimicADD only (A/B sub-option of add_root_displacement_"
                "features): additionally append the heading-frame root xy "
                "error channels (legacy 3-channel form)."
            )
        },
    )
    global_diff_bodies: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "MimicADD only (dormant; A/B option, rejected as Track D "
                "default 2026-07-10 — couples hand precision to deploy-time "
                "odom quality): body-name regexes whose position diff keeps "
                "the FULL world-frame error (root displacement included); "
                "all other bodies get root-stripped (articulation-relative) "
                "position diffs. None = stock diff computation."
            )
        },
    )
    add_diff_feature_scales: Optional[Dict[str, float]] = field(
        default=None,
        metadata={
            "help": (
                "MimicADD only (Track D per-body saliency bias, dormant by "
                "default): map of body-name regex -> scale applied to that "
                "body's channels in mimic_target_poses_diff (special key "
                "'root_displacement' scales the appended root channels). "
                "Unmatched bodies scale 1.0. A saliency bias on ADD's "
                "auto-balancing, not a hard weight; with normalize_obs=True "
                "on the discriminator the bias is strongest early in "
                "training (running normalizer absorbs static scales "
                "asymptotically)."
            )
        },
    )
