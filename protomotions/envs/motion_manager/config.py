# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for motion manager components.

This module contains all configuration dataclasses for motion manager functionality,
co-located with the motion manager implementations in the same directory.
"""

from typing import Optional, List, Union
from dataclasses import dataclass, field


@dataclass
class MotionManagerConfig:
    """Configuration for motion management."""

    _target_: str = "protomotions.envs.motion_manager.motion_manager.MotionManager"

    init_start_prob: float = field(
        default=0.2,
        metadata={
            "help": "Probability to sample an initial pose instead of random time. Helps prevent local-minima in AMP.",
            "min": 0.0,
            "max": 1.0,
        }
    )

    subset_method: Optional[Union[str, List[int]]] = field(
        default=None,
        metadata={
            "help": "Motion subset for evaluation: 'first', 'last', 'random', or list of motion IDs. None uses all motions.",
            "options": ["first", "last", "random"],
        }
    )

    exclude_motion_ids: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "Motion IDs to exclude from sampling. Useful for removing problematic motions.",
        }
    )

    exclude_motions_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to file with motion IDs to exclude (one per line). Can also be an expert training directory.",
        }
    )

    realign_motion_with_humanoid_on_each_step: bool = field(
        default=False,
        metadata={
            "help": "Realign motion with humanoid each step. Prevents tracking error accumulation for imperfect retargeting.",
        }
    )

    mirror_prob: float = field(
        default=0.0,
        metadata={
            "help": (
                "Online sagittal-mirror augmentation: per-episode Bernoulli "
                "probability of serving the sampled reference motion mirrored "
                "left<->right (components/motion_mirror.py). 0.0 disables "
                "(byte-identical). The PM_MIRROR_PROB env var overrides this at "
                "runtime (resume-safe)."
            ),
            "min": 0.0,
            "max": 1.0,
        },
    )

    reverse_prob: float = field(
        default=0.0,
        metadata={
            "help": (
                "Online time-reversal augmentation: per-episode Bernoulli "
                "probability of serving the sampled reference motion played "
                "backwards (components/motion_reverse.py) -- walking forward "
                "becomes walking backward, pick-up becomes put-down. 0.0 "
                "disables (byte-identical). Independent coin from mirror_prob. "
                "The PM_REVERSE_PROB env var overrides this at runtime "
                "(resume-safe)."
            ),
            "min": 0.0,
            "max": 1.0,
        },
    )


@dataclass
class MimicMotionManagerConfig(MotionManagerConfig):
    """Configuration for mimic motion management."""

    _target_: str = (
        "protomotions.envs.motion_manager.mimic_motion_manager.MimicMotionManager"
    )

    resample_on_reset: bool = field(
        default=True,
        metadata={"help": "Whether to resample motion on environment reset."}
    )

    # Reference freeze/resume augmentation (Track C/D, 2026-07-09; default
    # OFF = stock behavior). Teleop-shaped data aug: per env, the reference
    # clock randomly holds (targets freeze) then resumes, exposing training
    # to the deployment-time "hold last commanded pose" regime that the
    # teacher audit found fully out-of-distribution (hold-stutter finding).
    reference_freeze_prob_per_sec: float = field(
        default=0.0,
        metadata={
            "help": (
                "Per-second probability that an unfrozen env starts a "
                "reference freeze (motion time holds; targets/rewards/expert "
                "obs all see the frozen reference). 0.0 disables."
            ),
            "min": 0.0,
        },
    )
    reference_freeze_duration_range: tuple = field(
        default=(0.5, 2.0),
        metadata={
            "help": "Uniform range (seconds) for each freeze duration."
        },
    )
