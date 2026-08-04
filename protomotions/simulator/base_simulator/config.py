# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for base simulator and domain randomization."""

from typing import Literal, Tuple, List, Dict, Optional, Any, Union
import torch
import re
from dataclasses import dataclass, field


def get_matching_indices(
    names: List[str],
    names_to_match: Optional[List[str]] = None,
    indices_to_match: Optional[List[int]] = None,
) -> List[int]:
    """
    Get the indices of the names that match the given names or indices.

    Args:
        names: List of all available names
        names_to_match: List of regex patterns to match against names
        indices_to_match: List of indices to return directly

    Returns:
        List of indices where names match the regex patterns
    """
    assert (
        names_to_match is not None or indices_to_match is not None
    ), "Either names_to_match or indices_to_match must be provided"
    assert (
        names_to_match is None or indices_to_match is None
    ), "Only one of names_to_match or indices_to_match must be provided"

    if names_to_match is not None:
        # Set to store unique matching names (avoid duplicates from multiple regex)
        matching_names = set()

        # Go over all regex patterns
        for regex_pattern in names_to_match:
            # Find all names that match the current regex
            for i, name in enumerate(names):
                if re.fullmatch(regex_pattern, name):
                    assert (
                        i not in matching_names
                    ), f"Multiple regex patterns match the same name {name}"
                    matching_names.add(i)

        # Get indices for all unique matching names
        return list(matching_names)

    return indices_to_match


@dataclass
class MarkerConfig:
    """Configuration for a single marker instance."""

    size: Literal["tiny", "small", "regular"] = field(
        default="regular", metadata={"help": "Marker size for visualization."}
    )


@dataclass
class VisualizationMarkerConfig:
    """Configuration for a group of visualization markers."""

    type: Literal["sphere", "arrow"] = field(
        default="sphere", metadata={"help": "Marker geometry type."}
    )
    color: Tuple[float, float, float] = field(
        default=(1.0, 0.0, 0.0), metadata={"help": "RGB color values (0-1)."}
    )
    markers: List[MarkerConfig] = field(
        default_factory=list, metadata={"help": "List of marker configurations."}
    )


@dataclass
class MarkerState:
    """Represents the state of a marker in 3D space."""

    translation: torch.Tensor = field(
        default=None, metadata={"help": "Translation vector (position)."}
    )
    orientation: torch.Tensor = field(
        default=None, metadata={"help": "Orientation quaternion."}
    )
    color: Optional[Tuple[float, float, float]] = field(
        default=None, metadata={"help": "Optional RGB color override."}
    )
    scale: Optional[torch.Tensor] = field(
        default=None,
        metadata={
            "help": "Optional per-marker scale, broadcastable to the marker "
            "group's (num_envs * num_markers, 3). When None the static scale "
            "built from MarkerConfig.size at instantiation is used."
        },
    )


@dataclass
class ActionNoiseDomainRandomizationConfig:
    """Configuration for action noise domain randomization."""

    action_noise_range: Tuple[float, float] = field(
        default=None, metadata={"help": "Range (min, max) for action noise."}
    )
    dof_names: Optional[List[str]] = field(
        default=None, metadata={"help": "DOF names to apply noise to (regex patterns)."}
    )
    dof_indices: Optional[List[int]] = field(
        default=None, metadata={"help": "DOF indices to apply noise to."}
    )

    def __post_init__(self):
        """Validate that dof_names and dof_indices are not both provided."""
        if self.dof_names is not None and self.dof_indices is not None:
            raise ValueError("Only one of dof_names or dof_indices must be provided.")
        if self.dof_names is None and self.dof_indices is None:
            raise ValueError("Either dof_names or dof_indices must be provided.")
        if self.action_noise_range is None:
            raise ValueError("action_noise_range must be provided.")
        if self.action_noise_range[0] >= self.action_noise_range[1]:
            raise ValueError(
                "action_noise_range must be a tuple of two values where the first value is less than the second value."
            )


@dataclass
class FrictionDomainRandomizationConfig:
    """Configuration for friction domain randomization."""

    num_buckets: int = field(
        default=10,
        metadata={"help": "Number of friction buckets for environments.", "min": 1},
    )
    static_friction_range: Tuple[float, float] = field(
        default=(0.5, 1.5), metadata={"help": "Range (min, max) for static friction."}
    )
    dynamic_friction_range: Tuple[float, float] = field(
        default=(0.5, 1.5), metadata={"help": "Range (min, max) for dynamic friction."}
    )
    restitution_range: Tuple[float, float] = field(
        default=(0.0, 0.1), metadata={"help": "Range (min, max) for restitution."}
    )
    body_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Body names to apply randomization to (regex patterns)."},
    )
    body_indices: Optional[List[int]] = field(
        default=None, metadata={"help": "Body indices to apply randomization to."}
    )

    def __post_init__(self):
        """Validate that body_names and body_indices are not both provided."""
        if self.body_names is not None and self.body_indices is not None:
            raise ValueError("Only one of body_names or body_indices must be provided.")
        if self.body_names is None and self.body_indices is None:
            raise ValueError("Either body_names or body_indices must be provided.")


@dataclass
class ObjectAssetDomainRandomizationConfig:
    """Configuration for scene object asset domain randomization.

    Ranges are absolute sampled values. When a range is set, it overrides the
    matching base value from each scene object's ObjectOptions.
    """

    num_buckets: int = field(
        default=10,
        metadata={"help": "Number of object asset property buckets.", "min": 1},
    )
    static_friction_range: Optional[Tuple[float, float]] = field(
        default=None,
        metadata={"help": "Absolute range (min, max) for object static friction."},
    )
    dynamic_friction_range: Optional[Tuple[float, float]] = field(
        default=None,
        metadata={"help": "Absolute range (min, max) for object dynamic friction."},
    )
    restitution_range: Optional[Tuple[float, float]] = field(
        default=None,
        metadata={"help": "Absolute range (min, max) for object restitution."},
    )
    mass_range: Optional[Tuple[float, float]] = field(
        default=None, metadata={"help": "Absolute range (min, max) for object mass."}
    )
    density_range: Optional[Tuple[float, float]] = field(
        default=None,
        metadata={"help": "Absolute range (min, max) for object density."},
    )
    center_of_mass_range: Optional[Dict[str, Tuple[float, float]]] = field(
        default=None,
        metadata={
            "help": (
                "Absolute local center-of-mass range per axis, e.g. "
                "{'x': (-0.05, 0.05), 'y': (0.0, 0.0), 'z': (0.0, 0.1)}."
            )
        },
    )

    def __post_init__(self):
        if self.num_buckets < 1:
            raise ValueError("num_buckets must be at least 1.")
        if self.mass_range is not None and self.density_range is not None:
            raise ValueError("Only one of mass_range or density_range may be set.")

        range_fields = (
            "static_friction_range",
            "dynamic_friction_range",
            "restitution_range",
            "mass_range",
            "density_range",
            "center_of_mass_range",
        )
        if all(getattr(self, field_name) is None for field_name in range_fields):
            raise ValueError(
                "At least one object asset randomization range is required."
            )
        for field_name in range_fields:
            value_range = getattr(self, field_name)
            if value_range is None:
                continue
            if field_name == "center_of_mass_range":
                self._validate_center_of_mass_range(value_range)
                continue
            if len(value_range) != 2 or value_range[0] >= value_range[1]:
                raise ValueError(
                    f"{field_name} must be a tuple of two values where min < max."
                )

    @staticmethod
    def _validate_center_of_mass_range(
        center_of_mass_range: Dict[str, Tuple[float, float]],
    ) -> None:
        if not center_of_mass_range:
            raise ValueError("center_of_mass_range must define at least one axis.")
        invalid_axes = set(center_of_mass_range) - {"x", "y", "z"}
        if invalid_axes:
            raise ValueError(
                f"center_of_mass_range contains invalid axes: {invalid_axes}"
            )
        for axis, value_range in center_of_mass_range.items():
            # Equal bounds let configs pin an axis while randomizing others.
            if len(value_range) != 2 or value_range[0] > value_range[1]:
                raise ValueError(
                    f"center_of_mass_range['{axis}'] must be a tuple of two values where min <= max."
                )

    def sample(self, num_samples: int, num_assets: int, device=None) -> Dict[str, Any]:
        """Sample absolute object asset properties for each bucket and asset."""

        def sample_range(value_range):
            if value_range is None:
                return None
            return (
                torch.rand(num_samples, num_assets, device=device)
                * (value_range[1] - value_range[0])
                + value_range[0]
            )

        center_of_mass = None
        if self.center_of_mass_range is not None:
            center_of_mass = torch.zeros(
                num_samples, num_assets, 3, device=device, dtype=torch.float
            )
            for axis_idx, axis in enumerate(("x", "y", "z")):
                value_range = self.center_of_mass_range.get(axis)
                if value_range is None:
                    continue
                center_of_mass[..., axis_idx] = (
                    torch.rand(num_samples, num_assets, device=device)
                    * (value_range[1] - value_range[0])
                    + value_range[0]
                )

        return {
            "static_friction": sample_range(self.static_friction_range),
            "dynamic_friction": sample_range(self.dynamic_friction_range),
            "restitution": sample_range(self.restitution_range),
            "mass": sample_range(self.mass_range),
            "density": sample_range(self.density_range),
            "center_of_mass": center_of_mass,
        }


@dataclass
class CenterOfMassDomainRandomizationConfig:
    """Configuration for center of mass domain randomization."""

    com_range: Dict[str, Tuple[float, float]] = field(
        default_factory=dict,
        metadata={
            "help": "Range per axis: {'x': (-0.1, 0.1), 'y': (-0.1, 0.1), 'z': (-0.1, 0.1)}"
        },
    )
    body_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Body names to apply randomization to (regex patterns)."},
    )
    body_indices: Optional[List[int]] = field(
        default=None, metadata={"help": "Body indices to apply randomization to."}
    )

    def __post_init__(self):
        """Validate that com_range is a dictionary with valid keys."""
        if self.com_range is None:
            raise ValueError("com_range must be a dictionary with valid keys.")
        if not all(key in ["x", "y", "z"] for key in self.com_range.keys()):
            raise ValueError("com_range must be a dictionary with valid keys.")
        if self.body_names is None and self.body_indices is None:
            raise ValueError("Either body_names or body_indices must be provided.")
        if self.body_names is not None and self.body_indices is not None:
            raise ValueError("Only one of body_names or body_indices must be provided.")


@dataclass
class MassScaleDomainRandomizationConfig:
    """Configuration for robot body-mass scale domain randomization (MASS-DR).

    Gary directive 2026-07-10: a computer + gear will be mounted on the robot,
    so the MAIN BODY mass runs up to ~1.3x spec. Samples a per-env
    MULTIPLICATIVE mass scale for the configured bodies (typically the torso
    link) from ``mass_scale_range`` and, optionally, a small independent
    multiplier for ALL links from ``all_links_scale_range`` (manufacturing /
    model-error spread). Multipliers COMPOSE on the configured main bodies.

    Applied once after robot creation (per-env static assignment — a mounted
    computer is a permanent condition, not a per-episode event; across
    thousands of envs the population covers the range densely). Uses the
    PhysX articulation view mass API (``get_masses``/``set_masses``) that was
    unavailable when the heavydr/superdr recipes were written (their
    docstrings list per-link mass as an unavailable axis).
    """

    mass_scale_range: Tuple[float, float] = field(
        default=(1.0, 1.3),
        metadata={
            "help": "Multiplicative mass-scale range (min, max) for the main bodies."
        },
    )
    body_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Main-body names to scale (e.g. ['torso_link'])."},
    )
    body_indices: Optional[List[int]] = field(
        default=None, metadata={"help": "Main-body indices to scale."}
    )
    all_links_scale_range: Optional[Tuple[float, float]] = field(
        default=None,
        metadata={
            "help": (
                "Optional independent multiplicative range applied to EVERY "
                "link (e.g. (0.95, 1.05)); None = main bodies only."
            )
        },
    )

    def __post_init__(self):
        if self.body_names is None and self.body_indices is None:
            raise ValueError("Either body_names or body_indices must be provided.")
        if self.body_names is not None and self.body_indices is not None:
            raise ValueError("Only one of body_names or body_indices must be provided.")
        for name, rng in (
            ("mass_scale_range", self.mass_scale_range),
            ("all_links_scale_range", self.all_links_scale_range),
        ):
            if rng is None:
                continue
            lo, hi = rng
            if not (0.0 < lo <= hi):
                raise ValueError(f"{name} must satisfy 0 < min <= max, got {rng}.")


# PER-GROUP GAIN-DR (2026-08-04, marionette upper body): default joint-group
# regex patterns for the Unitree H1-2 27-DOF actuated set. Used when per-group
# stiffness/damping ranges are configured and ``group_dof_patterns`` is left
# unset. Every DR'd DOF must fall in EXACTLY ONE group (validated loudly at
# sample time) -- the union below is exhaustive for H1-2's DR'd dof list:
#   12 leg joints + 1 waist joint + 14 arm joints = 27.
H1_2_GAIN_DR_GROUP_PATTERNS: Dict[str, List[str]] = {
    "legs": [
        ".*_hip_(yaw|pitch|roll)_joint",
        ".*_knee_joint",
        ".*_ankle_(pitch|roll)_joint",
    ],
    "waist": ["torso_joint"],
    "arms": [
        ".*_shoulder_(pitch|roll|yaw)_joint",
        ".*_elbow_joint",
        ".*_wrist_(roll|pitch|yaw)_joint",
    ],
}


def resolve_gain_dr_groups(
    dof_names: List[str], group_patterns: Dict[str, List[str]]
) -> Dict[str, List[int]]:
    """Partition ``dof_names`` into joint groups by regex, COLUMN-indexed.

    Returns ``{group_name: [column, ...]}`` where ``column`` indexes into the
    given ``dof_names`` list (i.e. into the GAIN-DR sample matrices' second
    axis), not into the robot's full DOF ordering.

    Fails LOUD (``ValueError``) if any DOF matches zero groups or more than
    one group -- a silent mis-partition would quietly train the wrong limb at
    the wrong stiffness, which is exactly the failure this feature exists to
    avoid.
    """
    columns: Dict[str, List[int]] = {group: [] for group in group_patterns}
    hits: Dict[str, List[str]] = {}
    for column, name in enumerate(dof_names):
        matched = [
            group
            for group, patterns in group_patterns.items()
            if any(re.fullmatch(pattern, name) for pattern in patterns)
        ]
        hits[name] = matched
        if len(matched) == 1:
            columns[matched[0]].append(column)
    unmatched = [name for name, m in hits.items() if not m]
    doubled = {name: m for name, m in hits.items() if len(m) > 1}
    if unmatched or doubled:
        raise ValueError(
            "GAIN-DR per-group partition is not exhaustive/disjoint over the "
            f"randomized DOFs. unmatched={unmatched} double_matched={doubled} "
            f"groups={ {g: p for g, p in group_patterns.items()} }"
        )
    return columns


@dataclass
class ActuatorGainDomainRandomizationConfig:
    """Configuration for per-DOF actuator PD-gain domain randomization (GAIN-DR).

    T7 (2026-07-13): motors vary unit-to-unit and drift with wear/temperature,
    so trained policies should be robust to PD gains that differ from the
    nominal spec. Samples per-env, per-DOF MULTIPLICATIVE scales for
    stiffness (``stiffness_scale_range``) and damping
    (``damping_scale_range``) applied to every actuated DOF. Optionally also
    scales the per-DOF effort limit (``effort_limit_scale_range``) — the
    IsaacLab articulation view exposes ``set_dof_max_forces`` via
    ``write_joint_effort_limit_to_sim`` with the exact same per-env/per-DOF
    cost profile as the gain setters, so it is included as a third,
    independently-optional axis rather than skipped.

    Applied once after robot creation (per-env static assignment, mirroring
    MASS-DR — motor-to-motor variation is a fixed property of the unit, not a
    per-episode event). Uses the IsaacLab Articulation gain API
    (``write_joint_stiffness_to_sim`` / ``write_joint_damping_to_sim`` /
    ``write_joint_effort_limit_to_sim``), which for BUILT_IN_PD
    (ImplicitActuatorCfg) robots pushes directly into the PhysX implicit PD
    solver via ``root_physx_view.set_dof_stiffnesses`` / ``set_dof_dampings``
    / ``set_dof_max_forces`` — same idiom as the MASS-DR mass API.
    """

    stiffness_scale_range: Tuple[float, float] = field(
        default=(0.7, 1.3),
        metadata={
            "help": "Multiplicative stiffness (P-gain) scale range (min, max), per DOF per env."
        },
    )
    damping_scale_range: Tuple[float, float] = field(
        default=(0.7, 1.3),
        metadata={
            "help": "Multiplicative damping (D-gain) scale range (min, max), per DOF per env."
        },
    )
    effort_limit_scale_range: Optional[Tuple[float, float]] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative effort-limit scale range (min, max), "
                "per DOF per env; None = effort limits untouched."
            )
        },
    )
    dof_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "DOF names to randomize (regex patterns), e.g. ['.*'] for all DOFs."},
    )
    dof_indices: Optional[List[int]] = field(
        default=None, metadata={"help": "DOF indices to randomize."}
    )
    group_dof_patterns: Optional[Dict[str, List[str]]] = field(
        default=None,
        metadata={
            "help": (
                "PER-GROUP GAIN-DR (2026-08-04): {group_name: [regex, ...]} "
                "partition of the randomized DOFs into joint groups (e.g. "
                "legs / waist / arms). None + any group range set = use "
                "H1_2_GAIN_DR_GROUP_PATTERNS. Every randomized DOF must match "
                "exactly one group (unmatched or double-matched = loud "
                "ValueError at sample time)."
            )
        },
    )
    group_stiffness_scale_ranges: Optional[Dict[str, Tuple[float, float]]] = field(
        default=None,
        metadata={
            "help": (
                "Per-group stiffness scale ranges {group_name: (min, max)}; "
                "groups omitted here fall back to stiffness_scale_range. "
                "None = OFF = uniform stiffness_scale_range everywhere "
                "(byte-identical to the pre-2026-08-04 behavior)."
            )
        },
    )
    group_damping_scale_ranges: Optional[Dict[str, Tuple[float, float]]] = field(
        default=None,
        metadata={
            "help": (
                "Per-group damping scale ranges {group_name: (min, max)}; "
                "groups omitted fall back to damping_scale_range. Ignored "
                "when constant_damping_ratio=True."
            )
        },
    )
    group_effort_limit_scale_ranges: Optional[Dict[str, Tuple[float, float]]] = field(
        default=None,
        metadata={
            "help": (
                "Per-group EFFORT-LIMIT scale ranges {group_name: (min, max)}; "
                "groups omitted fall back to effort_limit_scale_range, or to "
                "the no-op (1.0, 1.0) when that is None. Setting this turns "
                "the effort axis ON (KINESTHETIC TEACHING 2026-08-04: capping "
                "arm torque bounds how hard the robot can fight a human hand "
                "regardless of kp). None = OFF."
            )
        },
    )
    constant_damping_ratio: bool = field(
        default=False,
        metadata={
            "help": (
                "When True, DERIVE each DOF's damping scale as "
                "sqrt(stiffness_scale) instead of sampling it independently, "
                "which holds the damping ratio zeta = d / (2*sqrt(k*m)) "
                "constant under the stiffness scaling (k -> s*k, "
                "d -> sqrt(s)*d). Default False = today's behavior (damping "
                "sampled independently from its own range, so a uniform "
                "scale s also drops zeta by sqrt(s) -- the confound the "
                "weak-gain eval sweep hit)."
            )
        },
    )
    env_gain_scale_group: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Which group's geometric-mean stiffness scale becomes the "
                "per-env aggregate 'env_gain_scale' consumed by the "
                "MARIONETTE perturbation coupling. None = AUTO: the 'legs' "
                "group when per-group ranges are active (balance authority "
                "lives in the legs), else all randomized DOFs (today's "
                "behavior). 'all' forces the all-DOF geometric mean; any "
                "other value must name a configured group."
            )
        },
    )

    def __post_init__(self):
        if self.dof_names is None and self.dof_indices is None:
            raise ValueError("Either dof_names or dof_indices must be provided.")
        if self.dof_names is not None and self.dof_indices is not None:
            raise ValueError("Only one of dof_names or dof_indices must be provided.")
        for name, rng in (
            ("stiffness_scale_range", self.stiffness_scale_range),
            ("damping_scale_range", self.damping_scale_range),
            ("effort_limit_scale_range", self.effort_limit_scale_range),
        ):
            if rng is None:
                continue
            lo, hi = rng
            if not (0.0 < lo <= hi):
                raise ValueError(f"{name} must satisfy 0 < min <= max, got {rng}.")
        for fld in (
            "group_stiffness_scale_ranges",
            "group_damping_scale_ranges",
            "group_effort_limit_scale_ranges",
        ):
            groups = getattr(self, fld)
            if not groups:
                continue
            for group, rng in groups.items():
                lo, hi = rng
                if not (0.0 < lo <= hi):
                    raise ValueError(
                        f"{fld}['{group}'] must satisfy 0 < min <= max, got {rng}."
                    )
        known = set((self.group_dof_patterns or H1_2_GAIN_DR_GROUP_PATTERNS).keys())
        for fld in (
            "group_stiffness_scale_ranges",
            "group_damping_scale_ranges",
            "group_effort_limit_scale_ranges",
        ):
            for group in (getattr(self, fld) or {}):
                if group not in known:
                    raise ValueError(
                        f"{fld} names unknown group '{group}'; configured "
                        f"groups are {sorted(known)}."
                    )
        if self.env_gain_scale_group not in (None, "all") and (
            self.env_gain_scale_group not in known
        ):
            raise ValueError(
                f"env_gain_scale_group='{self.env_gain_scale_group}' is neither "
                f"'all' nor one of the configured groups {sorted(known)}."
            )


@dataclass
class RobotNoiseConfig:
    """Configuration for robot state noise.

    Used for both observation noise (domain randomization during training) and
    reset noise (Reference State Initialization / RSI).

    For observation noise: adds noise to state observations for sim-to-real transfer.
    When enabled, regular state variables have noise applied while privileged_*
    versions remain clean for asymmetric actor-critic training.

    For reset noise (RSI): adds noise to the robot's physics state during
    environment resets, helping the policy learn to recover from imperfect
    initial conditions.

    Noise values are scales for additive uniform noise in [-scale, +scale].
    Each field accepts a scalar (uniform across all axes) or a list for
    per-axis control (e.g., [x, y, z]).

    Observation noise is applied hierarchically:
    - DOF noise: applied to joint positions and velocities
    - Root noise: applied to root body orientation and angular velocity
    - Anchor noise: applied to anchor body orientation and angular velocity
    - Whole-body noise: applied to all rigid body positions, rotations, velocities

    Root and anchor noise are applied on top of clean (privileged) data,
    not on already-noisy whole-body data.
    """

    # DOF-level noise
    dof_pos_noise: Union[float, List[float]] = field(
        default=0.0,
        metadata={
            "help": "Noise scale for DOF positions. Scalar or per-DOF list (radians)."
        },
    )
    dof_vel_noise: Union[float, List[float]] = field(
        default=0.0,
        metadata={
            "help": "Noise scale for DOF velocities. Scalar or per-DOF list (rad/s)."
        },
    )

    # Root body noise
    root_pos_noise: Union[float, List[float]] = field(
        default=0.0,
        metadata={
            "help": "Noise scale for root position. Scalar or per-axis [x,y,z] (meters)."
        },
    )
    root_rot_noise: Union[float, List[float]] = field(
        default=0.0,
        metadata={
            "help": "Noise scale for root orientation. Scalar or per-axis [roll,pitch,yaw] (radians)."
        },
    )
    root_vel_noise: Union[float, List[float]] = field(
        default=0.0,
        metadata={
            "help": "Noise scale for root linear velocity. Scalar or per-axis [x,y,z] (m/s)."
        },
    )
    root_ang_vel_noise: Union[float, List[float]] = field(
        default=0.0,
        metadata={
            "help": "Noise scale for root angular velocity. Scalar or per-axis (rad/s)."
        },
    )

    # Anchor body noise (observation noise only)
    anchor_rot_noise: Union[float, List[float]] = field(
        default=0.0,
        metadata={"help": "Noise scale for anchor body orientation quaternion."},
    )
    anchor_ang_vel_noise: Union[float, List[float]] = field(
        default=0.0,
        metadata={"help": "Noise scale for anchor body angular velocity (rad/s)."},
    )

    # Whole-body noise (all rigid bodies, observation noise only)
    body_pos_noise: Union[float, List[float]] = field(
        default=0.0,
        metadata={"help": "Noise scale for all rigid body positions (meters)."},
    )
    body_rot_noise: Union[float, List[float]] = field(
        default=0.0,
        metadata={"help": "Noise scale for all rigid body orientations (quaternion)."},
    )
    body_vel_noise: Union[float, List[float]] = field(
        default=0.0,
        metadata={"help": "Noise scale for all rigid body linear velocities (m/s)."},
    )
    body_ang_vel_noise: Union[float, List[float]] = field(
        default=0.0,
        metadata={"help": "Noise scale for all rigid body angular velocities (rad/s)."},
    )

    # Environment observation noise
    ground_height_noise: Union[float, List[float]] = field(
        default=0.0,
        metadata={"help": "Noise scale for ground height observations (meters)."},
    )

    def _is_nonzero(self, value: Union[float, List[float]]) -> bool:
        if isinstance(value, list):
            return any(v != 0.0 for v in value)
        return value != 0.0

    def has_noise(self) -> bool:
        """Check if any noise is configured."""
        return (
            self._is_nonzero(self.dof_pos_noise)
            or self._is_nonzero(self.dof_vel_noise)
            or self._is_nonzero(self.root_pos_noise)
            or self._is_nonzero(self.root_rot_noise)
            or self._is_nonzero(self.root_vel_noise)
            or self._is_nonzero(self.root_ang_vel_noise)
            or self._is_nonzero(self.anchor_rot_noise)
            or self._is_nonzero(self.anchor_ang_vel_noise)
            or self._is_nonzero(self.body_pos_noise)
            or self._is_nonzero(self.body_rot_noise)
            or self._is_nonzero(self.body_vel_noise)
            or self._is_nonzero(self.body_ang_vel_noise)
            or self._is_nonzero(self.ground_height_noise)
        )

@dataclass
class PushDomainRandomizationConfig:
    """Configuration for push/perturbation domain randomization.

    Applies random velocity impulses to the robot at random intervals to
    simulate external disturbances (bumps, pushes) for sim-to-real transfer.

    Push velocities are sampled uniformly from [-max, +max] for each component.
    Push is enabled when any velocity component is non-zero.
    """

    push_interval_range: Tuple[float, float] = field(
        default=(1.0, 3.0),
        metadata={"help": "Range (min, max) in seconds between pushes."},
    )
    max_linear_velocity: Tuple[float, float, float] = field(
        default=(0.0, 0.0, 0.0),
        metadata={"help": "Max linear velocity impulse (x, y, z) in m/s."},
    )
    max_angular_velocity: Tuple[float, float, float] = field(
        default=(0.0, 0.0, 0.0),
        metadata={"help": "Max angular velocity impulse (roll, pitch, yaw) in rad/s."},
    )
    action_rate_grace_sec: float = field(
        default=0.0,
        metadata={
            "help": (
                "Track D post-push grace window: seconds after each impulse "
                "push during which the env-side action-rate penalty is "
                "suspended for the pushed envs (decisive recovery swings go "
                "untaxed; see Simulator.get_action_rate_grace_mask). "
                "0.0 (default) = no grace."
            )
        },
    )
    # ---- Epoch-keyed magnitude ramp (night13 2026-07-14 DR curriculum). Runtime
    # multiplier on the sampled push linear+angular impulse (applied live in
    # _apply_push_if_due). Defaults are exact no-ops. env.on_epoch_end ramps
    # magnitude_scale = start + (1-start)*min(1,(epoch-start_epoch)/ramp_epochs). ----
    magnitude_scale: float = field(
        default=1.0,
        metadata={"help": "Runtime multiplier on push impulse magnitude (curriculum ramp knob)."},
    )
    magnitude_start_scale: float = field(
        default=1.0,
        metadata={"help": "push magnitude_scale at the ramp start epoch."},
    )
    magnitude_ramp_epochs: Optional[int] = field(
        default=None,
        metadata={"help": "If set, ramp push magnitude_scale start->1.0 over N epochs. None=off."},
    )
    magnitude_ramp_start_epoch: int = field(
        default=0,
        metadata={"help": "Absolute epoch at which the push magnitude ramp begins."},
    )

    def __post_init__(self):
        if self.push_interval_range[0] <= 0 or self.push_interval_range[1] <= 0:
            raise ValueError("push_interval_range values must be positive.")
        if self.push_interval_range[0] > self.push_interval_range[1]:
            raise ValueError(
                "push_interval_range[0] must be <= push_interval_range[1]."
            )

    def has_push(self) -> bool:
        """Check if any push velocity is configured (non-zero)."""
        return any(v != 0.0 for v in self.max_linear_velocity) or any(
            v != 0.0 for v in self.max_angular_velocity
        )


@dataclass
class WrenchDomainRandomizationConfig:
    """Configuration for random external wrench (force + torque) domain randomization.

    Unlike push DR (which SETS root velocity instantaneously), this applies a
    true external force/torque burst to a randomly chosen body from
    ``body_names`` for a sampled duration, at randomized intervals, per env.
    Force/torque direction is uniform on the sphere (world frame); magnitude
    is sampled uniformly from the configured range. Requires backend support
    (currently implemented for the IsaacLab backend via
    ``Articulation.set_external_force_and_torque``).
    """

    force_magnitude_range: Tuple[float, float] = field(
        default=(0.0, 0.0),
        metadata={"help": "Range (min, max) of force magnitude in N."},
    )
    torque_magnitude_range: Tuple[float, float] = field(
        default=(0.0, 0.0),
        metadata={"help": "Range (min, max) of torque magnitude in N*m."},
    )
    duration_range: Tuple[float, float] = field(
        default=(0.1, 0.3),
        metadata={"help": "Range (min, max) of wrench burst duration in seconds."},
    )
    interval_range: Tuple[float, float] = field(
        default=(1.0, 3.0),
        metadata={
            "help": "Range (min, max) in seconds between end of one burst and start of the next."
        },
    )
    # ---- Persistent "hanging weight" controls (night13 2026-07-14 wrist-bag) ----
    clean_remainder: bool = field(
        default=False,
        metadata={
            "help": (
                "When True, the (1 - persistent_fraction) envs NOT selected for the "
                "persistent load receive NO wrench at all (never cycle) — clean. "
                "Use with persistent_fraction<1 to apply the load to only a fraction "
                "of envs (e.g. 0.30) and leave the rest clean. Default False = "
                "remainder cycles normally."
            )
        },
    )
    persistent_all_bodies: bool = field(
        default=False,
        metadata={
            "help": (
                "When True, a persistent-cohort env loads ALL candidate bodies "
                "(e.g. BOTH wrists) with the full sampled magnitude each — a bag on "
                "each — instead of one randomly-chosen body. Sampled once per body "
                "per episode and held. Default False = one body per env."
            )
        },
    )
    persistent_ramp_in_sec: float = field(
        default=0.0,
        metadata={
            "help": (
                "Smooth onset for the persistent load: ease the held force in from 0 "
                "to its sampled value over this many seconds at episode start (cosine), "
                "then hold constant. 0.0 (default) = instant application."
            )
        },
    )
    body_names: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Candidate body names (sim naming); ONE is chosen per burst per env."
        },
    )
    persistent_fraction: float = field(
        default=0.0,
        metadata={
            "help": (
                "Fraction of envs (per-env Bernoulli draw at every reset, so "
                "cohort membership churns) that receive a PERSISTENT wrench: "
                "force/torque sampled once at reset and held constant for the "
                "whole episode (no interval/duration cycling). Remaining envs "
                "cycle normally. 0.0 (default) = exactly the previous behavior."
            )
        },
    )
    # ---- Ramped persistent forces (Track D 2026-07-10). Defaults preserve
    # the previous step-profile behavior exactly. ----
    ramp_in_range: Tuple[float, float] = field(
        default=(0.0, 0.0),
        metadata={
            "help": (
                "Range (min, max) seconds of cosine ease-IN before the hold "
                "phase. (0, 0) = instant onset (previous behavior). Sampled "
                "per event per env; total event = ramp_in + hold(duration_"
                "range) + ramp_out."
            )
        },
    )
    ramp_out_range: Tuple[float, float] = field(
        default=(0.0, 0.0),
        metadata={
            "help": (
                "Range (min, max) seconds of cosine ease-OUT after the hold "
                "phase. (0, 0) = instant release (previous behavior)."
            )
        },
    )
    direction_mode: str = field(
        default="uniform",
        metadata={
            "help": (
                "Force direction sampling: 'uniform' = uniform on the sphere "
                "(previous behavior); 'horizontal' = mostly horizontal (z "
                "shrunk 4x before normalization — chest persistent forces); "
                "'downward' = PAYLOAD: full sampled magnitude locked to "
                "constant -z (payload weight) plus a small horizontal "
                "component (10-18% of magnitude, constant per event = "
                "temporally-correlated bag swing/inertia)."
            )
        },
    )
    independent_bodies: bool = field(
        default=False,
        metadata={
            "help": (
                "When True, EVERY body in body_names gets its own independent "
                "per-env scheduler (events overlap: sometimes one body loaded, "
                "sometimes several — asymmetric and full carries). False "
                "(default) = one body chosen per event (previous behavior)."
            )
        },
    )
    all_bodies_prob: float = field(
        default=0.0,
        metadata={
            "help": (
                "Per-event probability that ALL candidate bodies are loaded "
                "TOGETHER, with the sampled magnitude split equally across "
                "them (two-hand drag/pull posture); otherwise one body is "
                "chosen as usual. Ignored with independent_bodies=True. "
                "0.0 (default) = previous behavior."
            )
        },
    )
    action_rate_grace: bool = field(
        default=False,
        metadata={
            "help": (
                "Track D grace window: while an event of this class is in its "
                "ramp-in or plateau phase, the env-side action-rate penalty is "
                "suspended for the affected envs (see "
                "Simulator.get_action_rate_grace_mask and the graced action-"
                "smoothness reward). Default False = no grace contribution."
            )
        },
    )
    action_rate_grace_ramp_in_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Grace the action-rate penalty ONLY during the ramp-in (load "
                "onset / load-shift) of this class's events, not the plateau "
                "(steady carrying/pulling should stay smooth). Implies grace "
                "even when action_rate_grace is False."
            )
        },
    )
    magnitude_scale: float = field(
        default=1.0,
        metadata={
            "help": (
                "STAGE-SCALE knob: global multiplier on every sampled force "
                "AND torque magnitude of this wrench class (applied after "
                "direction/magnitude sampling; 0.0 silences the class without "
                "disabling its scheduler). Exists so the DR curriculum "
                "ladder's stage patcher can ramp a family by rewriting ONE "
                "scalar in resolved_configs.pt (same mechanism as the other "
                "DR stage patches). Track D teacher schedule: chest "
                "0 -> 1/3 -> 2/3 -> 1 of the 120 N cap; wrist payload "
                "0 -> 25 -> 50 -> 75 -> 98 N per hand (= scale of the 98 N "
                "cap). 1.0 (default) = configured ranges apply unscaled."
            )
        },
    )
    downward_cone_deg: float = field(
        default=30.0,
        metadata={
            "help": (
                "For direction_mode='downward_cone': half-angle (degrees) of the "
                "cone around -z the force is sampled within (uniform over the cone "
                "cap). Dominant downward (payload weight) with bounded random sway "
                "up to this angle. Default 30 deg. This is the LIVE value the "
                "sampler reads; if downward_cone_deg_start/end are set it is ramped "
                "in-process by env.on_epoch_end on the magnitude-ramp schedule."
            )
        },
    )
    # ---- Epoch-keyed cone-WIDENING ramp (night13 2026-07-14): the cone half-angle
    # opens from _start (narrow, near-pure-down) to _end (wider, still primarily
    # down) over the SAME schedule as the magnitude ramp (magnitude_ramp_start_epoch
    # / magnitude_ramp_epochs). env.on_epoch_end writes downward_cone_deg =
    # start + (end-start)*frac. Both None (default) = fixed downward_cone_deg. ----
    downward_cone_deg_start: Optional[float] = field(
        default=None,
        metadata={"help": "Cone half-angle at ramp start (deg). None = no cone widening."},
    )
    downward_cone_deg_end: Optional[float] = field(
        default=None,
        metadata={"help": "Cone half-angle at ramp end / terminal (deg). None = no cone widening."},
    )
    # ---- Epoch-keyed gentle magnitude ramp (night13 2026-07-13, mirror of the
    # delay-DR ramp_epochs). Default (None/1.0/0) is an exact no-op. env.on_epoch_end
    # sets magnitude_scale = start + (1-start)*min(1, (epoch - start_epoch)/ramp_epochs),
    # read live by the simulator at each reset/burst -> persistent forces ramp up as
    # episodes turn over, no restarts. ----
    magnitude_ramp_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If set, ramp magnitude_scale from magnitude_start_scale up to 1.0 "
                "over this many epochs (starting at magnitude_ramp_start_epoch). "
                "None (default) = no ramp (magnitude_scale used as-is)."
            )
        },
    )
    magnitude_start_scale: float = field(
        default=1.0,
        metadata={"help": "Starting magnitude_scale at the ramp start epoch (see magnitude_ramp_epochs)."},
    )
    magnitude_ramp_start_epoch: int = field(
        default=0,
        metadata={"help": "Absolute epoch at which the magnitude ramp begins (for resume reproducibility)."},
    )
    # ---- Reference-conditioned payload modulation (night13 t8, 2026-07-15). ----
    # ANTI-CHEAT payload scaling: the bag is HEAVY when the arms are near the
    # chest (short lever arm, mechanically sane) and LIGHT when outstretched. The
    # scale is a function of the REFERENCE motion's wrist->chest distance (the
    # mocap target), which is EXOGENOUS to the policy — the robot cannot change
    # the demonstration by moving, so it cannot shed load by extending its arms.
    # It is rewarded for tracking the reference, so it goes where the load is and
    # must bear it. See simulator.reference_wrench_scale (a PURE function whose
    # ONLY inputs are reference positions — that is what makes the guarantee
    # structural). Defaults are the OFF/no-op path: buffer stays all-ones,
    # forces identical to the pre-feature behavior.
    reference_distance_modulation: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable reference-conditioned payload modulation: scale each "
                "wrist body's force by a function of the REFERENCE motion's "
                "wrist->chest distance (exogenous to the policy => anti-cheat). "
                "False (default) = no modulation (buffer all ones, forces "
                "identical to previous behavior)."
            )
        },
    )
    distance_ref_body: str = field(
        default="torso_link",
        metadata={
            "help": (
                "Chest/anchor body whose REFERENCE position is the near end of "
                "the wrist->chest distance used for modulation."
            )
        },
    )
    distance_wrist_bodies: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "Bodies whose force is modulated by their reference distance to "
                "distance_ref_body. None (default) = use body_names (all "
                "candidate wrench bodies)."
            )
        },
    )
    distance_near_m: float = field(
        default=0.25,
        metadata={
            "help": (
                "Reference wrist->chest distance (m) at or below which the load "
                "is at FULL scale (1.0) — arms near the chest, heavy bag."
            )
        },
    )
    distance_far_m: float = field(
        default=0.55,
        metadata={
            "help": (
                "Reference wrist->chest distance (m) at or above which the load "
                "is at the floor scale (distance_far_scale) — arms outstretched, "
                "light bag. Must be > distance_near_m."
            )
        },
    )
    distance_far_scale: float = field(
        default=0.20,
        metadata={
            "help": (
                "Floor multiplier applied to the force when the reference is "
                "fully outstretched (distance >= distance_far_m). In [0, 1]."
            )
        },
    )
    # ---- Secondary hardening (default OFF). posture_gate ALSO uses the ACTUAL
    # wrist pose (not reference-only), so it is NOT part of the structural
    # anti-cheat guarantee — it is an opt-in belt-and-suspenders term that
    # additionally attenuates the load when the robot braces its wrist far from
    # the reference wrist (a would-be cheat pose). The reference-conditioning
    # above is the primary, provable guarantee; this is secondary. ----
    posture_gate: bool = field(
        default=False,
        metadata={
            "help": (
                "SECONDARY HARDENING (opt-in, uses ACTUAL pose): additionally "
                "scale down the load when the actual wrist is farther than "
                "posture_gate_tol_m from the REFERENCE wrist, so the policy "
                "cannot brace in a cheat pose to relieve load. Default False. "
                "NOTE: unlike reference_distance_modulation this reads the live "
                "robot pose, so it is not part of the structural anti-cheat "
                "guarantee; it is belt-and-suspenders only."
            )
        },
    )
    posture_gate_tol_m: float = field(
        default=0.15,
        metadata={
            "help": (
                "Tolerance (m) for posture_gate: the load stays at full "
                "reference scale while the actual wrist is within this distance "
                "of the reference wrist, then decays linearly to zero at twice "
                "the tolerance. Only used when posture_gate=True."
            )
        },
    )

    def __post_init__(self):
        for name, rng in (
            ("force_magnitude_range", self.force_magnitude_range),
            ("torque_magnitude_range", self.torque_magnitude_range),
        ):
            if rng[0] < 0 or rng[1] < 0:
                raise ValueError(f"{name} values must be non-negative.")
            if rng[0] > rng[1]:
                raise ValueError(f"{name}[0] must be <= {name}[1].")
        for name, rng in (
            ("duration_range", self.duration_range),
            ("interval_range", self.interval_range),
        ):
            if rng[0] <= 0 or rng[1] <= 0:
                raise ValueError(f"{name} values must be positive.")
            if rng[0] > rng[1]:
                raise ValueError(f"{name}[0] must be <= {name}[1].")
        if self.has_wrench() and not self.body_names:
            raise ValueError(
                "body_names must be provided when wrench randomization is enabled."
            )
        if not (0.0 <= self.persistent_fraction <= 1.0):
            raise ValueError("persistent_fraction must be in [0, 1].")
        for name, rng in (
            ("ramp_in_range", self.ramp_in_range),
            ("ramp_out_range", self.ramp_out_range),
        ):
            if rng[0] < 0 or rng[1] < 0:
                raise ValueError(f"{name} values must be non-negative.")
            if rng[0] > rng[1]:
                raise ValueError(f"{name}[0] must be <= {name}[1].")
        if not (0.0 <= self.all_bodies_prob <= 1.0):
            raise ValueError("all_bodies_prob must be in [0, 1].")
        if self.direction_mode not in (
            "uniform", "horizontal", "downward", "downward_cone"
        ):
            raise ValueError(
                "direction_mode must be one of 'uniform', 'horizontal', "
                "'downward', 'downward_cone'."
            )
        if not (0.0 < self.downward_cone_deg <= 90.0):
            raise ValueError("downward_cone_deg must be in (0, 90].")
        for _n in ("downward_cone_deg_start", "downward_cone_deg_end"):
            _v = getattr(self, _n)
            if _v is not None and not (0.0 < _v <= 90.0):
                raise ValueError(f"{_n} must be in (0, 90] (or None).")
        if (self.downward_cone_deg_start is None) != (self.downward_cone_deg_end is None):
            raise ValueError(
                "downward_cone_deg_start and downward_cone_deg_end must both be set "
                "(to enable cone widening) or both None."
            )
        if self.magnitude_ramp_epochs is not None and self.magnitude_ramp_epochs <= 0:
            raise ValueError("magnitude_ramp_epochs must be a positive int (or None).")
        if self.magnitude_start_scale < 0.0:
            raise ValueError("magnitude_start_scale must be non-negative.")
        if self.magnitude_ramp_start_epoch < 0:
            raise ValueError("magnitude_ramp_start_epoch must be >= 0.")
        if self.magnitude_scale < 0.0:
            raise ValueError("magnitude_scale must be non-negative.")
        # Reference-conditioned modulation validation (mirrors the style above).
        if self.distance_near_m < 0.0:
            raise ValueError("distance_near_m must be non-negative.")
        if self.distance_far_m <= 0.0:
            raise ValueError("distance_far_m must be positive.")
        if not (self.distance_near_m < self.distance_far_m):
            raise ValueError("distance_near_m must be < distance_far_m.")
        if not (0.0 <= self.distance_far_scale <= 1.0):
            raise ValueError("distance_far_scale must be in [0, 1].")
        if self.posture_gate_tol_m <= 0.0:
            raise ValueError("posture_gate_tol_m must be positive.")

    def has_wrench(self) -> bool:
        """Check if any wrench magnitude is configured (non-zero)."""
        return self.force_magnitude_range[1] > 0.0 or (
            self.torque_magnitude_range[1] > 0.0
        )


@dataclass
class DelayDomainRandomizationConfig:
    """Actuation / observation latency domain randomization.

    Models sim2real control-loop latency: the actuator applies a PD target from a few
    control steps ago, and the policy observes state from a few control steps ago. A
    per-env integer delay (in CONTROL steps) is sampled uniformly from the configured
    inclusive range at every reset, so the cohort of delays churns across episodes.

    Both ranges default to (0, 0) = exactly the previous (no-delay) behavior. The ranges
    are the natural RAMP axis for an ADR curriculum (widen the upper bound over stages).
    """

    action_delay_steps: Tuple[int, int] = field(
        default=(0, 0),
        metadata={
            "help": "Range (min,max) of control-step delay on the PD target sent to the sim."
        },
    )
    # WARNING: obs-delay output currently feeds ONLY the critic's next_value GAE
    # bootstrap (ppo/agent.py), never the actor -- this injects noise into the
    # critic (which must see clean/privileged obs) and is a no-op for the policy.
    # DISABLED in the night13 recipe until re-wired to the actor's input only.
    # See PR discussion.
    observation_delay_steps: Tuple[int, int] = field(
        default=(0, 0),
        metadata={
            "help": "Range (min,max) of control-step delay on the observation returned to the policy."
        },
    )
    ramp_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If set, linearly ramp the EFFECTIVE max delay (both action and "
                "observation) in from 0 up to the configured max over this many training "
                "epochs: effective_max(epoch) = round(configured_max * min(1, epoch / "
                "ramp_epochs)), applied independently to action_delay_steps[1] and "
                "observation_delay_steps[1]. The configured min (steps[0]) is left "
                "unramped; if effective_max would fall below the configured min, it is "
                "clamped up to the min so the sampled range stays valid. Default None = "
                "off (full configured range from step 0, i.e. current behavior). "
                "Epoch is supplied by BaseEnv.on_epoch_end(current_epoch), which the "
                "agent calls once per epoch identically on every rank (see "
                "protomotions/agents/base_agent/agent.py:845) — the ramp fraction is "
                "therefore a pure function of a globally-lockstep counter, not any "
                "per-rank random or wall-clock state, so all ranks compute identical "
                "effective bounds and never diverge in the DR sampling collective. "
                "With action_delay_probs, the ramp CAPS the distribution's support "
                "instead (see that field's help)."
            )
        },
    )
    action_delay_probs: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": (
                "Optional DISCRETE distribution over per-env action delays "
                "(night13/T3, operator revision 2026-07-13): probs[d] = P(delay=d "
                "control steps), d = 0..len(probs)-1. When set, per-env action "
                "delays are sampled from this distribution at reset INSTEAD of "
                "uniform over action_delay_steps (which is then ignored for "
                "action-delay sampling; observation delay is unaffected). Must "
                "be non-negative and sum to 1 +/- 1e-6. max_action_delay() "
                "becomes len(probs)-1, so the action ring buffer automatically "
                "sizes to hold the largest delay. RAMP semantics: ramp_epochs "
                "caps the distribution's SUPPORT — at epoch E only delays "
                "<= effective_max_action_delay(E) are allowed, and the "
                "probabilities over the allowed delays are renormalized "
                "(truncate + renormalize; the simplest correct adaptation of "
                "the range-cap semantics). Read via getattr(cfg, "
                "'action_delay_probs', None) for pre-field pickles."
            )
        },
    )

    observation_delay_probs: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": (
                "Optional DISCRETE distribution over per-env OBSERVATION delays "
                "(night13/T3 operator revision 2026-07-13): probs[d] = P(obs "
                "delay=d control steps), d = 0..len(probs)-1. Mirror of "
                "action_delay_probs but for the observation-latency axis. When "
                "set, per-env observation delays are sampled from this "
                "distribution at reset INSTEAD of uniform over "
                "observation_delay_steps. Must be non-negative and sum to 1 "
                "+/- 1e-6. max_observation_delay() becomes len(probs)-1. Ramp "
                "semantics identical to action_delay_probs (ramp_epochs caps the "
                "support). Read via getattr(cfg, 'observation_delay_probs', None) "
                "for pre-field pickles."
            )
        },
    )

    def __post_init__(self):
        for name, rng in (
            ("action_delay_steps", self.action_delay_steps),
            ("observation_delay_steps", self.observation_delay_steps),
        ):
            if rng[0] < 0 or rng[1] < 0:
                raise ValueError(f"{name} values must be non-negative.")
            if rng[0] > rng[1]:
                raise ValueError(f"{name}[0] must be <= {name}[1].")
        if self.ramp_epochs is not None and self.ramp_epochs <= 0:
            raise ValueError("ramp_epochs must be a positive int (or None to disable).")
        if self.action_delay_probs is not None:
            probs = self.action_delay_probs
            if len(probs) < 1:
                raise ValueError("action_delay_probs must have at least one entry.")
            if any(p < 0 for p in probs):
                raise ValueError("action_delay_probs entries must be non-negative.")
            total = float(sum(probs))
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"action_delay_probs must sum to 1.0 +/- 1e-6, got {total}."
                )
        if self.observation_delay_probs is not None:
            probs = self.observation_delay_probs
            if len(probs) < 1:
                raise ValueError("observation_delay_probs must have at least one entry.")
            if any(p < 0 for p in probs):
                raise ValueError("observation_delay_probs entries must be non-negative.")
            total = float(sum(probs))
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"observation_delay_probs must sum to 1.0 +/- 1e-6, got {total}."
                )

    def effective_max_action_delay(self, current_epoch: int) -> int:
        return self._ramp(self.max_action_delay(), self.action_delay_steps[0], current_epoch)

    def effective_max_observation_delay(self, current_epoch: int) -> int:
        return self._ramp(
            self.max_observation_delay(), self.observation_delay_steps[0], current_epoch
        )

    def _ramp(self, configured_max: int, configured_min: int, current_epoch: int) -> int:
        if self.ramp_epochs is None:
            return configured_max
        frac = min(1.0, max(0.0, current_epoch) / self.ramp_epochs)
        eff = round(configured_max * frac)
        return max(eff, configured_min)

    def max_action_delay(self) -> int:
        probs = getattr(self, "action_delay_probs", None)
        if probs is not None:
            return len(probs) - 1
        return int(self.action_delay_steps[1])

    def max_observation_delay(self) -> int:
        probs = getattr(self, "observation_delay_probs", None)
        if probs is not None:
            return len(probs) - 1
        return int(self.observation_delay_steps[1])

    def has_action_delay(self) -> bool:
        return self.max_action_delay() > 0

    def has_observation_delay(self) -> bool:
        return self.max_observation_delay() > 0

    def has_delay(self) -> bool:
        return self.has_action_delay() or self.has_observation_delay()


@dataclass
class ProjectileConfig:
    """Configuration for projectile cube throwing (J-key perturbation)."""

    num_projectiles: int = 5
    cube_half_size_range: Tuple[float, float] = (0.05, 0.15)  # per-pool-index size
    density: float = 500.0  # kg/m^3
    speed_range: Tuple[float, float] = (30.0, 40.0)  # m/s (ASE uses 30-40)
    spawn_distance_range: Tuple[float, float] = (4.0, 5.0)  # meters from robot
    spawn_height_range: Tuple[float, float] = (
        -0.65,
        1.1,
    )  # meters relative to robot root
    direction_noise_std: float = 0.1  # std of Gaussian noise added to aim direction
    hide_delay: float = 2.0  # seconds before cube disappears
    hide_z: float = -2.0  # z-position when hidden

    def get_sizes(self) -> list:
        """Return per-pool-index half sizes, linearly interpolated."""
        lo, hi = self.cube_half_size_range
        n = self.num_projectiles
        if n == 1:
            return [lo]
        return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization."""

    action_noise: Optional[ActionNoiseDomainRandomizationConfig] = field(
        default=None, metadata={"help": "Action noise configuration."}
    )
    friction: Optional[FrictionDomainRandomizationConfig] = field(
        default=None, metadata={"help": "Friction randomization configuration."}
    )
    center_of_mass: Optional[CenterOfMassDomainRandomizationConfig] = field(
        default=None, metadata={"help": "Center of mass randomization configuration."}
    )
    object_assets: Optional[ObjectAssetDomainRandomizationConfig] = field(
        default=None,
        metadata={"help": "Scene object asset property randomization configuration."},
    )
    observation_noise: Optional[RobotNoiseConfig] = field(
        default=None,
        metadata={"help": "Observation noise configuration for sim-to-real transfer."},
    )
    push: Optional[PushDomainRandomizationConfig] = field(
        default=None,
        metadata={"help": "Push/perturbation randomization for sim-to-real transfer."},
    )
    wrench: Optional[WrenchDomainRandomizationConfig] = field(
        default=None,
        metadata={
            "help": "Random external force/torque burst randomization for sim-to-real transfer."
        },
    )
    delay: Optional[DelayDomainRandomizationConfig] = field(
        default=None,
        metadata={
            "help": (
                "Actuation / observation latency DR (control-step granularity, per-env, "
                "sampled at reset). Default None = off; existing configs and checkpoints "
                "are unaffected. Read via getattr(dr, 'delay', None) for pre-field pickles."
            )
        },
    )
    sustained_wrench: Optional[WrenchDomainRandomizationConfig] = field(
        default=None,
        metadata={
            "help": (
                "Second, independent wrench class for SUSTAINED (slowly-applied / "
                "quasi-static) external loads — e.g. a human leaning or pulling, "
                "tether/cable drag, carried payload. Same schema as `wrench` but "
                "intended for long duration_range, low magnitudes and high duty "
                "cycle. Runs on its own scheduler; forces from both classes are "
                "SUMMED before the single backend application call (backends "
                "overwrite, they do not accumulate). Default None = off; existing "
                "configs and checkpoints are unaffected."
            )
        },
    )
    mass_scale: Optional[MassScaleDomainRandomizationConfig] = field(
        default=None,
        metadata={
            "help": (
                "Robot body-mass scale DR (MASS-DR, Gary 2026-07-10: mounted "
                "computer/gear -> main-body mass up to 1.3x spec). Per-env "
                "multiplicative scales applied once after robot creation. "
                "Default None = off; read via getattr(dr, 'mass_scale', None) "
                "for pre-field pickles."
            )
        },
    )
    actuator_gain: Optional[ActuatorGainDomainRandomizationConfig] = field(
        default=None,
        metadata={
            "help": (
                "Actuator PD-gain scale DR (GAIN-DR, T7 2026-07-13: motor "
                "unit-to-unit variation / wear). Per-env, per-DOF "
                "multiplicative stiffness/damping (and optional effort-limit) "
                "scales applied once after robot creation. Default None = "
                "off; read via getattr(dr, 'actuator_gain', None) for "
                "pre-field pickles."
            )
        },
    )
    additional_wrenches: List[WrenchDomainRandomizationConfig] = field(
        default_factory=list,
        metadata={
            "help": (
                "Extra independent wrench/persistent-force classes beyond the "
                "two named slots (Track D 2026-07-10: e.g. wrist DRAG pulls "
                "alongside chest forces and wrist payloads). Each entry runs "
                "its own scheduler; all classes' forces are SUMMED before the "
                "single backend application call. Default empty = off."
            )
        },
    )


@dataclass
class SimParams:
    """Configuration for core simulation parameters."""

    fps: int = field(
        default=60, metadata={"help": "Simulation frames per second.", "min": 1}
    )
    decimation: int = field(
        default=4,
        metadata={"help": "Number of physics steps per control step.", "min": 1},
    )


@dataclass
class SimulatorConfig:
    """Main configuration class for the simulator."""

    _target_: str = field(
        default=None, metadata={"help": "Path to the simulator class."}
    )
    w_last: bool = field(
        default=None,
        metadata={"help": "Quaternion format: True for xyzw, False for wxyz."},
    )
    headless: bool = field(
        default=None, metadata={"help": "Run without GUI visualization."}
    )
    num_envs: int = field(
        default=None, metadata={"help": "Number of parallel environments.", "min": 1}
    )
    sim: SimParams = field(
        default=None, metadata={"help": "Simulation parameters (fps, decimation)."}
    )
    experiment_name: str = field(
        default=None, metadata={"help": "Name for this experiment (used for logging)."}
    )
    camera: Optional[Any] = field(
        default=None, metadata={"help": "Camera configuration for rendering."}
    )
    record_viewer: bool = field(
        default=False, metadata={"help": "Record viewer output to video."}
    )
    viewer_record_dir: str = field(
        default="output/recordings/viewer",
        metadata={"help": "Directory for viewer recordings."},
    )
    domain_randomization: Optional[DomainRandomizationConfig] = field(
        default=None,
        metadata={
            "help": "Domain randomization configuration for sim-to-real transfer."
        },
    )
    projectile: ProjectileConfig = field(
        default_factory=lambda: ProjectileConfig(),
        metadata={
            "help": (
                "Projectile cube perturbation configuration (J-key throws). "
                "Set num_projectiles=0 to disable projectiles."
            )
        },
    )
    pd_target_max_accel: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Maximum acceleration (second derivative) of PD targets in rad/step^2. "
                "Limits how quickly the direction of action change can reverse, preventing "
                "oscillatory jerk while allowing large single-step corrections. "
                "None = disabled (no acceleration limit)."
            ),
            "min": 0.0,
        }
    )

    def __post_init__(self):
        assert self._target_ is not None, "SimulatorConfig._target_ must be provided"
        assert self.w_last is not None, "SimulatorConfig.w_last must be provided"
        assert self.headless is not None, "SimulatorConfig.headless must be provided"
        assert self.num_envs is not None, "SimulatorConfig.num_envs must be provided"
        assert self.sim is not None, "SimulatorConfig.sim must be provided"
        assert (
            self.experiment_name is not None
        ), "SimulatorConfig.experiment_name must be provided"


@dataclass
class SimBodyOrdering:
    """Configuration for the ordering of bodies in the simulation."""

    body_names: List[str] = field(
        default_factory=list, metadata={"help": "Ordered list of rigid body names."}
    )
    dof_names: List[str] = field(
        default_factory=list, metadata={"help": "Ordered list of DOF (joint) names."}
    )
