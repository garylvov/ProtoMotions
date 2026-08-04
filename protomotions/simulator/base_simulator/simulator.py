# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base simulator interface for physics engines.

This module defines the abstract base class for physics simulators. It provides a
unified interface across different physics engines (IsaacGym, IsaacLab, Genesis, Newton)
while handling simulator-specific details in subclasses.

Key Classes:
    - Simulator: Abstract base class for all physics simulators

Key Features:
    - Unified robot state representation
    - Multi-simulator support with consistent API
    - PD control and torque control
    - Terrain integration
    - Scene and object management
    - Visualization marker system
    - Domain randomization support
"""

from abc import ABC, abstractmethod
import logging
import math
import os

from typing import Dict, List, Optional, Any, Tuple, Callable

import torch

log = logging.getLogger(__name__)

from protomotions.components.scene_lib import SceneLib
from protomotions.components.terrains.terrain import Terrain
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    DataConversionMapping,
    RootOnlyState,
    ObjectState,
    ResetState,
)
from protomotions.simulator.base_simulator.config import (
    MarkerState,
    VisualizationMarkerConfig,
    SimulatorConfig,
    SimBodyOrdering,
    ActionNoiseDomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
    ObjectAssetDomainRandomizationConfig,
    CenterOfMassDomainRandomizationConfig,
    ProjectileConfig,
    get_matching_indices,
    H1_2_GAIN_DR_GROUP_PATTERNS,
    resolve_gain_dr_groups,
)
from protomotions.robot_configs.base import ControlType, RobotConfig
from protomotions.simulator.base_simulator.record import RecordingMixin
from protomotions.simulator.base_simulator.user_interface import UserInterface


def reference_wrench_scale(
    ref_wrist_pos: torch.Tensor,
    ref_chest_pos: torch.Tensor,
    near_m: float,
    far_m: float,
    far_scale: float,
) -> torch.Tensor:
    """Anti-cheat payload scale from the REFERENCE motion's wrist->chest distance.

    This is the structural anti-cheat guarantee for reference-conditioned payload
    modulation. Its ONLY inputs are REFERENCE (mocap target) positions — it takes
    NO robot/live pose. Because the demonstration is exogenous to the policy, the
    robot cannot change this scale by moving: the load is heavy when the reference
    holds the arms near the chest (short lever arm) and light when the reference
    is outstretched. The policy is rewarded for tracking the reference, so it goes
    where the load is and must bear it; it cannot shed load by extending its arms.

    Modulation is a clamped linear falloff of the reference distance ``d``:
      - ``d <= near_m`` -> scale 1.0 (full load, arms near chest)
      - ``d >= far_m``  -> scale ``far_scale`` (floor, arms outstretched)
      - in between       -> linear interpolation.

    Args:
        ref_wrist_pos: [..., 3] REFERENCE wrist body position(s) (world frame).
        ref_chest_pos: [..., 3] REFERENCE chest/anchor position(s), broadcastable
            to ``ref_wrist_pos``.
        near_m: distance at/below which the scale is 1.0.
        far_m: distance at/above which the scale is ``far_scale`` (far_m > near_m).
        far_scale: floor multiplier in [0, 1].

    Returns:
        Tensor of scales in [far_scale, 1.0], shape = ref_wrist_pos.shape[:-1].
    """
    d = torch.linalg.norm(ref_wrist_pos - ref_chest_pos, dim=-1)
    s = ((far_m - d) / (far_m - near_m)).clamp(far_scale, 1.0)
    return s


class Simulator(RecordingMixin, ABC):
    """Base class for physics simulators.

    Provides a unified interface for different physics engines (IsaacGym, IsaacLab, Genesis, Newton).
    Handles robot spawning, environment setup, scene management, terrain integration,
    and state management. Subclasses implement simulator-specific details while
    maintaining a consistent API.

    Key responsibilities:
    - **Environment setup**: Spawns robots, objects, and terrain
    - **State management**:
        - Getters return RobotState with full rigid body data (FK computed i.e. max coord)
        - Setters accept ResetState with only root + DOF (simulators compute FK from reduced corrd)
    - **Control**: Applies PD control or direct torques
    - **Visualization**: Manages markers and rendering
    - **Data conversion**: Handles ordering differences between simulators

    Args:
        config: Simulator configuration (num_envs, physics params, etc.).
        robot_config: Robot morphology and control configuration.
        terrain: Optional terrain for complex ground surfaces.
        device: PyTorch device for computations.
        scene_lib: Optional scene library for object spawning.
        visualization_markers: Optional markers for visualization.

    Attributes:
        num_envs: Number of parallel environments.
        dt: Simulation timestep.
        robot_state: Current robot state in unified format.

    Example:
        >>> from protomotions.simulator.isaacgym.simulator import IsaacGymSimulator
        >>> sim = IsaacGymSimulator(config, robot_config, device=device)
        >>> sim.reset()
        >>> for _ in range(1000):
        >>>     actions = policy(sim.robot_state)
        >>>     sim.step(actions)
    """

    # -------------------------
    # ⚙️ Group 1: Initialization & Configuration
    # -------------------------
    def __init__(
        self,
        config: SimulatorConfig,
        robot_config: RobotConfig,
        terrain: Optional[Terrain],
        device: torch.device,
        scene_lib: SceneLib,
    ) -> None:
        """Initialize the Simulator shell without creating simulation.

        Creates a minimal simulator shell. The actual simulation is created later
        via _initialize_with_markers() after Env creates visualization markers.

        Args:
            config: Simulator configuration including num_envs and physics parameters.
            robot_config: Robot morphology, control parameters, and asset files.
            terrain: Terrain instance (can be None for some visualizers).
            device: PyTorch device for tensor operations.
            scene_lib: SceneLib instance (always provided, can be empty).
        """
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.config = config
        self.robot_config = robot_config
        self.device = device
        self.scene_lib = scene_lib  # Always provided (empty if no scenes)
        self.terrain = terrain  # Always provided
        self.headless: bool = self.config.headless
        self.num_envs: int = self.config.num_envs

        self.control_type: ControlType = self.robot_config.control.control_type
        self.decimation: int = self.config.sim.decimation
        self.dt: float = self.decimation * 1.0 / self.config.sim.fps

        self._num_bodies: int = self.robot_config.kinematic_info.num_bodies
        self._num_dof: int = self.robot_config.kinematic_info.num_dofs
        self._dof_names: List[str] = self.robot_config.kinematic_info.dof_names
        self._body_names: List[str] = self.robot_config.kinematic_info.body_names
        # Joint limits are now parsed from MJCF by pose_lib.py
        # Simulator-specific limits are only retrieved for verification via _get_simulator_dof_limits_for_verification()

        self._domain_randomization: Dict[str, Any] = (
            self._process_domain_randomization()
        )

        self.user_interface = UserInterface()
        self._register_base_user_interface_keys()

        self._simulation_running: bool = True

        self._init_recording_state()
        self._common_actions = torch.zeros(
            self.num_envs,
            self.robot_config.number_of_actions,
            device=self.device,
            dtype=torch.float,
        )
        self._previous_actions = torch.zeros(
            self.num_envs,
            self.robot_config.number_of_actions,
            device=self.device,
            dtype=torch.float,
        )
        self._prev_prev_actions = torch.zeros(
            self.num_envs,
            self.robot_config.number_of_actions,
            device=self.device,
            dtype=torch.float,
        )
        # Steps since last reset per env, for skipping accel clamp on first 2 steps
        self._steps_since_reset = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        # Two-phase initialization support
        self._initialized = False
        self._visualization_markers: Optional[Dict[str, VisualizationMarkerConfig]] = (
            None
        )

    def _initialize_with_markers(
        self, visualization_markers: Optional[Dict[str, VisualizationMarkerConfig]]
    ) -> None:
        """Finalize simulator initialization with visualization markers.

        Called by Env after it creates task-specific markers. This triggers
        the actual simulation creation in subclasses.

        Args:
            visualization_markers: Visualization markers configuration created by Env
        """
        if self._initialized:
            raise RuntimeError("Simulator already initialized")

        self._visualization_markers = visualization_markers
        # Save original marker configs before simulator-specific init may
        # replace them (e.g. IsaacLab wraps them in its own class)
        self._original_marker_configs = (
            dict(visualization_markers) if visualization_markers else {}
        )
        # Call simulator-specific initialization (subclass implements this)
        self._create_simulation()
        # Setup data conversion and finalize
        self._finalize_setup()
        self._initialized = True

    @abstractmethod
    def _create_simulation(self) -> None:
        """Create the actual simulation environment.

        Subclasses must implement this to create their simulation environments,
        load assets, and prepare for physics simulation. Can access
        self._visualization_markers set by _initialize_with_markers().
        """
        raise NotImplementedError

    # -------------------------
    # 🌄 Group 2: Environment Setup & Configuration
    # -------------------------
    def _finalize_setup(self) -> None:
        """
        Configure internal tensors after the simulation environment is initialized.
        This includes conversion tensors for bodies, DOFs, and contact sensors.
        """
        self._process_control_properties()

        body_ordering = self._get_sim_body_ordering()

        body_convert_to_common = torch.tensor(
            [
                body_ordering.body_names.index(body_name)
                for body_name in self._body_names
            ],
            dtype=torch.long,
            device=self.device,
        )

        body_convert_to_sim = torch.tensor(
            [
                self._body_names.index(body_name)
                for body_name in body_ordering.body_names
            ],
            dtype=torch.long,
            device=self.device,
        )

        dof_convert_to_sim = torch.tensor(
            [self._dof_names.index(dof_name) for dof_name in body_ordering.dof_names],
            dtype=torch.long,
            device=self.device,
        )
        dof_convert_to_common = torch.tensor(
            [body_ordering.dof_names.index(dof_name) for dof_name in self._dof_names],
            dtype=torch.long,
            device=self.device,
        )

        self.data_conversion = DataConversionMapping(
            body_convert_to_common=body_convert_to_common,
            body_convert_to_sim=body_convert_to_sim,
            dof_convert_to_sim=dof_convert_to_sim,
            dof_convert_to_common=dof_convert_to_common,
            sim_w_last=self.config.w_last,
        )

        # Use joint limits from KinematicInfo instead of simulator-specific ones
        # Verify that simulator-specific limits match the parsed ones
        self._verify_joint_limits()

        # Initialize push randomization state
        self._init_push_randomization()

        # Initialize external wrench randomization state
        self._init_wrench_randomization()

        # Initialize projectile system
        self._init_projectiles()

    def _init_push_randomization(self) -> None:
        """Initialize push randomization state buffers."""
        push_cfg = None
        if (
            self.config.domain_randomization is not None
            and self.config.domain_randomization.push is not None
            and self.config.domain_randomization.push.has_push()
        ):
            push_cfg = self.config.domain_randomization.push

        self._push_enabled = push_cfg is not None

        if self._push_enabled:
            self._simulation_time = torch.zeros(self.num_envs, device=self.device)
            self._push_next_time = torch.zeros(self.num_envs, device=self.device)
            self._push_interval_range = push_cfg.push_interval_range
            self._push_max_lin_vel = torch.tensor(
                push_cfg.max_linear_velocity, device=self.device, dtype=torch.float
            )
            self._push_max_ang_vel = torch.tensor(
                push_cfg.max_angular_velocity, device=self.device, dtype=torch.float
            )
            self._schedule_push(torch.arange(self.num_envs, device=self.device))

        # Post-push action-rate grace (Track D 2026-07-10): per-env countdown
        # of control steps during which the action-rate penalty is suspended
        # (getattr: pickles predating the field must keep loading).
        grace_sec = (
            float(getattr(push_cfg, "action_rate_grace_sec", 0.0) or 0.0)
            if push_cfg is not None
            else 0.0
        )
        self._push_grace_steps = (
            max(1, int(round(grace_sec / self.dt))) if grace_sec > 0.0 else 0
        )
        self._push_grace_left = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _schedule_push(self, env_ids: torch.Tensor) -> None:
        """Schedule next push time for specified environments."""
        if not self._push_enabled or len(env_ids) == 0:
            return

        interval_min, interval_max = self._push_interval_range
        random_intervals = (
            torch.rand(len(env_ids), device=self.device) * (interval_max - interval_min)
            + interval_min
        )
        self._push_next_time[env_ids] = (
            self._simulation_time[env_ids] + random_intervals
        )

    def _apply_push_if_due(self) -> None:
        """Check if any environments are due for a push and apply it."""
        if not self._push_enabled:
            return

        due_mask = self._simulation_time >= self._push_next_time
        if not due_mask.any():
            return

        due_env_ids = torch.where(due_mask)[0]
        num_due = len(due_env_ids)

        # Curriculum ramp: live multiplier on the impulse magnitude (env.on_epoch_end
        # ramps push.magnitude_scale). Default 1.0 = unchanged.
        _pscale = getattr(
            self.config.domain_randomization.push, "magnitude_scale", 1.0
        )
        lin_vel = (
            torch.rand(num_due, 3, device=self.device) * 2 - 1
        ) * self._push_max_lin_vel * _pscale
        ang_vel = (
            torch.rand(num_due, 3, device=self.device) * 2 - 1
        ) * self._push_max_ang_vel * _pscale

        # MARIONETTE coupling: soft-plant (low-gain) envs get proportionally
        # weaker pushes. None (byte-identical) unless PM_PERTURB_GAIN_EXP set.
        _gmult = self._perturb_gain_multiplier()
        if _gmult is not None:
            lin_vel = lin_vel * _gmult[due_env_ids, None]
            ang_vel = ang_vel * _gmult[due_env_ids, None]

        self._apply_root_velocity_impulse(lin_vel, ang_vel, due_env_ids)
        if getattr(self, "_push_grace_steps", 0) > 0:
            # Arm the post-push action-rate grace window for the pushed envs.
            self._push_grace_left[due_env_ids] = self._push_grace_steps
        self._schedule_push(due_env_ids)

    # -------------------------
    # External wrench randomization (random force/torque bursts)
    # -------------------------
    def _init_wrench_randomization(self) -> None:
        """Initialize external wrench randomization state buffers.

        Supports up to two independent wrench classes, each with its own
        scheduler state: ``domain_randomization.wrench`` (short hard bursts)
        and ``domain_randomization.sustained_wrench`` (long, low-magnitude
        quasi-static loads — leaning/tether/payload). Their force/torque
        buffers are SUMMED before the single ``_apply_external_wrenches``
        call because backend application overwrites rather than accumulates
        (IsaacLab ``set_external_force_and_torque`` sets the persistent
        buffers; MuJoCo assigns ``xfrc_applied``).

        ``getattr`` is used for ``sustained_wrench`` so configs pickled
        before this field existed (old resolved_configs.pt) keep loading.
        """
        dr = self.config.domain_randomization
        cfgs = []
        if dr is not None:
            for attr in ("wrench", "sustained_wrench"):
                cfg = getattr(dr, attr, None)
                if cfg is not None and cfg.has_wrench():
                    cfgs.append(cfg)
            # Extra classes (Track D: e.g. wrist drag); getattr keeps
            # pre-field pickles loading.
            for cfg in getattr(dr, "additional_wrenches", None) or []:
                if cfg is not None and cfg.has_wrench():
                    cfgs.append(cfg)

        self._wrench_enabled = len(cfgs) > 0
        if not self._wrench_enabled:
            return

        # Scheduler entries: normally one per class, but a class with
        # ``independent_bodies=True`` (Track D ramped payload forces) expands
        # into one entry PER BODY so each body cycles independently —
        # sometimes one wrist loaded, sometimes both (asymmetric/full
        # carries). ``getattr`` keeps pre-field pickles loading.
        entries: List[tuple] = []
        for cfg in cfgs:
            if getattr(cfg, "independent_bodies", False):
                entries.extend((cfg, [name]) for name in cfg.body_names)
            else:
                entries.append((cfg, list(cfg.body_names)))

        # Resolve the UNION of candidate bodies once (single backend body-id
        # list => single application call covers all classes).
        union_names: List[str] = []
        for cfg in cfgs:
            for name in cfg.body_names:
                if name not in union_names:
                    union_names.append(name)
        num_union = self._resolve_wrench_bodies(union_names)

        # Union body name -> column index (over the shared union body layout);
        # used by the env-side reference-conditioned scale setter to map its
        # configured wrist body names onto the wrench force columns.
        self._wrench_union_names = list(union_names)
        # Reference-conditioned payload scale (anti-cheat), per [env, union body].
        # ONES = no modulation (identical to previous behavior); the env pushes
        # wrist-body rows down each step from reference_wrench_scale(...). Applied
        # to the SUMMED per-body force in _summed_wrench_buffers (broadcast over
        # xyz); bodies never modulated stay 1.0 -> unchanged.
        self._wrench_ref_scale = torch.ones(
            self.num_envs, num_union, device=self.device
        )

        # Per-entry scheduler state; force/torque buffers are laid out over
        # the union body list, with each entry restricted to its own columns.
        self._wrench_scheds = []
        for cfg, entry_names in entries:
            cols = torch.tensor(
                [union_names.index(n) for n in entry_names],
                dtype=torch.long,
                device=self.device,
            )
            ramp_in_range = getattr(cfg, "ramp_in_range", (0.0, 0.0))
            ramp_out_range = getattr(cfg, "ramp_out_range", (0.0, 0.0))
            self._wrench_scheds.append(
                {
                    "cfg": cfg,
                    "entry_names": list(entry_names),
                    "cols": cols,
                    "time": torch.zeros(self.num_envs, device=self.device),
                    "next_start": torch.zeros(self.num_envs, device=self.device),
                    "end_time": torch.zeros(self.num_envs, device=self.device),
                    "active": torch.zeros(
                        self.num_envs, dtype=torch.bool, device=self.device
                    ),
                    "forces": torch.zeros(
                        self.num_envs, num_union, 3, device=self.device
                    ),
                    "torques": torch.zeros(
                        self.num_envs, num_union, 3, device=self.device
                    ),
                    # Ramped-profile state (Track D): targets are the sampled
                    # hold-phase wrench; forces/torques hold target * s(t)
                    # with a cosine ease-in/out envelope.
                    "ramped": (
                        ramp_in_range[1] > 0.0
                        or ramp_out_range[1] > 0.0
                        or float(getattr(cfg, "persistent_ramp_in_sec", 0.0) or 0.0) > 0.0
                    ),
                    "target_forces": torch.zeros(
                        self.num_envs, num_union, 3, device=self.device
                    ),
                    "target_torques": torch.zeros(
                        self.num_envs, num_union, 3, device=self.device
                    ),
                    "start_time": torch.zeros(self.num_envs, device=self.device),
                    "ramp_in": torch.zeros(self.num_envs, device=self.device),
                    "ramp_out": torch.zeros(self.num_envs, device=self.device),
                    # Set when wrench values are written outside the update
                    # loop (init/reset persistent cohort) so the next update
                    # pushes them through the single apply call.
                    "dirty": False,
                }
            )
        all_envs = torch.arange(self.num_envs, device=self.device)
        for sched in self._wrench_scheds:
            if self._reset_wrench_class(sched, all_envs):
                sched["dirty"] = True

    def _schedule_wrench(self, sched: dict, env_ids: torch.Tensor) -> None:
        """Schedule the next wrench start time for the given envs of one class."""
        if len(env_ids) == 0:
            return
        interval_min, interval_max = sched["cfg"].interval_range
        random_intervals = (
            torch.rand(len(env_ids), device=self.device)
            * (interval_max - interval_min)
            + interval_min
        )
        sched["next_start"][env_ids] = sched["time"][env_ids] + random_intervals

    def _reset_wrench_class(self, sched: dict, env_ids: torch.Tensor) -> bool:
        """Clear one class's state for env_ids and re-draw its persistent cohort.

        A ``persistent_fraction`` (per-env Bernoulli at every reset, so cohort
        membership churns) of envs gets a wrench sampled ONCE and held for the
        whole episode (``end_time = +inf``, no interval/duration cycling);
        the rest are rescheduled to cycle normally. Returns True if wrench
        values were written (caller must ensure they reach the backend).
        """
        cfg = sched["cfg"]
        sched["forces"][env_ids] = 0.0
        sched["torques"][env_ids] = 0.0
        sched["active"][env_ids] = False
        sched["time"][env_ids] = 0.0
        sched["end_time"][env_ids] = 0.0  # clear stale +inf persistent markers
        sched["target_forces"][env_ids] = 0.0
        sched["target_torques"][env_ids] = 0.0
        sched["start_time"][env_ids] = 0.0
        sched["ramp_in"][env_ids] = 0.0
        sched["ramp_out"][env_ids] = 0.0
        wrote = False
        # getattr: configs pickled before this field existed must keep loading.
        p = getattr(cfg, "persistent_fraction", 0.0) or 0.0
        scale = getattr(cfg, "magnitude_scale", 1.0)
        all_bodies = getattr(cfg, "persistent_all_bodies", False)
        p_ramp_in = float(getattr(cfg, "persistent_ramp_in_sec", 0.0) or 0.0)
        clean_rem = getattr(cfg, "clean_remainder", False)
        mode = getattr(cfg, "direction_mode", "uniform")
        cone = getattr(cfg, "downward_cone_deg", 30.0)
        cycling_ids = env_ids
        if p > 0.0:
            mask = torch.rand(len(env_ids), device=self.device) < p
            persistent_ids = env_ids[mask]
            cycling_ids = env_ids[~mask]
            n = len(persistent_ids)
            if n > 0:
                # Sampled ONCE per episode per (env, body) and held constant
                # (magnitude AND direction fixed for the whole episode).
                if all_bodies:
                    cols = [int(c) for c in sched["cols"].tolist()]  # e.g. BOTH wrists
                else:
                    cols = [int(sched["cols"][torch.randint(len(sched["cols"]), (1,), device=self.device)].item())]
                for col in cols:
                    sched["target_forces"][persistent_ids, col] = (
                        self._sample_wrench_vectors(
                            n, cfg.force_magnitude_range, self.device, mode=mode, cone_deg=cone
                        ) * scale
                    )
                    sched["target_torques"][persistent_ids, col] = (
                        self._sample_wrench_vectors(n, cfg.torque_magnitude_range, self.device) * scale
                    )
                sched["active"][persistent_ids] = True
                sched["end_time"][persistent_ids] = float("inf")
                if p_ramp_in > 0.0:
                    # Smooth onset: start at 0, cosine-ease target*s(t) in over
                    # p_ramp_in sec (envelope pass), then hold constant.
                    sched["forces"][persistent_ids] = 0.0
                    sched["torques"][persistent_ids] = 0.0
                    sched["start_time"][persistent_ids] = sched["time"][persistent_ids]
                    sched["ramp_in"][persistent_ids] = p_ramp_in
                else:
                    sched["forces"][persistent_ids] = sched["target_forces"][persistent_ids]
                    sched["torques"][persistent_ids] = sched["target_torques"][persistent_ids]
                wrote = True
        if clean_rem:
            # Remainder gets NO wrench (clean tracking): never schedule a burst.
            sched["next_start"][cycling_ids] = float("inf")
        else:
            self._schedule_wrench(sched, cycling_ids)
        return wrote

    def _summed_wrench_buffers(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sum force/torque buffers across wrench classes (superposition).

        The summed per-body FORCE is then multiplied by the reference-conditioned
        anti-cheat scale ``_wrench_ref_scale`` (broadcast over xyz). That buffer
        is all-ones unless the env pushes wrist-body rows down from the REFERENCE
        wrist->chest distance (see ``reference_wrench_scale`` /
        ``set_wrench_reference_scale_from_reference``), so with the feature off the
        forces are byte-for-byte identical to the previous behavior. Torques are
        left unmodulated (the payload weight is a force).
        """
        total_f = self._wrench_scheds[0]["forces"].clone()
        total_t = self._wrench_scheds[0]["torques"].clone()
        for sched in self._wrench_scheds[1:]:
            total_f += sched["forces"]
            total_t += sched["torques"]
        ref_scale = getattr(self, "_wrench_ref_scale", None)
        if ref_scale is not None:
            total_f = total_f * ref_scale[..., None]
        # --- Env-gated DR overrides (all no-ops unless the env var is set) ---
        # PM_DR_ENV_FRACTION: restrict the SUMMED wrench+bag force/torque to a
        # random per-env subset (persistent mask, resampled per reset in
        # _reset_wrench_randomization); zero the unmasked (~1-fraction) envs.
        mask = self._wrench_dr_env_mask()
        if mask is not None:
            keep = mask[:, None, None]
            total_f = total_f * keep
            total_t = total_t * keep
        # PM_DR_FORCE_SCALE: runtime constant magnitude multiplier on the
        # applied wrench+bag force/torque (distinct from the per-class epoch
        # ramp); 0.2 = anneal to 20% force.
        fscale = self._wrench_dr_force_scale()
        if fscale is not None:
            total_f = total_f * fscale
            total_t = total_t * fscale
        # MARIONETTE coupling: per-env gain-tied down-scale of the SUMMED
        # wrench+bag force/torque (covers the short-burst wrench class, the
        # sustained wrist-bag class, and any additional classes alike, since
        # all are summed here). None (byte-identical) unless
        # PM_PERTURB_GAIN_EXP is set — see _perturb_gain_multiplier.
        gmult = self._perturb_gain_multiplier()
        if gmult is not None:
            total_f = total_f * gmult[:, None, None]
            total_t = total_t * gmult[:, None, None]
        return total_f, total_t

    def _wrench_dr_force_scale(self):
        """PM_DR_FORCE_SCALE: constant multiplier on the applied wrench+bag
        force/torque. Returns None (a full no-op) when the env var is unset,
        so the production run's applied forces are byte-for-byte unchanged.
        Parsed once and cached."""
        if not hasattr(self, "_pm_dr_force_scale"):
            _v = os.environ.get("PM_DR_FORCE_SCALE")
            self._pm_dr_force_scale = float(_v) if _v else None
        return self._pm_dr_force_scale

    def _wrench_dr_env_mask(self):
        """PM_DR_ENV_FRACTION: persistent per-env boolean mask selecting the
        fraction of envs that receive wrench+bag forces (resampled per reset).
        Returns None (all envs forced -> current behavior) when the env var is
        unset. Motor/friction/mass DR are unaffected. Parsed once and cached."""
        if not hasattr(self, "_pm_dr_env_fraction"):
            _v = os.environ.get("PM_DR_ENV_FRACTION")
            self._pm_dr_env_fraction = float(_v) if _v else None
            self._pm_dr_env_mask = None
        if self._pm_dr_env_fraction is None:
            return None
        if self._pm_dr_env_mask is None:
            self._pm_dr_env_mask = (
                torch.rand(self.num_envs, device=self.device)
                < self._pm_dr_env_fraction
            )
        return self._pm_dr_env_mask

    def _perturb_gain_multiplier(self):
        """MARIONETTE coupling (2026-08-04): per-env perturbation scale tied
        to the episode's sampled actuator-gain scale.

        ``multiplier_e = clamp(g_e ** PM_PERTURB_GAIN_EXP,
        PM_PERTURB_SCALE_MIN, 1.0)`` where ``g_e`` is the per-env geometric
        mean of the GAIN-DR stiffness scales (``env_gain_scale``, stamped by
        ``_process_actuator_gain_domain_randomization`` at sample time — the
        exact scales the episode's plant runs with; GAIN-DR is a static
        per-env assignment). A soft plant cannot resist full-strength pokes:
        unscaled pushes/wrenches on low-gain envs just teach falling.

        Returns None (a FULL no-op — push and wrench paths untouched,
        byte-identical) when PM_PERTURB_GAIN_EXP is unset or 0, or when
        GAIN-DR is inactive (loud WARNING in that case: the knob was set but
        cannot act). Parsed and built once, cached on ``self.device``.
        Knobs: PM_PERTURB_GAIN_EXP (default 0 = OFF; 1.0 = linear coupling),
        PM_PERTURB_SCALE_MIN (default 0.25 = floor of the down-scale).
        """
        if not hasattr(self, "_pm_perturb_gain_mult"):
            _exp = float(os.environ.get("PM_PERTURB_GAIN_EXP", "0") or "0")
            _min = float(os.environ.get("PM_PERTURB_SCALE_MIN", "0.25") or "0.25")
            self._pm_perturb_gain_mult = None
            if _exp != 0.0:
                _dr = getattr(self, "_domain_randomization", None) or {}
                _gain_dr = _dr.get("actuator_gain")
                _g = None if _gain_dr is None else _gain_dr.get("env_gain_scale")
                if _g is None:
                    log.warning(
                        "[marionette] PM_PERTURB_GAIN_EXP=%s is set but "
                        "actuator_gain DR is not active (no per-env gain "
                        "scale) — perturbation-gain coupling is a NO-OP",
                        _exp,
                    )
                else:
                    _mult = torch.clamp(
                        _g.to(device=self.device, dtype=torch.float) ** _exp,
                        min=_min,
                        max=1.0,
                    )
                    self._pm_perturb_gain_mult = _mult
                    log.warning(
                        "[marionette] perturbation-gain coupling ACTIVE: "
                        "exp=%.3f min=%.3f -> multiplier mean/min/max="
                        "%.4f/%.4f/%.4f over %d envs (applies to push "
                        "velocities + wrench/sustained-burst forces+torques)",
                        _exp,
                        _min,
                        _mult.mean().item(),
                        _mult.min().item(),
                        _mult.max().item(),
                        self.num_envs,
                    )
        return self._pm_perturb_gain_mult

    def set_wrench_reference_scale_from_reference(
        self,
        ref_body_pos: torch.Tensor,
        ref_body_names: List[str],
        actual_body_pos: Optional[torch.Tensor] = None,
    ) -> None:
        """Push the anti-cheat payload scale into ``_wrench_ref_scale`` each step.

        The env calls this once per control step with the REFERENCE motion's body
        positions (and their body-name ordering). For every wrench class with
        ``reference_distance_modulation=True`` it computes, per configured wrist
        body, the reference-conditioned scale via the pure ``reference_wrench_scale``
        (REFERENCE wrist->chest distance ONLY) and writes it onto that body's
        column in the shared ``_wrench_ref_scale`` buffer. Bodies never modulated
        stay 1.0. This is the sole source that can lower the load, and it depends
        only on the exogenous demonstration => the policy cannot game it by moving.

        ``actual_body_pos`` is consumed ONLY by the opt-in secondary hardening
        ``posture_gate`` (which, unlike the primary path, reads the live robot pose
        — see the config docstring); leave it None to keep the pure reference path.

        Args:
            ref_body_pos: [num_envs, num_bodies, 3] REFERENCE body positions
                (world frame), ordered as ``ref_body_names``.
            ref_body_names: body-name list indexing ``ref_body_pos`` (and
                ``actual_body_pos``).
            actual_body_pos: optional [num_envs, num_bodies, 3] LIVE body positions
                (only for ``posture_gate``).
        """
        if not getattr(self, "_wrench_enabled", False):
            return
        ref_scale = getattr(self, "_wrench_ref_scale", None)
        if ref_scale is None:
            return
        union_names = getattr(self, "_wrench_union_names", None)
        if not union_names:
            return

        name_to_idx = None
        seen_cfgs = set()
        for sched in self._wrench_scheds:
            cfg = sched["cfg"]
            # Independent-body classes expand into one sched per body but share
            # one cfg; process each unique cfg once.
            if id(cfg) in seen_cfgs:
                continue
            seen_cfgs.add(id(cfg))
            if not getattr(cfg, "reference_distance_modulation", False):
                continue

            if name_to_idx is None:
                name_to_idx = {n: i for i, n in enumerate(ref_body_names)}

            chest_name = cfg.distance_ref_body
            if chest_name not in name_to_idx:
                raise ValueError(
                    f"distance_ref_body '{chest_name}' not found in reference "
                    f"body names for reference-conditioned wrench modulation."
                )
            ref_chest = ref_body_pos[:, name_to_idx[chest_name], :]

            wrist_names = cfg.distance_wrist_bodies or cfg.body_names
            for wname in wrist_names:
                if wname not in union_names:
                    # Not a wrench body -> no force column to scale; skip.
                    continue
                if wname not in name_to_idx:
                    raise ValueError(
                        f"distance_wrist_bodies entry '{wname}' not found in "
                        f"reference body names for wrench modulation."
                    )
                body_idx = name_to_idx[wname]
                ref_wrist = ref_body_pos[:, body_idx, :]
                s = reference_wrench_scale(
                    ref_wrist,
                    ref_chest,
                    cfg.distance_near_m,
                    cfg.distance_far_m,
                    cfg.distance_far_scale,
                )
                # --- Secondary hardening (opt-in; USES ACTUAL POSE) -----------
                # Not part of the structural anti-cheat guarantee: attenuate the
                # load when the ACTUAL wrist braces far (> tol) from the REFERENCE
                # wrist, decaying linearly to 0 at twice the tolerance. Off by
                # default; the reference path above is the primary guarantee.
                if getattr(cfg, "posture_gate", False) and actual_body_pos is not None:
                    tol = cfg.posture_gate_tol_m
                    err = torch.linalg.norm(
                        actual_body_pos[:, body_idx, :] - ref_wrist, dim=-1
                    )
                    gate = (((tol - err) / tol) + 1.0).clamp(0.0, 1.0)
                    s = s * gate
                col = union_names.index(wname)
                ref_scale[:, col] = s

    @staticmethod
    def _sample_wrench_vectors(
        num: int,
        magnitude_range: Tuple[float, float],
        device: torch.device,
        mode: str = "uniform",
        cone_deg: float = 30.0,
    ) -> torch.Tensor:
        """Sample [num, 3] vectors with mode-dependent direction and uniform magnitude.

        Modes (Track D ramped persistent forces, 2026-07-10):
          - ``uniform``: uniform on the sphere (previous behavior; torques
            always use this mode).
          - ``horizontal``: mostly horizontal — z shrunk 4x pre-normalization
            (chest persistent forces / pushes).
          - ``downward``: PAYLOAD simulation — the FULL sampled magnitude is
            locked to constant -z (the payload weight), plus a small
            horizontal component of 10-18% of the magnitude in a random
            per-event direction (constant within the event = temporally
            correlated bag-swing/inertia, re-drawn per event).
        """
        mag_min, mag_max = magnitude_range
        magnitudes = (
            torch.rand(num, 1, device=device) * (mag_max - mag_min) + mag_min
        )
        if mode == "downward":
            h_dir = torch.randn(num, 2, device=device)
            h_dir = h_dir / h_dir.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            h_frac = 0.10 + 0.08 * torch.rand(num, 1, device=device)
            xy = h_dir * h_frac * magnitudes
            z = -magnitudes
            return torch.cat([xy, z], dim=-1)
        if mode == "downward_cone":
            # Unit direction uniform over the spherical cap of half-angle
            # ``cone_deg`` around -z (dominant downward payload + bounded sway),
            # scaled by the sampled magnitude (so |force| == magnitude exactly).
            cos_theta = math.cos(math.radians(cone_deg))
            u = torch.rand(num, 1, device=device)
            cos_a = cos_theta + u * (1.0 - cos_theta)  # in [cos_theta, 1]
            sin_a = (1.0 - cos_a * cos_a).clamp_min(0.0).sqrt()
            phi = (2.0 * math.pi) * torch.rand(num, 1, device=device)
            dirs = torch.cat(
                [sin_a * torch.cos(phi), sin_a * torch.sin(phi), -cos_a], dim=-1
            )
            return dirs * magnitudes
        directions = torch.randn(num, 3, device=device)
        if mode == "horizontal":
            directions[:, 2] *= 0.25
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return directions * magnitudes

    def _update_wrench_randomization(self) -> None:
        """Advance all wrench-class timers; start due wrenches, expire finished ones."""
        if not self._wrench_enabled:
            return
        changed = False
        for sched in self._wrench_scheds:
            cfg = sched["cfg"]
            sched["time"] += self.dt

            # Values written at init/reset (persistent cohort) not yet applied.
            if sched["dirty"]:
                sched["dirty"] = False
                changed = True

            # Expire finished wrenches -> zero their rows.
            expired = sched["active"] & (sched["time"] >= sched["end_time"])
            if expired.any():
                expired_ids = torch.where(expired)[0]
                sched["forces"][expired_ids] = 0.0
                sched["torques"][expired_ids] = 0.0
                sched["active"][expired_ids] = False
                self._schedule_wrench(sched, expired_ids)
                changed = True

            # Start wrenches that are due.
            due = (~sched["active"]) & (sched["time"] >= sched["next_start"])
            if due.any():
                due_ids = torch.where(due)[0]
                num_due = len(due_ids)
                # Choose one candidate body of THIS entry per env (entries
                # from independent_bodies classes have exactly one body).
                body_choice = sched["cols"][
                    torch.randint(len(sched["cols"]), (num_due,), device=self.device)
                ]
                # Stage-scale knob (DR ladder patches this one scalar in
                # resolved_configs.pt to ramp a family across stages).
                scale = getattr(cfg, "magnitude_scale", 1.0)
                forces = self._sample_wrench_vectors(
                    num_due, cfg.force_magnitude_range, self.device,
                    mode=getattr(cfg, "direction_mode", "uniform"),
                    cone_deg=getattr(cfg, "downward_cone_deg", 30.0),
                ) * scale
                torques = self._sample_wrench_vectors(
                    num_due, cfg.torque_magnitude_range, self.device
                ) * scale
                # Only the chosen body gets the wrench; other rows stay 0.
                sched["target_forces"][due_ids] = 0.0
                sched["target_torques"][due_ids] = 0.0
                sched["target_forces"][due_ids, body_choice] = forces
                sched["target_torques"][due_ids, body_choice] = torques
                # Two-hand posture (Track D drag): with all_bodies_prob, some
                # events load ALL candidate bodies together, the sampled
                # magnitude split equally across them.
                all_p = getattr(cfg, "all_bodies_prob", 0.0) or 0.0
                if all_p > 0.0 and len(sched["cols"]) > 1:
                    all_mask = (
                        torch.rand(num_due, device=self.device) < all_p
                    )
                    if all_mask.any():
                        ids = due_ids[all_mask]
                        per_f = forces[all_mask] / len(sched["cols"])
                        per_t = torques[all_mask] / len(sched["cols"])
                        sched["target_forces"][ids] = 0.0
                        sched["target_torques"][ids] = 0.0
                        for col in sched["cols"]:
                            sched["target_forces"][ids, col] = per_f
                            sched["target_torques"][ids, col] = per_t
                dur_min, dur_max = cfg.duration_range
                durations = (
                    torch.rand(num_due, device=self.device) * (dur_max - dur_min)
                    + dur_min
                )
                if sched["ramped"]:
                    # Ramped profile: ease-in U(ramp_in_range) -> hold at the
                    # sampled magnitude for U(duration_range) -> ease-out
                    # U(ramp_out_range). Applied force starts at 0 here; the
                    # envelope pass below writes target * s(t) every step.
                    ri_min, ri_max = getattr(cfg, "ramp_in_range", (0.0, 0.0))
                    ro_min, ro_max = getattr(cfg, "ramp_out_range", (0.0, 0.0))
                    ramp_in = (
                        torch.rand(num_due, device=self.device) * (ri_max - ri_min)
                        + ri_min
                    )
                    ramp_out = (
                        torch.rand(num_due, device=self.device) * (ro_max - ro_min)
                        + ro_min
                    )
                    sched["ramp_in"][due_ids] = ramp_in
                    sched["ramp_out"][due_ids] = ramp_out
                    sched["start_time"][due_ids] = sched["time"][due_ids]
                    sched["end_time"][due_ids] = (
                        sched["time"][due_ids] + ramp_in + durations + ramp_out
                    )
                    sched["forces"][due_ids] = 0.0
                    sched["torques"][due_ids] = 0.0
                else:
                    # Step profile (previous behavior): full magnitude now.
                    sched["forces"][due_ids] = sched["target_forces"][due_ids]
                    sched["torques"][due_ids] = sched["target_torques"][due_ids]
                    sched["end_time"][due_ids] = sched["time"][due_ids] + durations
                sched["active"][due_ids] = True
                changed = True

            # Ramp envelope: write target * s(t) for active ramped envs
            # (persistent cohort has end_time=+inf and ramp_in=0 -> s=1).
            if sched["ramped"]:
                # Include persistent-cohort envs (end_time=+inf) that are easing in
                # (ramp_in>0): they ramp force in then hold (t_left=inf => s_out=1).
                act = sched["active"] & (
                    torch.isfinite(sched["end_time"]) | (sched["ramp_in"] > 0)
                )
                if act.any():
                    t_rel = sched["time"] - sched["start_time"]
                    s_in = torch.ones_like(t_rel)
                    ri = sched["ramp_in"]
                    m_in = act & (ri > 0) & (t_rel < ri)
                    if m_in.any():
                        x = (t_rel[m_in] / ri[m_in]).clamp(0.0, 1.0)
                        s_in[m_in] = 0.5 * (1.0 - torch.cos(torch.pi * x))
                    t_left = sched["end_time"] - sched["time"]
                    s_out = torch.ones_like(t_rel)
                    ro = sched["ramp_out"]
                    m_out = act & (ro > 0) & (t_left < ro)
                    if m_out.any():
                        x = (t_left[m_out] / ro[m_out]).clamp(0.0, 1.0)
                        s_out[m_out] = 0.5 * (1.0 - torch.cos(torch.pi * x))
                    s = torch.minimum(s_in, s_out)
                    rows = torch.where(act)[0]
                    sched["forces"][rows] = (
                        sched["target_forces"][rows] * s[rows, None, None]
                    )
                    sched["torques"][rows] = (
                        sched["target_torques"][rows] * s[rows, None, None]
                    )
                    changed = True

        if changed:
            self._apply_external_wrenches(*self._summed_wrench_buffers())

    def get_action_rate_grace_mask(self):
        """Per-env mask suspending the action-rate penalty (Track D grace).

        True while (a) the post-push grace countdown is running
        (``PushDomainRandomizationConfig.action_rate_grace_sec``) or (b) any
        persistent-force event of a class flagged
        ``action_rate_grace=True`` is in its ramp-in or PLATEAU phase (the
        ease-out is taxed again — the robot should settle smoothly).
        Returns None when no grace source is configured (component treats
        None as no grace).
        """
        pieces = []
        if getattr(self, "_push_grace_steps", 0) > 0:
            pieces.append(self._push_grace_left > 0)
        if getattr(self, "_wrench_enabled", False):
            for sched in self._wrench_scheds:
                cfg = sched["cfg"]
                ramp_in_only = getattr(cfg, "action_rate_grace_ramp_in_only", False)
                if not (getattr(cfg, "action_rate_grace", False) or ramp_in_only):
                    continue
                in_event = sched["active"] & torch.isfinite(sched["end_time"])
                if ramp_in_only:
                    # Load onset / load-shift only: grace during the ramp-in;
                    # steady carrying/pulling stays taxed.
                    t_rel = sched["time"] - sched["start_time"]
                    pieces.append(in_event & (t_rel < sched["ramp_in"]))
                else:
                    # ramp-in + plateau: before end_time - ramp_out.
                    pre_ease_out = sched["time"] < (
                        sched["end_time"] - sched["ramp_out"]
                    )
                    pieces.append(in_event & pre_ease_out)
        if not pieces:
            return None
        mask = pieces[0]
        for p in pieces[1:]:
            mask = mask | p
        return mask

    def _reset_wrench_randomization(self, env_ids: torch.Tensor) -> None:
        """Clear active wrenches (all classes) and reschedule for reset envs."""
        if getattr(self, "_push_grace_steps", 0) > 0 and len(env_ids) > 0:
            self._push_grace_left[env_ids] = 0
        # PM_DR_ENV_FRACTION (env-gated): resample the persistent per-env force
        # mask for the reset envs so cohort membership churns per episode.
        # No-op unless PM_DR_ENV_FRACTION is set (the mask attr stays None).
        pm_mask = getattr(self, "_pm_dr_env_mask", None)
        if pm_mask is not None and len(env_ids) > 0:
            pm_mask[env_ids] = (
                torch.rand(len(env_ids), device=self.device)
                < self._pm_dr_env_fraction
            )
        if not self._wrench_enabled or len(env_ids) == 0:
            return
        need_apply = False
        for sched in self._wrench_scheds:
            was_active = bool(sched["active"][env_ids].any())
            wrote = self._reset_wrench_class(sched, env_ids)
            need_apply = need_apply or was_active or wrote
        if need_apply:
            self._apply_external_wrenches(*self._summed_wrench_buffers())

    def _resolve_wrench_bodies(self, body_names: List[str]) -> int:
        """Resolve configured wrench body names to backend body indices.

        Returns the number of candidate bodies. Backends that support wrench
        DR must override this (and _apply_external_wrenches).
        """
        raise NotImplementedError(
            "Wrench domain randomization is not implemented for this simulator backend."
        )

    def _apply_external_wrenches(
        self, forces: torch.Tensor, torques: torch.Tensor
    ) -> None:
        """Write [num_envs, num_wrench_bodies, 3] force/torque buffers to the sim.

        World-frame wrenches, persistently applied every physics step until
        changed. Backends that support wrench DR must override this.
        """
        raise NotImplementedError(
            "Wrench domain randomization is not implemented for this simulator backend."
        )

    @abstractmethod
    def _apply_root_velocity_impulse(
        self,
        linear_velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Apply velocity impulse to robot root.

        Adds the given velocities to the robot's current root velocities.

        Args:
            linear_velocity: Linear velocity impulse [num_envs, 3] in m/s.
            angular_velocity: Angular velocity impulse [num_envs, 3] in rad/s.
            env_ids: Environment indices to apply impulse to.
        """
        raise NotImplementedError

    # -------------------------
    # Projectile system
    # -------------------------
    @abstractmethod
    def _create_projectiles(self, config: ProjectileConfig) -> None:
        """Create projectile rigid bodies in the simulator.

        Called during _finalize_setup.
        """
        raise NotImplementedError

    @abstractmethod
    def _set_projectile_root_states(
        self,
        proj_indices: torch.Tensor,
        positions: torch.Tensor,
        rotations_xyzw: torch.Tensor,
        velocities: torch.Tensor,
        ang_velocities: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Set root state for specific projectiles in specific envs.

        Args:
            proj_indices: [N] which projectile index per env
            positions: [N, 3]
            rotations_xyzw: [N, 4] quaternion in common format (xyzw)
            velocities: [N, 3]
            ang_velocities: [N, 3]
            env_ids: [N] which environments
        """
        raise NotImplementedError

    @abstractmethod
    def _get_projectile_positions_rotations(
        self,
    ) -> tuple:
        """Return projectile positions and rotations in common format.

        Returns:
            (positions, rotations_xyzw) where:
                positions: [num_envs, num_projectiles, 3]
                rotations_xyzw: [num_envs, num_projectiles, 4]
        """
        raise NotImplementedError

    def _resolve_proj_config(self) -> "ProjectileConfig":
        """Resolve and cache the active projectile config.

        Cached on ``self._proj_config`` so callers get the same instance whether
        invoked from a backend ``__init__`` (which needs the config before scene
        construction) or later from ``_init_projectiles``. To disable projectiles, set
        ``SimulatorConfig.projectile.num_projectiles = 0``.
        """
        if not hasattr(self, "_proj_config"):
            self._proj_config = self.config.projectile
        return self._proj_config

    def _init_projectiles(self) -> None:
        """Initialize projectile pool state and create physics bodies."""
        self._resolve_proj_config()
        N = self._proj_config.num_projectiles

        self._proj_next_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._proj_throw_time = torch.full(
            (self.num_envs, N), float("-inf"), device=self.device
        )
        self._proj_sim_time = torch.zeros(self.num_envs, device=self.device)

        self._create_projectiles(self._proj_config)
        self._hide_all_projectiles()

    def _throw_projectile(self) -> None:
        """J-key handler: launch next projectile cube at each robot.

        Follows ASE humanoid_perturb.py logic:
        1. Spawn at random angle/distance around robot, height relative to root
        2. Aim at robot root with small Gaussian noise on all 3 direction components
        3. Lead the target by adding robot XY velocity to launch velocity
        """
        cfg = self._proj_config
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        cube_idx = self._proj_next_idx.clone()

        robot_state = self._get_simulator_root_state()
        robot_pos = robot_state.root_pos  # [num_envs, 3]

        # Random spawn in polar coords around robot
        angle = torch.rand(self.num_envs, device=self.device) * 2 * math.pi
        dist_min, dist_max = cfg.spawn_distance_range
        distance = (
            torch.rand(self.num_envs, device=self.device) * (dist_max - dist_min)
            + dist_min
        )
        h_min, h_max = cfg.spawn_height_range
        height_offset = (
            torch.rand(self.num_envs, device=self.device) * (h_max - h_min) + h_min
        )

        spawn_pos = robot_pos.clone()
        spawn_pos[:, 0] += torch.cos(angle) * distance
        spawn_pos[:, 1] += torch.sin(angle) * distance
        spawn_pos[:, 2] = robot_pos[:, 2] + height_offset

        # Velocity aimed at robot with slight noise on all 3 components (ASE-style)
        launch_dir = robot_pos - spawn_pos
        launch_dir += cfg.direction_noise_std * torch.randn_like(launch_dir)
        launch_dir = launch_dir / (torch.norm(launch_dir, dim=-1, keepdim=True) + 1e-8)
        speed_min, speed_max = cfg.speed_range
        speed = (
            torch.rand(self.num_envs, 1, device=self.device) * (speed_max - speed_min)
            + speed_min
        )
        velocity = launch_dir * speed

        # Lead the target: add robot XY velocity to projectile velocity
        velocity[:, 0:2] += robot_state.root_vel[:, 0:2]

        rotation = torch.zeros(self.num_envs, 4, device=self.device)
        rotation[:, 3] = 1.0  # identity quaternion xyzw
        ang_vel = torch.zeros(self.num_envs, 3, device=self.device)

        self._set_projectile_root_states(
            cube_idx, spawn_pos, rotation, velocity, ang_vel, all_env_ids
        )

        self._proj_throw_time[all_env_ids, cube_idx] = self._proj_sim_time
        self._proj_next_idx = (cube_idx + 1) % cfg.num_projectiles
        log.info("Projectile thrown (cube indices: %s...)", cube_idx[:4].tolist())

    def _update_projectiles(self) -> None:
        """Timer-based hiding of expired projectiles."""
        self._proj_sim_time += self.dt
        elapsed = self._proj_sim_time.unsqueeze(1) - self._proj_throw_time
        expired_mask = (elapsed > self._proj_config.hide_delay) & (
            self._proj_throw_time > float("-inf")
        )
        if not expired_mask.any():
            return

        env_indices, proj_indices = torch.where(expired_mask)
        hide_pos = torch.zeros(len(env_indices), 3, device=self.device)
        hide_pos[:, 2] = self._proj_config.hide_z
        zero_rot = torch.zeros(len(env_indices), 4, device=self.device)
        zero_rot[:, 3] = 1.0
        zero_vel = torch.zeros(len(env_indices), 3, device=self.device)

        self._set_projectile_root_states(
            proj_indices, hide_pos, zero_rot, zero_vel, zero_vel, env_indices
        )
        self._proj_throw_time[expired_mask] = float("-inf")

    def _reset_projectiles(self, env_ids: torch.Tensor) -> None:
        """Reset projectile state on environment reset."""
        self._proj_sim_time[env_ids] = 0.0
        self._proj_throw_time[env_ids] = float("-inf")
        self._proj_next_idx[env_ids] = 0
        self._hide_projectiles_for_envs(env_ids)

    def _hide_all_projectiles(self) -> None:
        """Move all projectiles underground."""
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._hide_projectiles_for_envs(all_env_ids)

    def _hide_projectiles_for_envs(self, env_ids: torch.Tensor) -> None:
        """Move all projectiles for given envs underground.

        Each (env, proj_idx) slot is given a unique hide position
        ``(env_id * N + proj_idx, 0, hide_z)`` rather than the same world
        ``(0, 0, hide_z)``. With many environments and multiple projectiles,
        colocating every projectile rigid body at one world point causes a
        broadphase / actor-aliasing pathology in PhysX (issue #210) where the
        projectile's hide pose ends up stamped onto unrelated scene-object
        bodies on a subsequent physics step. Spreading by 1m per slot keeps
        each cube actor in a distinct world cell, breaking the aliasing while
        leaving projectiles equally hidden from active gameplay.
        """
        N = self._proj_config.num_projectiles
        num_e = len(env_ids)

        # Expand: each env x each projectile
        env_expanded = env_ids.repeat_interleave(N)
        proj_expanded = torch.arange(N, device=self.device).repeat(num_e)

        hide_pos = torch.zeros(len(env_expanded), 3, device=self.device)
        hide_pos[:, 0] = env_expanded.float() * float(N) + proj_expanded.float()
        hide_pos[:, 2] = self._proj_config.hide_z
        zero_rot = torch.zeros(len(env_expanded), 4, device=self.device)
        zero_rot[:, 3] = 1.0
        zero_vel = torch.zeros(len(env_expanded), 3, device=self.device)

        self._set_projectile_root_states(
            proj_expanded, hide_pos, zero_rot, zero_vel, zero_vel, env_expanded
        )

    def _verify_joint_limits(self) -> None:
        """
        Verify that if we instead load the joint limits from the simulator's internal API,
        they match those parsed from MJCF with pose_lib.py.

        This is useful for verifying that the joint limits are correctly parsed from MJCF.
        It also serves as a sanity check that the simulator's internal API is correctly implemented.
        """
        try:
            # Get simulator's internal joint limits for verification
            sim_lower, sim_upper = self._get_simulator_dof_limits_for_verification()

            # Convert simulator limits to common ordering for comparison
            sim_lower_common = sim_lower[self.data_conversion.dof_convert_to_common]
            sim_upper_common = sim_upper[self.data_conversion.dof_convert_to_common]

            # Get MJCF-parsed limits directly from robot_config
            dof_limits_lower = self.robot_config.kinematic_info.dof_limits_lower.to(
                self.device
            )
            dof_limits_upper = self.robot_config.kinematic_info.dof_limits_upper.to(
                self.device
            )

            # Compare with MJCF-parsed limits
            lower_diff = torch.abs(sim_lower_common - dof_limits_lower)
            upper_diff = torch.abs(sim_upper_common - dof_limits_upper)

            tolerance = 1e-5

            # Check for mismatches and raise errors instead of printing warnings
            for i, dof_name in enumerate(self._dof_names):
                if lower_diff[i] > tolerance:
                    raise ValueError(
                        f"Joint limit mismatch for {dof_name} (lower): "
                        f"MJCF={dof_limits_lower[i]:.4f}, "
                        f"Simulator={sim_lower_common[i]:.4f}"
                    )
                if upper_diff[i] > tolerance:
                    raise ValueError(
                        f"Joint limit mismatch for {dof_name} (upper): "
                        f"MJCF={dof_limits_upper[i]:.4f}, "
                        f"Simulator={sim_upper_common[i]:.4f}"
                    )
        except NotImplementedError:
            raise
        except ValueError:
            raise

    # -------------------------
    # ⏱️ Group 3: Simulation Steps & State Management
    # -------------------------
    def _register_base_user_interface_keys(self) -> None:
        """Register simulator-owned viewer controls.

        Environment and task controls register their own semantic keys
        separately. Duplicate registration fails in UserInterface with a clear
        owner/use message.
        """
        self.user_interface.register_key(
            "Q",
            owner="simulator",
            description="Close simulator viewer",
            on_press=self._request_close,
        )
        self.user_interface.register_key(
            "J",
            owner="simulator",
            description="Throw projectile",
            on_press=self._throw_projectile,
        )
        self.user_interface.register_key(
            "L",
            owner="simulator",
            description="Toggle viewer recording",
            on_press=self._toggle_video_record,
        )
        self.user_interface.register_key(
            ";",
            owner="simulator",
            description="Cancel viewer recording",
            on_press=self._cancel_video_record,
        )
        self.user_interface.register_key(
            "O",
            owner="simulator",
            description="Switch camera target",
            on_press=self._toggle_camera_target,
        )
        self.user_interface.register_key(
            "M",
            owner="simulator",
            description="Toggle visualization markers",
            on_press=self._toggle_markers,
        )

    def _request_close(self) -> None:
        self._simulation_running = False

    def get_previous_actions(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get the previous actions.
        """
        if env_ids is not None:
            return self._previous_actions[env_ids]
        return self._previous_actions

    def get_current_actions(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get the current actions.
        """
        if env_ids is not None:
            return self._common_actions[env_ids]
        return self._common_actions

    def step(
        self,
        common_actions: torch.Tensor,
        markers_callback: Optional[Callable[[], Dict[str, MarkerState]]] = None,
    ) -> None:
        """
        Perform a simulation step by:
          1. Converting common actions to simulator-specific actions.
          2. Stepping the physics simulation.
          3. Updating visualization markers (via callback to get fresh state).
          4. Rendering the environment.

        Args:
            common_actions (torch.Tensor): Action tensor in common format.
            markers_callback (Callable): Optional callback function that returns marker states.
                                        Called after physics step but before rendering.
        """
        # Store the action history (two-step buffer for acceleration clamp)
        self._prev_prev_actions = self._previous_actions.clone()
        self._previous_actions = self._common_actions.clone()
        self.user_interface.begin_step()
        self._common_actions = common_actions.to(self.device)

        # Apply PD target acceleration clamp (limits oscillatory jerk)
        if self.config.pd_target_max_accel is not None:
            self._apply_accel_clamp()

        self._steps_since_reset += 1
        self._physics_step()

        # Update simulation time and apply push randomization
        if self._push_enabled:
            self._simulation_time += self.dt
            self._apply_push_if_due()
        if getattr(self, "_push_grace_steps", 0) > 0:
            self._push_grace_left.clamp_(min=0).sub_(1).clamp_(min=0)

        # Update external wrench randomization (random force/torque bursts)
        self._update_wrench_randomization()

        # Update projectile timers (hide expired cubes)
        self._update_projectiles()

        # Get fresh markers state after physics step
        markers_state = markers_callback() if markers_callback is not None else None
        self._last_markers_state = markers_state
        self._update_markers(markers_state)

        self.render()

    def reset_envs(
        self,
        new_states: ResetState,
        new_object_states: Optional[ObjectState] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Reset the specified environments with the given new states.

        Args:
            new_states: Reset state containing root pose/vel and DOF pos/vel.
                       Simulators will compute FK internally - do NOT provide rigid_body_pos/rot/vel.
            new_object_states: Optional object states.
            env_ids: Tensor of environment ids to reset.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        new_states = new_states.convert_to_sim(self.data_conversion)

        self._previous_actions[env_ids] = 0.0
        self._prev_prev_actions[env_ids] = 0.0
        self._steps_since_reset[env_ids] = 0
        if new_object_states is not None:
            if self.scene_lib.num_objects_per_scene > 0:
                new_object_states = new_object_states.convert_to_sim(
                    self.data_conversion
                )
            else:
                new_object_states = None
        self._set_simulator_env_state(new_states, new_object_states, env_ids)

        # Reset push randomization state for reset environments
        if self._push_enabled:
            self._simulation_time[env_ids] = 0.0
            self._schedule_push(env_ids)

        # Reset external wrench randomization state for reset environments
        self._reset_wrench_randomization(env_ids)

        # Reset projectiles for reset environments
        self._reset_projectiles(env_ids)

    def park_envs(
        self,
        env_ids: torch.Tensor,
        hide_z: float = -50.0,
    ) -> None:
        """Move robot and scene objects for ``env_ids`` far below the terrain.

        Used during evaluation to disable physics for envs that are not being
        evaluated, eliminating their contribution to the PhysX broadphase pair
        budget. Parked bodies sit well below any terrain/object AABB so no
        broadphase pairs are generated, no narrow-phase contacts are computed,
        and no found/lost pair churn occurs. Velocities are zeroed so the
        parked bodies stay put.

        Pre-eval state is restored later via ``BaseEnv.restore_state(snapshot)``,
        which calls ``reset_envs`` over all envs with the saved snapshot.

        Args:
            env_ids: Environment IDs to park. No-op if empty/None.
            hide_z: World z-coordinate to teleport robot roots to. Object roots
                are placed slightly below at ``hide_z - 1.0`` so their AABBs
                cannot overlap with the parked robot's AABB.
        """
        from protomotions.simulator.base_simulator.simulator_state import (
            StateConversion,
        )

        if env_ids is None or env_ids.numel() == 0:
            return

        n = env_ids.numel()
        device = self.device

        # Preserve current root xy so parked envs stay in their own grid cell
        # (avoids stacking all parked envs at world origin, which could exceed
        # PhysX broadphase region limits at large num_envs).
        current_root = self.get_root_state(env_ids)
        park_root_pos = current_root.root_pos.clone()
        park_root_pos[:, 2] = hide_z
        park_root_rot = current_root.root_rot.clone()
        zero_root_vel = torch.zeros((n, 3), device=device, dtype=torch.float32)

        num_dofs = self.robot_config.kinematic_info.num_dofs
        park_dof_pos = (
            self.robot_config.default_dof_pos.unsqueeze(0)
            .repeat(n, 1)
            .to(device=device, dtype=torch.float32)
        )
        park_dof_vel = torch.zeros((n, num_dofs), device=device, dtype=torch.float32)

        park_state = ResetState(
            root_pos=park_root_pos,
            root_rot=park_root_rot,
            root_vel=zero_root_vel,
            root_ang_vel=zero_root_vel.clone(),
            dof_pos=park_dof_pos,
            dof_vel=park_dof_vel,
            state_conversion=StateConversion.COMMON,
        )

        park_object_state = None
        if self.scene_lib.num_objects_per_scene > 0:
            current_obj = self.get_object_root_state(env_ids)
            obj_root_pos = current_obj.root_pos.clone()
            obj_root_pos[..., 2] = hide_z - 1.0
            obj_root_rot = current_obj.root_rot.clone()
            m = self.scene_lib.num_objects_per_scene
            zero_obj_vel = torch.zeros(
                (n, m, 3), device=device, dtype=torch.float32
            )
            park_object_state = ObjectState(
                root_pos=obj_root_pos,
                root_rot=obj_root_rot,
                root_vel=zero_obj_vel,
                root_ang_vel=zero_obj_vel.clone(),
                state_conversion=StateConversion.COMMON,
            )

        self.reset_envs(park_state, park_object_state, env_ids)

    @abstractmethod
    def _set_simulator_env_state(
        self,
        new_states: ResetState,
        new_object_states: Optional[ObjectState] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Apply reset state to simulation environments.

        IMPORTANT: new_states is ResetState with only root + DOF state.
        Simulators must compute forward kinematics internally to update rigid body positions/rotations.
        Never pass or expect full RobotState with rigid_body_pos/rot/vel - those are outputs, not inputs.

        Args:
            new_states: Reset state containing root pose/vel and DOF pos/vel.
            new_object_states: Optional object states.
            env_ids: Tensor of environment IDs to update.
        """
        raise NotImplementedError

    @abstractmethod
    def _physics_step(self) -> None:
        """
        Advance the physics simulation by one step.

        Must be implemented in a simulator-specific manner.
        """
        raise NotImplementedError

    # -------------------------
    # 📊 Group 4: State Getters
    # -------------------------
    def get_default_robot_reset_state(self) -> ResetState:
        """
        Get default reset state for the robot.

        Uses robot_config.default_dof_pos if specified, otherwise zeros.
        Root position uses robot_config.default_root_height for z-axis.
        All velocities are zero.

        Returns:
            ResetState: Default reset state in COMMON format.
        """
        from protomotions.simulator.base_simulator.simulator_state import (
            StateConversion,
        )

        num_envs = self.num_envs
        device = self.device

        # DOF positions: from robot_config (guaranteed to be set in post_init, defaults to zeros)
        dof_pos = (
            self.robot_config.default_dof_pos.unsqueeze(0)
            .repeat(num_envs, 1)
            .to(device)
        )

        # Root pose
        root_pos = torch.zeros(num_envs, 3, device=device, dtype=torch.float32)
        root_pos[:, 2] = self.robot_config.default_root_height
        root_rot = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(num_envs, 1)
        )  # xyzw

        # Zero velocities
        dof_vel = torch.zeros_like(dof_pos)
        root_vel = torch.zeros(num_envs, 3, device=device, dtype=torch.float32)
        root_ang_vel = torch.zeros(num_envs, 3, device=device, dtype=torch.float32)

        return ResetState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            state_conversion=StateConversion.COMMON,
        )

    @abstractmethod
    def _get_sim_body_ordering(self) -> SimBodyOrdering:
        """
        Retrieve the ordering of bodies and DOFs as defined by the simulator.

        Returns:
            SimBodyOrdering: A dictionary with keys 'body_names', 'dof_names',
                                  and 'contact_sensor_body_names'.
        """
        raise NotImplementedError

    def get_root_state(self, env_ids: Optional[torch.Tensor] = None) -> RootOnlyState:
        """
        Retrieve the root state of the simulator as an RootOnlyState.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.

        Returns:
            RootOnlyState: The environment state corresponding to the robot root.
        """
        simulator_root_state: RootOnlyState = self._get_simulator_root_state(env_ids)
        simulator_root_state = simulator_root_state.convert_to_common(
            self.data_conversion
        )
        return simulator_root_state

    @abstractmethod
    def _get_simulator_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RootOnlyState:
        """
        Retrieve the raw simulator root state as an RootOnlyState.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.

        Returns:
            RootOnlyState: The raw environment state for the robot root.
        """
        raise NotImplementedError

    def get_robot_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        """
        Retrieve the simulator's bodies and DOF state as an RobotState.
        """
        bodies_state: RobotState = self.get_bodies_state(env_ids)
        dof_state: RobotState = self.get_dof_state(env_ids)
        contact_state: RobotState = self.get_binary_body_contacts(env_ids)
        dof_forces: torch.Tensor = self.get_dof_forces(env_ids)
        bodies_state.merge_fields_from(dof_state)
        bodies_state.merge_fields_from(contact_state)
        bodies_state.merge_fields_from(dof_forces)
        return bodies_state

    def get_bodies_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        """
        Retrieve the simulator's bodies state as an RobotState.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.

        Returns:
            RobotState: An RobotState instance with rigid body state fields set.
        """
        bodies_state: RobotState = self._get_simulator_bodies_state(env_ids)
        bodies_state = bodies_state.convert_to_common(self.data_conversion)
        return bodies_state

    @abstractmethod
    def _get_simulator_bodies_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the raw simulator bodies state.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.

        Returns:
            RobotState: The raw bodies state (with rigid body fields set).
        """
        raise NotImplementedError

    def get_dof_forces(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        """
        Retrieve the DOF forces from the simulator.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment ids.

        Returns:
            RobotState: RobotState containing DOF forces in the simulator's common ordering.
        """
        simulator_dof_forces = self._get_simulator_dof_forces(env_ids)
        simulator_dof_forces = simulator_dof_forces.convert_to_common(
            self.data_conversion
        )
        return simulator_dof_forces

    @abstractmethod
    def _get_simulator_dof_forces(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the raw simulator DOF forces.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.

        Returns:
            RobotState: The raw DOF forces.
        """
        raise NotImplementedError

    def get_dof_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        """
        Retrieve the simulator's DOF state as an RobotState.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.

        Returns:
            RobotState: An RobotState instance with dof_pos and dof_vel set.
        """
        simulator_dof_state: RobotState = self._get_simulator_dof_state(env_ids)
        simulator_dof_state = simulator_dof_state.convert_to_common(
            self.data_conversion
        )
        return simulator_dof_state

    @abstractmethod
    def _get_simulator_dof_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the raw simulator DOF state.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.

        Returns:
            RobotState: The raw DOF state containing dof_pos and dof_vel.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_simulator_dof_limits_for_verification(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve DOF limits from the simulator's internal API for verification purposes only.

        This method should query the simulator's internal representation of joint limits
        and return them in the simulator's native DOF ordering. These limits are used
        solely for verification against the MJCF-parsed limits and should NOT be used
        for any control or computation purposes.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of (lower_limits, upper_limits)
                                              in the simulator's DOF ordering.
        """
        raise NotImplementedError

    def get_bodies_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the bodies' contact buffer.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment ids.

        Returns:
            torch.Tensor: Tensor containing contact forces for bodies in the common ordering.
        """
        simulator_bodies_contact_forces: RobotState = (
            self._get_simulator_bodies_contact_buf(env_ids)
        )
        simulator_bodies_contact_forces = (
            simulator_bodies_contact_forces.convert_to_common(self.data_conversion)
        )
        return simulator_bodies_contact_forces

    def get_binary_body_contacts(
        self, env_ids: Optional[torch.Tensor] = None, threshold: float = 0.01
    ) -> RobotState:
        """
        Get binary contact flags for specified bodies.

        Converts contact forces to binary contact indicators based on force magnitude.
        This is the canonical method for computing contact states from simulator forces.

        Args:
            body_ids: Indices of bodies to get contacts for [num_bodies]
            threshold: Force magnitude threshold in Newtons (default: 0.01)
            env_ids: Optional environment indices to query

        Returns:
            Binary contact flags [num_envs, num_bodies] as float (0.0 or 1.0)
        """
        contact_state = self.get_bodies_contact_buf(env_ids)
        force_magnitudes = torch.norm(
            contact_state.rigid_body_contact_forces, dim=-1
        )  # [num_envs, num_bodies]
        binary_contacts = (force_magnitudes > threshold).float()
        contact_state.rigid_body_contacts = binary_contacts

        contact_state = contact_state.convert_to_common(self.data_conversion)
        return contact_state

    @abstractmethod
    def _get_simulator_bodies_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Retrieve the raw simulator buffer of bodies' contact forces.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment ids.

        Returns:
            torch.Tensor: Raw bodies contact buffer.
        """
        raise NotImplementedError

    def get_object_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """
        Retrieve the root state of objects in the simulator as an RobotState.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.

        Returns:
            RobotState: The environment state corresponding to objects.
        """
        # No objects: return None
        if self.scene_lib.num_objects_per_scene == 0:
            return None
        simulator_object_root_state: ObjectState = (
            self._get_simulator_object_root_state(env_ids)
        )
        simulator_object_root_state = simulator_object_root_state.convert_to_common(
            self.data_conversion
        )
        return simulator_object_root_state

    @abstractmethod
    def _get_simulator_object_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """
        Retrieve the raw simulator object root state as an RobotState.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.

        Returns:
            RobotState: The raw environment state for object roots.
        """
        raise NotImplementedError

    def get_object_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """
        Retrieve object contact forces from the simulator.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment ids.

        Returns:
            ObjectState: Containing tensor of object contact forces.
        """
        simulator_object_contact_forces = self._get_simulator_object_contact_buf(
            env_ids
        )
        return simulator_object_contact_forces

    @abstractmethod
    def _get_simulator_object_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """
        Retrieve the raw object contact buffer.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment ids.

        Returns:
            ObjectState: Raw object contact forces.
        """
        raise NotImplementedError

    # -------------------------
    # 🎮 Group 5: Control & Computation Methods
    # -------------------------

    @abstractmethod
    def _apply_simulator_pd_targets(self, pd_targets: torch.Tensor) -> None:
        """
        Apply PD position targets using the simulator's internal PD controller.

        Called by _apply_control() when control_type is BUILT_IN_PD.
        pd_targets are already in simulator ordering.

        Args:
            pd_targets (torch.Tensor): PD position targets in simulator DOF ordering.
        """
        raise NotImplementedError

    def _apply_accel_clamp(self) -> None:
        """Clamp PD target acceleration (second derivative) to prevent oscillatory jerk.

        Allows large single-step corrections (high velocity) but limits how fast
        the direction of change can reverse. Back-and-forth oscillation hits the
        clamp every frame; a clean step-change only hits it once.

        Skipped for the first 2 steps after reset (insufficient history).
        Modifies self._common_actions in place.
        """
        max_accel = self.config.pd_target_max_accel
        # Only apply where we have 2+ steps of history
        active = self._steps_since_reset >= 2
        if not active.any():
            return

        delta = self._common_actions - self._previous_actions
        prev_delta = self._previous_actions - self._prev_prev_actions
        accel = delta - prev_delta

        clamped_accel = accel.clamp(-max_accel, max_accel)
        clamped_actions = self._previous_actions + prev_delta + clamped_accel

        # Only apply to envs with enough history
        self._common_actions[active] = clamped_actions[active]

    @abstractmethod
    def _apply_simulator_torques(self, torques: torch.Tensor) -> None:
        """
        Apply torques/forces to DOFs using the simulator's API.

        Called by _apply_control() when control_type is PROPORTIONAL or TORQUE.
        torques are already in simulator ordering.

        Args:
            torques (torch.Tensor): Torques in simulator DOF ordering.
        """
        raise NotImplementedError

    def _apply_control(self) -> None:
        """
        Apply control based on control type.

        Actions are expected to be pre-processed by ActionProcessor in the network:
        - For BUILT_IN_PD/PROPORTIONAL: actions are PD targets (already clamped and mapped)
        - For TORQUE: actions are torques (already clamped and scaled)

        All three control modes are co-located here. Child simulators call this method
        from _physics_step() instead of branching on control_type themselves.
        """
        if self.control_type == ControlType.BUILT_IN_PD:
            targets = self._common_actions
            if (
                self._domain_randomization is not None
                and "action_noise" in self._domain_randomization
            ):
                targets = targets.clone()
                targets[
                    ..., self._domain_randomization["action_noise"]["dof_indices"]
                ] += self._domain_randomization["action_noise"]["action_noise"]

            sim_targets = targets[:, self.data_conversion.dof_convert_to_sim]
            self._apply_simulator_pd_targets(sim_targets)

        elif self.control_type == ControlType.PROPORTIONAL:
            targets = self._common_actions
            if (
                self._domain_randomization is not None
                and "action_noise" in self._domain_randomization
            ):
                targets = targets.clone()
                targets[
                    ..., self._domain_randomization["action_noise"]["dof_indices"]
                ] += self._domain_randomization["action_noise"]["action_noise"]

            common_dof_state = self._get_simulator_dof_state().convert_to_common(
                self.data_conversion
            )
            torques = (
                self._common_p_gains * (targets - common_dof_state.dof_pos)
                - self._common_d_gains * common_dof_state.dof_vel
            )
            torques = torch.clip(
                torques, -self._torque_limits_common, self._torque_limits_common
            )
            sim_torques = torques[:, self.data_conversion.dof_convert_to_sim]
            self._apply_simulator_torques(sim_torques)

        elif self.control_type == ControlType.TORQUE:
            torques = self._common_actions

            if (
                self._domain_randomization is not None
                and "action_noise" in self._domain_randomization
            ):
                torques = torques.clone()
                torques[
                    ..., self._domain_randomization["action_noise"]["dof_indices"]
                ] += self._domain_randomization["action_noise"]["action_noise"]

            torques = torch.clip(
                torques, -self._torque_limits_common, self._torque_limits_common
            )
            sim_torques = torques[:, self.data_conversion.dof_convert_to_sim]
            self._apply_simulator_torques(sim_torques)

        else:
            raise NameError(f"Unknown controller type: {self.control_type}")

    def _process_control_properties(self) -> None:
        """
        Process control properties from robot config.

        Creates tensors for:
        - PD gains (stiffness and damping)
        - Torque/effort limits
        """

        # Initialize tensors
        p_gains = torch.zeros(
            self.robot_config.number_of_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        d_gains = torch.zeros(
            self.robot_config.number_of_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        dof_effort_limits = torch.ones(
            self.robot_config.number_of_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # Populate from robot config
        for dof_name in self.robot_config.kinematic_info.dof_names:
            dof_idx = self.robot_config.kinematic_info.dof_names.index(dof_name)
            dof_info = self.robot_config.control.control_info[dof_name]

            # PD gains
            assert (
                dof_info.stiffness is not None and dof_info.damping is not None
            ), f"PD gains must be defined for DOF {dof_name}"
            p_gains[dof_idx] = dof_info.stiffness
            d_gains[dof_idx] = dof_info.damping

            # Effort limits
            if dof_info.effort_limit is not None:
                dof_effort_limits[dof_idx] = dof_info.effort_limit

        self._common_p_gains = p_gains
        self._common_d_gains = d_gains
        self._torque_limits_common = dof_effort_limits

    def _process_domain_randomization(self) -> None:
        """
        Process domain randomization from the config.
        """
        if self.config.domain_randomization is None:
            return

        domain_randomization_dict: Dict[str, Any] = {}
        if self.config.domain_randomization.action_noise is not None:
            domain_randomization_dict["action_noise"] = (
                self._process_action_noise_domain_randomization(
                    self.config.domain_randomization.action_noise
                )
            )
        if self.config.domain_randomization.friction is not None:
            domain_randomization_dict["friction"] = (
                self._process_friction_domain_randomization(
                    self.config.domain_randomization.friction
                )
            )
        if self.config.domain_randomization.center_of_mass is not None:
            domain_randomization_dict["center_of_mass"] = (
                self._process_center_of_mass_domain_randomization(
                    self.config.domain_randomization.center_of_mass
                )
            )
        if self.config.domain_randomization.object_assets is not None:
            domain_randomization_dict["object_assets"] = (
                self._process_object_asset_domain_randomization(
                    self.config.domain_randomization.object_assets
                )
            )
        # getattr: resolved-config pickles persisted before this field exist.
        if getattr(self.config.domain_randomization, "mass_scale", None) is not None:
            domain_randomization_dict["mass_scale"] = (
                self._process_mass_scale_domain_randomization(
                    self.config.domain_randomization.mass_scale
                )
            )
        # getattr: resolved-config pickles persisted before this field exist.
        if getattr(self.config.domain_randomization, "actuator_gain", None) is not None:
            domain_randomization_dict["actuator_gain"] = (
                self._process_actuator_gain_domain_randomization(
                    self.config.domain_randomization.actuator_gain
                )
            )

        return domain_randomization_dict

    def _process_action_noise_domain_randomization(
        self, domain_randomization: ActionNoiseDomainRandomizationConfig
    ) -> None:
        """
        Process action noise domain randomization.
        """
        dof_indices = get_matching_indices(
            self.robot_config.kinematic_info.dof_names,
            domain_randomization.dof_names,
            domain_randomization.dof_indices,
        )
        num_matching_dofs = len(dof_indices)
        action_noise = (
            torch.rand(self.num_envs, num_matching_dofs, device=self.device)
            * (
                domain_randomization.action_noise_range[1]
                - domain_randomization.action_noise_range[0]
            )
            + domain_randomization.action_noise_range[0]
        )

        noise_dict = {"dof_indices": dof_indices, "action_noise": action_noise}
        return noise_dict

    def _process_friction_domain_randomization(
        self, domain_randomization: FrictionDomainRandomizationConfig
    ) -> None:
        """
        Process friction domain randomization.
        """
        body_indices = get_matching_indices(
            self.robot_config.kinematic_info.body_names,
            domain_randomization.body_names,
            domain_randomization.body_indices,
        )
        num_matching_bodies = len(body_indices)

        static_friction = dynamic_friction = restitution = None

        num_samples = min(self.num_envs, domain_randomization.num_buckets)

        if domain_randomization.static_friction_range is not None:
            static_friction = (
                torch.rand(num_samples, num_matching_bodies)
                * (
                    domain_randomization.static_friction_range[1]
                    - domain_randomization.static_friction_range[0]
                )
                + domain_randomization.static_friction_range[0]
            )
            # # or linspace?
            # static_friction = torch.linspace(domain_randomization.static_friction_range[0], domain_randomization.static_friction_range[1], num_samples)
            # static_friction = static_friction.unsqueeze(1).repeat(1, num_matching_bodies)
        if domain_randomization.dynamic_friction_range is not None:
            dynamic_friction = (
                torch.rand(num_samples, num_matching_bodies)
                * (
                    domain_randomization.dynamic_friction_range[1]
                    - domain_randomization.dynamic_friction_range[0]
                )
                + domain_randomization.dynamic_friction_range[0]
            )
            # # or linspace?
            # dynamic_friction = torch.linspace(domain_randomization.dynamic_friction_range[0], domain_randomization.dynamic_friction_range[1], num_samples)
            # dynamic_friction = dynamic_friction.unsqueeze(1).repeat(1, num_matching_bodies)
        if domain_randomization.restitution_range is not None:
            restitution = (
                torch.rand(num_samples, num_matching_bodies)
                * (
                    domain_randomization.restitution_range[1]
                    - domain_randomization.restitution_range[0]
                )
                + domain_randomization.restitution_range[0]
            )
            # # or linspace?
            # restitution = torch.linspace(domain_randomization.restitution_range[0], domain_randomization.restitution_range[1], num_samples)
            # restitution = restitution.unsqueeze(1).repeat(1, num_matching_bodies)

        friction_dict = {
            "body_indices": body_indices,
            "static_friction": static_friction,
            "dynamic_friction": dynamic_friction,
            "restitution": restitution,
        }
        return friction_dict

    def _process_center_of_mass_domain_randomization(
        self, domain_randomization: CenterOfMassDomainRandomizationConfig
    ) -> None:
        """
        Process center of mass domain randomization.
        """
        body_indices = get_matching_indices(
            self.robot_config.kinematic_info.body_names,
            domain_randomization.body_names,
            domain_randomization.body_indices,
        )
        num_matching_bodies = len(body_indices)
        com_range = domain_randomization.com_range
        com_range_x = com_range["x"]
        com_range_y = com_range["y"]
        com_range_z = com_range["z"]
        com = torch.rand(self.num_envs, num_matching_bodies, 3)
        com[..., 0] = com[..., 0] * (com_range_x[1] - com_range_x[0]) + com_range_x[0]
        com[..., 1] = com[..., 1] * (com_range_y[1] - com_range_y[0]) + com_range_y[0]
        com[..., 2] = com[..., 2] * (com_range_z[1] - com_range_z[0]) + com_range_z[0]

        com_dict = {"body_indices": body_indices, "com": com}
        return com_dict

    def _process_mass_scale_domain_randomization(
        self, domain_randomization: "MassScaleDomainRandomizationConfig"
    ) -> Dict[str, Any]:
        """Sample per-env body-mass scale multipliers (MASS-DR).

        Samples [num_envs, num_matching_bodies] main-body multipliers from
        ``mass_scale_range`` and, when ``all_links_scale_range`` is set,
        [num_envs, num_bodies] all-links multipliers. Backend apply paths
        compose them multiplicatively on the default masses. Logs
        ``[mass-dr]`` sample statistics for dump-verify.
        """
        body_indices = get_matching_indices(
            self.robot_config.kinematic_info.body_names,
            domain_randomization.body_names,
            domain_randomization.body_indices,
        )
        num_matching_bodies = len(body_indices)
        lo, hi = domain_randomization.mass_scale_range
        scales = (
            torch.rand(self.num_envs, num_matching_bodies) * (hi - lo) + lo
        )

        all_links_scales = None
        if domain_randomization.all_links_scale_range is not None:
            alo, ahi = domain_randomization.all_links_scale_range
            num_bodies = len(self.robot_config.kinematic_info.body_names)
            all_links_scales = (
                torch.rand(self.num_envs, num_bodies) * (ahi - alo) + alo
            )

        body_names = [
            self.robot_config.kinematic_info.body_names[i] for i in body_indices
        ]
        print(
            f"[mass-dr] enabled: main_bodies={body_names} range=({lo}, {hi}) "
            f"sampled mean/min/max="
            f"{scales.mean().item():.4f}/{scales.min().item():.4f}/"
            f"{scales.max().item():.4f} over {self.num_envs} envs; "
            f"all_links_range={domain_randomization.all_links_scale_range}"
            + (
                f" all_links mean/min/max={all_links_scales.mean().item():.4f}/"
                f"{all_links_scales.min().item():.4f}/{all_links_scales.max().item():.4f}"
                if all_links_scales is not None
                else ""
            )
        )

        return {
            "body_indices": body_indices,
            "scales": scales,
            "all_links_scales": all_links_scales,
        }

    def _process_actuator_gain_domain_randomization(
        self, domain_randomization: "ActuatorGainDomainRandomizationConfig"
    ) -> Dict[str, Any]:
        """Sample per-env, per-DOF actuator PD-gain scale multipliers (GAIN-DR).

        Samples [num_envs, num_matching_dofs] multiplicative scales for
        stiffness and damping from their respective ranges and, when
        ``effort_limit_scale_range`` is set, an additional independent
        [num_envs, num_matching_dofs] effort-limit scale. Backend apply paths
        compose these multiplicatively on the nominal (config) gains. Logs
        ``[gain-dr]`` sample statistics for dump-verify.

        PER-GROUP (2026-08-04): each of the three axes (stiffness / damping /
        effort limit) also accepts per-joint-group bands
        (``group_*_scale_ranges``) resolved through ``group_dof_patterns``.
        ``constant_damping_ratio`` derives damping from stiffness instead of
        sampling it. All default OFF = byte-identical to the prior behavior.
        """
        dof_indices = get_matching_indices(
            self.robot_config.kinematic_info.dof_names,
            domain_randomization.dof_names,
            domain_randomization.dof_indices,
        )
        num_matching_dofs = len(dof_indices)
        dof_names_matched = [
            self.robot_config.kinematic_info.dof_names[i] for i in dof_indices
        ]

        # PER-GROUP GAIN-DR (2026-08-04): resolve the joint-group partition and
        # build per-COLUMN (per-DOF) lo/hi vectors. When no per-group range is
        # configured these vectors are constant and the sampling expression
        # below is arithmetically identical to the pre-2026-08-04 scalar form
        # (same single torch.rand draw, same RNG consumption) -- so an unset
        # config resumes byte-identically (fork Rule-10).
        group_stiffness = domain_randomization.group_stiffness_scale_ranges or {}
        group_damping = domain_randomization.group_damping_scale_ranges or {}
        group_effort = (
            getattr(domain_randomization, "group_effort_limit_scale_ranges", None)
            or {}
        )
        constant_zeta = bool(
            getattr(domain_randomization, "constant_damping_ratio", False)
        )
        per_group_active = (
            bool(group_stiffness) or bool(group_damping) or bool(group_effort)
        )
        group_columns: Dict[str, List[int]] = {}
        if per_group_active:
            group_patterns = (
                domain_randomization.group_dof_patterns
                or H1_2_GAIN_DR_GROUP_PATTERNS
            )
            group_columns = resolve_gain_dr_groups(dof_names_matched, group_patterns)
            print(
                "[gain-dr] per-group partition: "
                + " ".join(
                    f"{group}({len(cols)})="
                    + str([dof_names_matched[c] for c in cols])
                    for group, cols in group_columns.items()
                )
            )

        s_lo, s_hi = domain_randomization.stiffness_scale_range
        d_lo, d_hi = domain_randomization.damping_scale_range

        def _sample(default_lo, default_hi, overrides):
            """Uniform sample honoring per-group bounds.

            With no overrides this is the LITERAL pre-2026-08-04 scalar
            expression (python-float bounds), not a vectorized equivalent —
            float32 rounds ``u * (hi_vec - lo_vec)`` differently from
            ``u * (hi - lo)`` in the last ulp, and Rule-10 wants byte-identity,
            not near-equality, when the feature is off.
            """
            u = torch.rand(self.num_envs, num_matching_dofs)
            if not overrides:
                return u * (default_hi - default_lo) + default_lo
            lo = torch.full((num_matching_dofs,), float(default_lo))
            hi = torch.full((num_matching_dofs,), float(default_hi))
            for group, (g_lo, g_hi) in overrides.items():
                cols = group_columns.get(group, [])
                if not cols:
                    continue
                lo[cols] = float(g_lo)
                hi[cols] = float(g_hi)
            return u * (hi - lo) + lo

        stiffness_scales = _sample(s_lo, s_hi, group_stiffness)
        if constant_zeta:
            # CONSTANT DAMPING RATIO: zeta = d / (2*sqrt(k*m)); scaling k by s
            # and d by sqrt(s) leaves zeta unchanged. Derived, NOT sampled --
            # so no independent damping draw is consumed in this mode.
            damping_scales = torch.sqrt(stiffness_scales)
        else:
            damping_scales = _sample(d_lo, d_hi, group_damping)

        # EFFORT-LIMIT axis (KINESTHETIC TEACHING 2026-08-04): long-existing
        # field, first wired to env knobs today. A per-group range turns the
        # axis on even when the global range is None, with (1.0, 1.0) as the
        # implicit no-op base so unmentioned groups keep nominal limits.
        effort_limit_scales = None
        if domain_randomization.effort_limit_scale_range is not None or group_effort:
            e_lo, e_hi = domain_randomization.effort_limit_scale_range or (1.0, 1.0)
            effort_limit_scales = _sample(e_lo, e_hi, group_effort)

        dof_names = dof_names_matched
        print(
            f"[gain-dr] enabled: dofs={dof_names} "
            f"stiffness_range=({s_lo}, {s_hi}) sampled mean/min/max="
            f"{stiffness_scales.mean().item():.4f}/{stiffness_scales.min().item():.4f}/"
            f"{stiffness_scales.max().item():.4f} "
            f"damping_range=({d_lo}, {d_hi}) sampled mean/min/max="
            f"{damping_scales.mean().item():.4f}/{damping_scales.min().item():.4f}/"
            f"{damping_scales.max().item():.4f} over {self.num_envs} envs; "
            f"effort_limit_range={domain_randomization.effort_limit_scale_range}"
            + (
                f" effort_limit mean/min/max={effort_limit_scales.mean().item():.4f}/"
                f"{effort_limit_scales.min().item():.4f}/{effort_limit_scales.max().item():.4f}"
                if effort_limit_scales is not None
                else ""
            )
        )

        if per_group_active:
            print(
                "[gain-dr] per-group sampled bands: "
                + " ".join(
                    f"{group}: stiffness="
                    f"{tuple(group_stiffness.get(group, (s_lo, s_hi)))} "
                    f"damping="
                    f"{tuple(group_damping.get(group, (d_lo, d_hi)))} "
                    f"sampled_k_mean={stiffness_scales[:, cols].mean().item():.4f} "
                    f"sampled_d_mean={damping_scales[:, cols].mean().item():.4f}"
                    + (
                        ""
                        if effort_limit_scales is None
                        else (
                            f" effort="
                            f"{tuple(group_effort.get(group, (e_lo, e_hi)))} "
                            f"sampled_e_mean="
                            f"{effort_limit_scales[:, cols].mean().item():.4f}"
                        )
                    )
                    for group, cols in group_columns.items()
                    if cols
                )
                + f" constant_damping_ratio={constant_zeta}"
            )

        # MARIONETTE coupling (2026-08-04): per-env aggregate gain scale
        # g_e = geometric mean of the sampled stiffness scales. Consumed by
        # _perturb_gain_multiplier() to scale push/wrench perturbations DOWN
        # on soft-plant (low-gain) envs. Pure extra dict key — backend apply
        # paths read only the four keys above, so this is byte-identical for
        # every existing consumer.
        #
        # SOURCE (2026-08-04, per-group era): the multiplier is a statement
        # about BALANCE AUTHORITY -- how hard the plant can push back against
        # a shove. With stiff legs and a floppy upper body the all-DOF mean no
        # longer measures that, so an env with nominal legs would get its
        # pushes discounted merely because its arms are compliant, teaching
        # under-perturbed balance. AUTO (env_gain_scale_group=None) therefore
        # keys off the 'legs' group whenever per-group ranges are active and
        # falls back to the all-DOF mean otherwise = today's behavior exactly.
        scale_source = domain_randomization.env_gain_scale_group
        if scale_source is None:
            scale_source = (
                "legs"
                if (per_group_active and group_columns.get("legs"))
                else "all"
            )
        if scale_source == "all":
            scale_columns = list(range(num_matching_dofs))
        else:
            scale_columns = group_columns.get(scale_source, [])
            if not scale_columns:
                raise ValueError(
                    f"env_gain_scale_group='{scale_source}' selects no "
                    f"randomized DOFs (resolved groups: "
                    f"{ {g: len(c) for g, c in group_columns.items()} })."
                )
        env_gain_scale = torch.exp(
            torch.log(stiffness_scales[:, scale_columns]).mean(dim=1)
        )
        if per_group_active:
            print(
                f"[gain-dr] env_gain_scale source='{scale_source}' over "
                f"{len(scale_columns)}/{num_matching_dofs} dofs; mean/min/max="
                f"{env_gain_scale.mean().item():.4f}/"
                f"{env_gain_scale.min().item():.4f}/"
                f"{env_gain_scale.max().item():.4f}"
            )

        return {
            "dof_indices": dof_indices,
            "stiffness_scales": stiffness_scales,
            "damping_scales": damping_scales,
            "effort_limit_scales": effort_limit_scales,
            "env_gain_scale": env_gain_scale,
            "group_columns": group_columns,
            "env_gain_scale_source": scale_source,
        }

    def _process_object_asset_domain_randomization(
        self, domain_randomization: ObjectAssetDomainRandomizationConfig
    ) -> Dict[str, Any]:
        """Sample absolute randomized properties for unique scene object assets."""
        if self.scene_lib.num_scenes() == 0:
            return None

        asset_ids = sorted(
            {
                obj.first_instance_id
                for scene in self.scene_lib.scenes
                for obj in scene.objects
            }
        )
        num_assets = len(asset_ids)
        num_samples = min(self.num_envs, domain_randomization.num_buckets)
        samples = domain_randomization.sample(num_samples, num_assets)

        return {
            "asset_ids": asset_ids,
            "asset_id_to_column": {
                asset_id: column for column, asset_id in enumerate(asset_ids)
            },
            "bucket_ids": torch.arange(self.num_envs, dtype=torch.long) % num_samples,
            "num_buckets": num_samples,
            **samples,
        }

    def _get_object_options_for_randomized_asset(
        self,
        obj,
        env_id: Optional[int] = None,
        bucket_id: Optional[int] = None,
    ):
        """Return object options with object-asset DR overrides applied."""
        if self._domain_randomization is None:
            return obj.options
        object_dr = self._domain_randomization.get("object_assets")
        if object_dr is None:
            return obj.options

        column = object_dr["asset_id_to_column"].get(obj.first_instance_id)
        if column is None:
            return obj.options

        if bucket_id is None:
            if env_id is None:
                raise ValueError("env_id or bucket_id is required for object asset DR.")
            bucket_id = int(object_dr["bucket_ids"][env_id].item())

        overrides = {}
        for field_name in (
            "static_friction",
            "dynamic_friction",
            "restitution",
            "mass",
            "density",
        ):
            values = object_dr[field_name]
            if values is not None:
                overrides[field_name] = float(values[bucket_id, column].item())

        if not overrides:
            return obj.options
        return obj.options.with_asset_property_overrides(overrides)

    def _get_object_center_of_mass_for_randomized_asset(
        self,
        obj,
        env_id: Optional[int] = None,
        bucket_id: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Return absolute local CoM sampled for an object asset, if configured."""
        if self._domain_randomization is None:
            return None
        object_dr = self._domain_randomization.get("object_assets")
        if object_dr is None or object_dr.get("center_of_mass") is None:
            return None

        column = object_dr["asset_id_to_column"].get(obj.first_instance_id)
        if column is None:
            return None

        if bucket_id is None:
            if env_id is None:
                raise ValueError("env_id or bucket_id is required for object asset DR.")
            bucket_id = int(object_dr["bucket_ids"][env_id].item())

        return object_dr["center_of_mass"][bucket_id, column]

    def _num_object_asset_randomization_buckets(self) -> int:
        """Return number of object asset DR buckets, or one when disabled."""
        if self._domain_randomization is None:
            return 1
        object_dr = self._domain_randomization.get("object_assets")
        if object_dr is None:
            return 1
        return object_dr["num_buckets"]

    # -------------------------
    # 🎨 Group 6: Rendering & Visualization (abstract methods only)
    # -------------------------
    # Non-abstract rendering methods (render, _toggle_camera_target,
    # _toggle_video_record, _cancel_video_record, _toggle_markers,
    # _update_markers, _build_markers_save_data, _build_objects_save_data)
    # are provided by RecordingMixin in record.py.

    @abstractmethod
    def _write_viewport_to_file(self, file_name: str) -> None:
        """
        Write the current viewport to a file.
        """
        raise NotImplementedError

    @abstractmethod
    def _init_camera(self) -> None:
        """
        Initialize the camera for visualization.

        Must be implemented in a simulator-specific manner.
        """
        raise NotImplementedError

    @abstractmethod
    def _update_simulator_markers(
        self, markers_state: Optional[Dict[str, MarkerState]] = None
    ) -> None:
        """
        Simulator-specific update of marker states.

        Args:
            markers_state (Dict[str, MarkerState]): Dictionary containing marker states.
        """
        raise NotImplementedError

    def is_simulation_running(self) -> bool:
        """
        Check if the simulation is running.
        """
        return self._simulation_running

    def close(self) -> None:
        """
        Close the simulator and perform cleanup operations.
        """
        self._simulation_running = False
