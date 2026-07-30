# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ONNX export for BeyondMimic tracker policies.

Exports a ProtoMotions BeyondMimic tracker policy to a unified ONNX model
**without** running a simulator. The script auto-detects actor observation
keys from the checkpoint's agent config.

Typical actor obs for BM configs::

    noisy_reduced_coords_obs
    noisy_mimic_reduced_coords_target_poses
    historical_previous_processed_actions

The obs function ``build_reduced_coords_target_poses`` uses anchor-body
references (``future_anchor_rot/pos/vel/ang_vel``) rather than full-body
``future_rot``, and an action-history input is included.

Usage
-----
::

    python deployment/export_bm_tracker_onnx.py \\
        --checkpoint exps/my_exp/last.ckpt \\
        --output     exps/my_exp/compiled_models/

Outputs
-------
``<output>/unified_pipeline.onnx``
    The exported ONNX model.

``<output>/unified_pipeline.yaml``
    Rich metadata / deployment contract.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ---------------------------------------------------------------------------
# MockContext
# ---------------------------------------------------------------------------


class _MockState:
    """Mock for CurrentStateView."""

    def __init__(self, num_envs: int, num_dofs: int, num_bodies: int, anchor_idx: int):
        import torch
        import torch.nn.functional as F

        # Actor obs fields
        self.dof_pos = torch.randn(num_envs, num_dofs)
        self.dof_vel = torch.randn(num_envs, num_dofs)
        self.anchor_rot = F.normalize(torch.randn(num_envs, 4), dim=-1)
        self.anchor_pos = torch.randn(num_envs, 3)
        self.root_local_ang_vel = torch.randn(num_envs, 3)
        # Critic obs fields
        self.rigid_body_pos     = torch.randn(num_envs, num_bodies, 3)
        self.rigid_body_rot     = F.normalize(torch.randn(num_envs, num_bodies, 4), dim=-1)
        self.rigid_body_vel     = torch.randn(num_envs, num_bodies, 3)
        self.rigid_body_ang_vel = torch.randn(num_envs, num_bodies, 3)


class _MockMimic:
    """Mock for MimicContext."""

    def __init__(
        self,
        num_envs: int,
        num_future_steps: int,
        num_dofs: int,
        num_bodies: int,
    ):
        import torch
        import torch.nn.functional as F

        # Full-body arrays (used by max_coords obs or older configs)
        self.future_rot     = F.normalize(
            torch.randn(num_envs, num_future_steps, num_bodies, 4), dim=-1
        )
        self.future_pos     = torch.randn(num_envs, num_future_steps, num_bodies, 3)
        self.future_vel     = torch.randn(num_envs, num_future_steps, num_bodies, 3)
        self.future_ang_vel = torch.randn(num_envs, num_future_steps, num_bodies, 3)
        self.future_dof_pos = torch.randn(num_envs, num_future_steps, num_dofs)
        self.future_dof_vel = torch.randn(num_envs, num_future_steps, num_dofs)

        # Anchor-body arrays (used by reduced_coords_target_poses)
        self.future_anchor_rot     = F.normalize(
            torch.randn(num_envs, num_future_steps, 4), dim=-1
        )
        self.future_anchor_pos     = torch.randn(num_envs, num_future_steps, 3)
        self.future_anchor_vel     = torch.randn(num_envs, num_future_steps, 3)
        self.future_anchor_ang_vel = torch.randn(num_envs, num_future_steps, 3)

        # Current reference anchor position
        self.ref_anchor_pos = torch.randn(num_envs, 3)


class _MockHistorical:
    """Mock for HistoricalContext (action history)."""

    def __init__(self, num_envs: int, history_steps: int, num_dofs: int):
        import torch

        self.processed_actions = torch.randn(num_envs, history_steps, num_dofs)


class MockContext:
    """Minimal stand-in for EnvContext used only during ONNX export tracing."""

    def __init__(
        self,
        num_envs: int,
        num_dofs: int,
        num_bodies: int,
        num_future_steps: int,
        anchor_idx: int,
        history_steps: int = 1,
    ):
        import torch

        self.current = _MockState(num_envs, num_dofs, num_bodies, anchor_idx)
        # noisy.*: observation-noise view of the robot state. At inference the
        # noise is unconfigured, so the noisy view aliases the clean tensors
        # (see protomotions.envs.obs.observation_noise.NoisyObservations) --
        # the same alias is correct for export tracing.
        self.noisy = self.current
        self.mimic   = _MockMimic(num_envs, num_future_steps, num_dofs, num_bodies)
        self.historical = _MockHistorical(num_envs, history_steps, num_dofs)
        # body_contacts: used by max_coords_obs observe_contacts
        self.body_contacts  = torch.zeros(num_envs, num_bodies, dtype=torch.bool)
        # ground_heights: used by max_coords_obs root_height_obs
        self.ground_heights = torch.zeros(num_envs)


# ---------------------------------------------------------------------------
# Arm-gain override resolution (PM_ARM_*)
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS
# ---------------
# Training resumes can hot-swap arm PD gains / effort limits in memory via
# PM_ARM_{KP,KD,EFFORT[_SHOULDER|_ELBOW|_WRIST]} env vars (see the resume path
# in protomotions/train_agent.py, "RESUME override arm ..."). That mutation is
# IN-MEMORY ONLY: resolved_configs*.pt on disk keeps the ORIGINAL launch gains.
# This exporter reads the frozen pickle, so without the logic below every
# post-hot-swap export writes a unified_pipeline.yaml whose `control` section
# carries the stale pre-swap gains — downstream MuJoCo evals then run the wrong
# PD controller (e.g. canonical_teacher_20260729_v54: arms trained with
# KP=40/KD=2.0 + effort 120/30 since ep3130, but the ep4000 yaml said
# KP=19.739/KD=1.257).
#
# WHAT IT DOES
# ------------
# Replicates train_agent.py's resume hot-swap on the baked per-DOF
# control_info: for each arm joint group, if an override value is found, set
# stiffness / damping / effort_limit before the yaml gain lists are built.
#
# Override sources, in priority order (first hit per key wins):
#   1. Real process env vars (identical names/semantics to train_agent.py and
#      robot_configs/h1_2.py: per-group PM_ARM_EFFORT_WRIST beats uniform
#      PM_ARM_EFFORT, etc.).
#   2. File named by $PM_ARM_OVERRIDES_FILE (KEY=VALUE lines, # comments ok).
#   3. `pm_arm_overrides.env` sitting next to the checkpoint.
#   4. Trend-loop fallback: the eval loop exports from a snapshot dir named
#      `snap_<run>_ep<N>` that contains only a copied ckpt+config, so also try
#      `<ProtoMotions>/results/<run>/pm_arm_overrides.env`. This lets a marker
#      file dropped once in the training results dir correct every future
#      export from the already-running trend loop (which has no PM_ARM_* env).
#
# Complete no-op when no source yields any PM_ARM_* key (pre-swap runs).
#
# DELIBERATELY NOT TOUCHED: env_config.action_config. Its tensors
# (pd_action_offset / action_scale / stiffness / damping fed to bm_pd_action)
# are frozen from the ORIGINAL launch during resume training too — the policy
# trained against the stale action mapping, so the exported ONNX must keep it.
# Only the yaml `control`/`default_joint_*` sections (what the deployment-side
# PD controller uses) reflect the hot-swapped gains.

_ARM_GROUP_PATTERNS = {
    # Mirrors _arm_groups in train_agent.py's resume hot-swap.
    "SHOULDER": r".*_shoulder_(pitch|roll|yaw)_joint$",
    "ELBOW": r".*_elbow_joint$",
    "WRIST": r".*_wrist_(roll|pitch|yaw)_joint$",
}


def _read_env_file(path) -> dict:
    """Parse a simple KEY=VALUE env-dump file; returns {} if unreadable."""
    out = {}
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def _resolve_arm_overrides(checkpoint_dir: Path) -> dict:
    """Merge PM_ARM_* keys from all sources (env > file tiers, see above)."""
    import os
    import re

    merged: dict = {}
    # Lowest priority first; later updates overwrite earlier ones.
    # Tier 4: trend-loop snapshot dir -> original training results dir.
    m = re.match(r"^snap_(.+)_ep\d+$", checkpoint_dir.name)
    if m:
        results_dir = Path(__file__).resolve().parent.parent / "results" / m.group(1)
        merged.update(_read_env_file(results_dir / "pm_arm_overrides.env"))
    # Tier 3: marker file next to the checkpoint (training results dir itself).
    merged.update(_read_env_file(checkpoint_dir / "pm_arm_overrides.env"))
    # Tier 2: explicitly named file.
    if os.environ.get("PM_ARM_OVERRIDES_FILE"):
        merged.update(_read_env_file(os.environ["PM_ARM_OVERRIDES_FILE"]))
    # Tier 1: real environment (only PM_ARM_* keys).
    for k, v in os.environ.items():
        if k.startswith("PM_ARM_"):
            merged[k] = v
    return {k: v for k, v in merged.items() if k.startswith("PM_ARM_") and v}


def _apply_arm_control_overrides(robot_config, checkpoint_dir: Path) -> None:
    """Apply PM_ARM_* overrides to the baked per-DOF control_info in place.

    train_agent.py mutates the regex-keyed override_control_info and then
    rebuilds control_info from the MJCF; here we already hold the fully baked
    per-joint-name control_info from the pickle, so mutating the matching
    entries directly is equivalent and avoids re-parsing the MJCF.
    """
    import re

    ov = _resolve_arm_overrides(checkpoint_dir)
    if not ov:
        return  # no overrides anywhere -> exact legacy behavior

    ctrl = getattr(robot_config, "control", None)
    ci = getattr(ctrl, "control_info", None)
    if not ci:
        log.warning("PM_ARM overrides present but no baked control_info; skipped")
        return

    for joint_name, entry in ci.items():
        for grp, pat in _ARM_GROUP_PATTERNS.items():
            if not re.match(pat, joint_name):
                continue
            for field, base in (
                ("stiffness", "PM_ARM_KP"),
                ("damping", "PM_ARM_KD"),
                ("effort_limit", "PM_ARM_EFFORT"),
            ):
                # Per-group override beats the uniform one (same rule as
                # train_agent.py / h1_2.py).
                val = ov.get(f"{base}_{grp}") or ov.get(base)
                if not val:
                    continue
                old = getattr(entry, field, None)
                setattr(entry, field, float(val))
                if float(val) != old:
                    log.warning(
                        f"EXPORT arm-gain override {joint_name}.{field}: "
                        f"{old} -> {float(val)} (from {base}[_{grp}])"
                    )


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------


def export_tracker(
    checkpoint: str,
    output_dir: str,
    validate: bool = True,
) -> Path:
    """Export the tracker policy to a unified ONNX model.

    Parameters
    ----------
    checkpoint:
        Path to ``last.ckpt`` (or any ``*.ckpt``).
    output_dir:
        Directory where ONNX + YAML files will be written.
    validate:
        If True, run onnxruntime validation after export.

    Returns
    -------
    Path to the exported ``.onnx`` file.
    """
    import torch
    import torch.nn.functional as F
    from tensordict import TensorDict

    from protomotions.utils.export_utils import (
        ObservationExportModule,
        ActionExportModule,
        UnifiedPipelineModule,
    )
    from protomotions.utils.hydra_replacement import get_class

    checkpoint_path = Path(checkpoint)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load resolved configs (no simulator import required)
    # ------------------------------------------------------------------
    resolved_path = checkpoint_path.parent / "resolved_configs_inference.pt"
    if not resolved_path.exists():
        log.warning(
            "resolved_configs_inference.pt not found, falling back to "
            "resolved_configs.pt.  Domain randomization may still be active!"
        )
        resolved_path = checkpoint_path.parent / "resolved_configs.pt"
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Could not find resolved_configs*.pt in {checkpoint_path.parent}"
        )

    log.info(f"Loading configs from {resolved_path}")
    resolved = torch.load(resolved_path, map_location="cpu", weights_only=False)

    robot_config    = resolved["robot"]
    env_config      = resolved["env"]
    agent_config    = resolved["agent"]
    simulator_config = resolved.get("simulator")

    # Re-apply any in-memory arm-gain hot-swap the training resume performed
    # (PM_ARM_* env vars / pm_arm_overrides.env marker file) BEFORE the yaml
    # gain lists below are read from control_info. No-op when nothing is set.
    # See the block comment above _apply_arm_control_overrides for rationale.
    _apply_arm_control_overrides(robot_config, checkpoint_path.parent)

    # ------------------------------------------------------------------
    # 2. Auto-detect actor obs keys from agent config
    # ------------------------------------------------------------------
    actor_in_keys = list(agent_config.model.actor.in_keys)
    actor_obs_keys = set(actor_in_keys)
    log.info(f"Auto-detected actor obs keys: {actor_in_keys}")

    # ------------------------------------------------------------------
    # 3. Extract dimensions from configs
    # ------------------------------------------------------------------
    num_dofs   = robot_config.kinematic_info.num_dofs
    num_bodies = len(robot_config.kinematic_info.body_names)
    body_names = list(robot_config.kinematic_info.body_names)
    joint_names = list(robot_config.kinematic_info.dof_names)
    anchor_body_name = robot_config.anchor_body_name
    anchor_body_index = robot_config.anchor_body_index
    root_body_index = 0  # pelvis is always first body

    mimic_ctrl_cfg = env_config.control_components.get("mimic")
    if mimic_ctrl_cfg is None:
        raise ValueError("env_config.control_components must contain 'mimic'")
    raw_future_steps = mimic_ctrl_cfg.future_steps
    if isinstance(raw_future_steps, int):
        num_future_steps = raw_future_steps
        future_step_indices = list(range(1, raw_future_steps + 1))
    else:
        num_future_steps = len(raw_future_steps)
        future_step_indices = list(raw_future_steps)

    # Detect action history steps from obs component config
    history_steps = 1
    for k in actor_obs_keys:
        comp = env_config.observation_components.get(k)
        if comp is not None and hasattr(comp, "static_params"):
            sp = comp.static_params if isinstance(comp.static_params, dict) else {}
            if "history_steps" in sp:
                history_steps = sp["history_steps"]

    # Resolve MuJoCo-specific timing.
    control_dt = 0.02
    physics_dt = 0.001
    decimation = 20
    pd_target_max_accel = None
    if simulator_config is not None:
        try:
            from protomotions.simulator.factory import update_simulator_config_for_test
            mj_sim_cfg = update_simulator_config_for_test(
                current_simulator_config=simulator_config,
                new_simulator="mujoco",
                robot_config=robot_config,
            )
            physics_dt = 1.0 / mj_sim_cfg.sim.fps
            decimation = mj_sim_cfg.sim.decimation
            control_dt = physics_dt * decimation
        except Exception as e:
            log.warning(f"Could not apply sim2sim conversion: {e}")
            sim_cfg = getattr(simulator_config, "sim", None)
            if sim_cfg is not None:
                _fps = getattr(sim_cfg, "fps", None)
                _dec = getattr(sim_cfg, "decimation", None)
                if _fps and _dec:
                    physics_dt = 1.0 / _fps
                    decimation = _dec
                    control_dt = physics_dt * decimation
        _accel = getattr(simulator_config, "pd_target_max_accel", None)
        if _accel is not None:
            pd_target_max_accel = float(_accel)

    log.info(
        f"Robot: {num_dofs} DOFs, {num_bodies} bodies, "
        f"anchor={anchor_body_name}(idx={anchor_body_index})"
    )
    log.info(
        f"Timing: control_dt={control_dt}s  physics_dt={physics_dt}s  "
        f"decimation={decimation}"
    )
    log.info(f"Future steps: {future_step_indices} ({num_future_steps} total)")

    # ------------------------------------------------------------------
    # 4. Build MockContext for ONNX tracing shape inference
    # ------------------------------------------------------------------
    mock = MockContext(
        num_envs=1,
        num_dofs=num_dofs,
        num_bodies=num_bodies,
        num_future_steps=num_future_steps,
        anchor_idx=anchor_body_index,
        history_steps=history_steps,
    )

    # ------------------------------------------------------------------
    # 5. Build ObservationExportModule (actor obs only, for export)
    # ------------------------------------------------------------------
    actor_obs_configs = {
        k: v
        for k, v in env_config.observation_components.items()
        if k in actor_obs_keys
    }
    missing = actor_obs_keys - set(env_config.observation_components.keys())
    if missing:
        raise ValueError(
            f"Actor requires obs keys {actor_obs_keys} but these are missing "
            f"from env_config.observation_components: {missing}"
        )

    log.info(f"Observation components for export: {list(actor_obs_configs.keys())}")
    obs_module = ObservationExportModule(actor_obs_configs, mock, device="cpu")
    obs_module.eval()

    obs_input_keys  = obs_module.get_input_keys()
    obs_output_keys = obs_module.get_output_keys()
    log.info(f"  Context input keys:  {obs_input_keys}")
    log.info(f"  Observation outputs: {obs_output_keys}")

    # ------------------------------------------------------------------
    # 6. Reconstruct actor-only and load weights
    # ------------------------------------------------------------------
    log.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    ActorClass = get_class(agent_config.model.actor._target_)
    actor = ActorClass(agent_config.model.actor)

    # nn.LazyLinear needs one forward pass to materialise concrete layer shapes.
    actor.eval()
    mock_obs_inputs = [_resolve_attr_path(k, mock) for k in obs_input_keys]
    with torch.no_grad():
        mock_obs_out = obs_module(*mock_obs_inputs)
        mock_obs_td = TensorDict(
            {k: v for k, v in zip(obs_output_keys, mock_obs_out)},
            batch_size=[1],
        )
        actor(mock_obs_td)  # materialises LazyLinear layers

    log.info(
        "LazyLinear materialised with actor obs: "
        + ", ".join(f"{k}={v.shape[-1]}" for k, v in mock_obs_td.items())
    )

    # Strip "_actor." prefix from checkpoint keys and load into actor.
    actor_state = {
        k[len("_actor."):]: v
        for k, v in ckpt["model"].items()
        if k.startswith("_actor.")
    }
    actor.load_state_dict(actor_state)
    actor.eval()

    log.info(f"Actor in_keys:  {list(actor.in_keys)}")
    log.info(f"Actor out_keys: {list(actor.out_keys)}")

    # Verify actor obs keys are covered by our obs module
    uncovered = set(actor.in_keys) - set(obs_output_keys)
    if uncovered:
        raise ValueError(
            f"Actor requires obs keys not produced by ObservationExportModule: "
            f"{uncovered}.  Check observation_components."
        )

    # ------------------------------------------------------------------
    # 7. Build ActionExportModule
    # ------------------------------------------------------------------
    action_module = ActionExportModule(env_config.action_config, device="cpu")
    action_module.eval()

    # ------------------------------------------------------------------
    # 8. Compose UnifiedPipelineModule
    # ------------------------------------------------------------------
    unified = UnifiedPipelineModule(
        observation_module=obs_module,
        policy_module=actor,
        action_module=action_module,
        policy_in_keys=list(actor.in_keys),
        policy_action_key="mean_action",
    )
    unified.cpu().eval()

    # ------------------------------------------------------------------
    # 9. Collect sample inputs and verify forward pass
    # ------------------------------------------------------------------
    sample_inputs = [_resolve_attr_path(k, mock) for k in obs_input_keys]
    input_shapes = {k: list(v.shape) for k, v in zip(obs_input_keys, sample_inputs)}

    with torch.no_grad():
        actions, pd_targets, stiffness_t, damping_t = unified(*sample_inputs)

    log.info(f"Forward pass OK: actions={list(actions.shape)}, "
             f"pd_targets={list(pd_targets.shape)}")

    # ------------------------------------------------------------------
    # 10. Export to ONNX
    # ------------------------------------------------------------------
    def _sanitize(name: str) -> str:
        return name.replace(".", "_").replace("[", "_").replace("]", "_")

    onnx_input_names  = [_sanitize(k) for k in obs_input_keys]
    onnx_output_names = ["actions", "joint_pos_targets",
                         "stiffness_targets", "damping_targets"]

    onnx_path = output_path / "unified_pipeline.onnx"
    log.info(f"Exporting ONNX to {onnx_path} ...")
    torch.onnx.export(
        unified,
        tuple(sample_inputs),
        str(onnx_path),
        input_names=onnx_input_names,
        output_names=onnx_output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            **{name: {0: "batch_size"} for name in onnx_input_names},
            **{name: {0: "batch_size"} for name in onnx_output_names},
        },
        dynamo=False,
    )
    log.info(f"ONNX exported -> {onnx_path}")

    # ------------------------------------------------------------------
    # 11. Read back actual ONNX names (ONNX may rename inputs)
    # ------------------------------------------------------------------
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual_in_names  = [inp.name for inp in session.get_inputs()]
    actual_out_names = [out.name for out in session.get_outputs()]

    # Build onnx_name -> semantic_key mapping
    sanitized_to_key = {_sanitize(k): k for k in obs_input_keys}
    onnx_name_to_key: dict[str, str] = {}
    for onnx_name in actual_in_names:
        base = onnx_name
        for suffix in (".1", ".2", ".3", "_1", "_2", "_3"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        if base in sanitized_to_key:
            onnx_name_to_key[onnx_name] = sanitized_to_key[base]
        elif onnx_name in sanitized_to_key:
            onnx_name_to_key[onnx_name] = sanitized_to_key[onnx_name]
        else:
            log.warning(f"Cannot map ONNX input '{onnx_name}' to a semantic key")

    # ------------------------------------------------------------------
    # 12. Validate with onnxruntime
    # ------------------------------------------------------------------
    if validate:
        import numpy as np

        log.info("Validating with onnxruntime ...")
        key_to_tensor = {k: t for k, t in zip(obs_input_keys, sample_inputs)}
        ort_inputs = {
            name: key_to_tensor[onnx_name_to_key[name]].detach().numpy()
            for name in actual_in_names
            if name in onnx_name_to_key
        }
        ort_outputs = session.run(actual_out_names, ort_inputs)

        pytorch_outputs = [
            actions.detach().numpy(),
            pd_targets.detach().numpy(),
            stiffness_t.detach().numpy(),
            damping_t.detach().numpy(),
        ]
        for i, (name, pt_out) in enumerate(zip(onnx_output_names, pytorch_outputs)):
            diff = np.abs(ort_outputs[i] - pt_out).max()
            status = "OK" if diff < 1e-4 else "WARN"
            log.info(f"  {status}  {name}: max_diff = {diff:.2e}")
        log.info("Validation complete")

    # ------------------------------------------------------------------
    # 13. Build and write rich YAML metadata
    # ------------------------------------------------------------------
    stiffness_vals = [
        float(robot_config.control.control_info[j].stiffness) for j in joint_names
    ]
    damping_vals = [
        float(robot_config.control.control_info[j].damping) for j in joint_names
    ]

    # Effort limits (if available). NOTE: baked ControlInfo entries name the
    # field `effort_limit` (pose_lib.ControlInfo); the old `.effort`-only read
    # always raised AttributeError, which is why every previous yaml carried
    # `effort_limits: null`. Try both spellings per joint.
    effort_limits = None
    try:
        _ci = robot_config.control.control_info
        effort_limits = [
            float(
                getattr(_ci[j], "effort", None)
                if getattr(_ci[j], "effort", None) is not None
                else _ci[j].effort_limit
            )
            for j in joint_names
        ]
    except (AttributeError, KeyError, TypeError):
        pass

    mjcf_path = robot_config.asset.asset_file_name

    # Control type detection
    control_type = "BUILT_IN_PD"
    action_cfg = env_config.action_config
    if hasattr(action_cfg, "_target_"):
        control_type = action_cfg._target_.rsplit(".", 1)[-1]

    yaml_content = _build_yaml(
        onnx_in_names=actual_in_names,
        onnx_out_names=actual_out_names,
        onnx_name_to_key=onnx_name_to_key,
        input_shapes=input_shapes,
        obs_input_keys=obs_input_keys,
        actor_obs_configs=actor_obs_configs,
        joint_names=joint_names,
        body_names=body_names,
        stiffness=stiffness_vals,
        damping=damping_vals,
        effort_limits=effort_limits,
        pd_target_max_accel=pd_target_max_accel,
        anchor_body_name=anchor_body_name,
        anchor_body_index=anchor_body_index,
        root_body_index=root_body_index,
        num_bodies=num_bodies,
        num_dofs=num_dofs,
        mjcf_path=mjcf_path,
        control_dt=control_dt,
        physics_dt=physics_dt,
        decimation=decimation,
        future_step_indices=future_step_indices,
        checkpoint=str(checkpoint_path),
        control_type=control_type,
    )

    yaml_path = output_path / "unified_pipeline.yaml"
    import yaml

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=None, sort_keys=False)
    log.info(f"YAML metadata -> {yaml_path}")

    return onnx_path


# ---------------------------------------------------------------------------
# YAML builder
# ---------------------------------------------------------------------------


def _build_yaml(
    *,
    onnx_in_names,
    onnx_out_names,
    onnx_name_to_key,
    input_shapes,
    obs_input_keys,
    actor_obs_configs,
    joint_names,
    body_names,
    stiffness,
    damping,
    effort_limits,
    pd_target_max_accel,
    anchor_body_name,
    anchor_body_index,
    root_body_index,
    num_bodies,
    num_dofs,
    mjcf_path,
    control_dt,
    physics_dt,
    decimation,
    future_step_indices,
    checkpoint,
    control_type,
) -> dict:
    """Build the rich YAML metadata dict."""

    # Build per-input descriptors with element names
    policy_inputs = []
    for onnx_name in onnx_in_names:
        key = onnx_name_to_key.get(onnx_name, onnx_name)
        shape = input_shapes.get(key, "unknown")
        entry: dict = {
            "name": onnx_name,
            "key": key,
            "shape": shape,
        }

        # Determine kind and element names
        if "dof_pos" in key and "future" not in key:
            entry["kind"] = "joint_pos"
            entry["element_names"] = [joint_names]
        elif "dof_vel" in key and "future" not in key:
            entry["kind"] = "joint_vel"
            entry["element_names"] = [joint_names]
        elif "anchor_rot" in key and "future" not in key and "mimic" not in key:
            entry["kind"] = "anchor_rot"
            entry["element_names"] = [["x", "y", "z", "w"]]
        elif "root_local_ang_vel" in key:
            entry["kind"] = "local_root_ang_vel"
        elif "processed_actions" in key or "historical" in key:
            entry["kind"] = "last_actions"
            # The ONNX output key this feeds back from
            entry["output_key"] = "robot_action"
        elif "mimic" in key and "anchor_rot" in key:
            entry["kind"] = "reference_motion_body_rot"
            entry["future_steps"] = len(future_step_indices)
            entry["element_names"] = [[anchor_body_name], ["x", "y", "z", "w"]]
        elif "mimic" in key and "dof_pos" in key:
            entry["kind"] = "reference_motion_joint_pos"
            entry["future_steps"] = len(future_step_indices)
            entry["element_names"] = [joint_names]
        elif "mimic" in key and "dof_vel" in key:
            entry["kind"] = "reference_motion_joint_vel"
            entry["future_steps"] = len(future_step_indices)
            entry["element_names"] = [joint_names]

        policy_inputs.append(entry)

    # Build output descriptors
    policy_outputs = [
        {"name": "actions", "kind": "actions", "key": "actions",
         "shape": [1, num_dofs]},
        {"name": "joint_pos_targets", "kind": "joint_pos_targets",
         "key": "joint_pos_targets", "shape": [1, num_dofs],
         "joint_names": joint_names},
        {"name": "stiffness_targets", "kind": "stiffness_targets",
         "key": "stiffness_targets", "shape": [1, num_dofs],
         "joint_names": joint_names},
        {"name": "damping_targets", "kind": "damping_targets",
         "key": "damping_targets", "shape": [1, num_dofs],
         "joint_names": joint_names},
    ]

    # Passthrough keys: obs_input_keys that aren't ONNX inputs
    # (resolved to constants during tracing)
    passthrough_keys = [
        k for k in obs_input_keys
        if k not in onnx_name_to_key.values()
    ]

    content = {
        "type": "unified_pipeline",
        "dt": control_dt,
        "joint_names": joint_names,
        "body_names": body_names,
        "default_joint_stiffness": stiffness,
        "default_joint_damping": damping,
        "policy_inputs": policy_inputs,
        "policy_outputs": policy_outputs,
        "_runtime": {
            "onnx_in_names": onnx_in_names,
            "onnx_out_names": onnx_out_names,
            "onnx_name_to_in_key": onnx_name_to_key,
            "passthrough_keys": passthrough_keys,
            "obs_context_keys": obs_input_keys,
        },
        "metadata": {
            "checkpoint": checkpoint,
            "control_type": control_type,
        },
        # Full deployment contract fields (for robojudo integration)
        "robot": {
            "mjcf_path": mjcf_path,
            "num_bodies": num_bodies,
            "num_dofs": num_dofs,
            "anchor_body_name": anchor_body_name,
            "anchor_body_index": anchor_body_index,
            "root_body_name": body_names[root_body_index],
            "root_body_index": root_body_index,
            "body_names": body_names,
            "joint_names": joint_names,
        },
        "control": {
            "stiffness": stiffness,
            "damping": damping,
            "effort_limits": effort_limits,
            "pd_target_max_accel": pd_target_max_accel,
            "action_ema_alpha": 1.0,
        },
        "timing": {
            "control_dt": control_dt,
            "physics_dt": physics_dt,
            "decimation": decimation,
        },
        "motion": {
            "future_step_indices": future_step_indices,
            "future_dt_seconds": [round(s * control_dt, 6) for s in future_step_indices],
        },
    }
    return content


# ---------------------------------------------------------------------------
# Attribute path resolver
# ---------------------------------------------------------------------------


def _resolve_attr_path(path: str, obj):
    """Resolve a dotted attribute path on *obj*, e.g. ``"current.dof_pos"``."""
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Export a BeyondMimic tracker policy to ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint file (e.g. exps/my_exp/last.ckpt)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <checkpoint_dir>/compiled_models/)",
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Skip onnxruntime validation after export",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    out = args.output
    if out is None:
        out = str(Path(args.checkpoint).parent / "compiled_models")

    onnx_file = export_tracker(
        checkpoint=args.checkpoint,
        output_dir=out,
        validate=not args.no_validate,
    )
    log.info(f"\nDone!  Model exported to: {onnx_file}")
    log.info(f"YAML sidecar: {onnx_file.with_suffix('.yaml')}")
