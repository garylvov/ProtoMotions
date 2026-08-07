# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train reinforcement learning agents for physics-based character animation.

This is the main training script for ProtoMotions. It handles configuration loading,
distributed training setup, agent initialization, and checkpoint management.

Configuration System
--------------------

When you train a model, all configurations are automatically saved to the experiment
directory for exact reproducibility::

    results/my_experiment/
    ├── config.yaml              # CLI arguments
    ├── resolved_configs.pt      # Full config objects (pickled)
    ├── resolved_configs.yaml    # Human-readable configs
    ├── experiment_config.py     # Copy of experiment file
    └── last.ckpt               # Model checkpoint

The system saves three types of configuration files:

1. **resolved_configs.pt** (Primary): Full Python objects saved with pickle.
   Handles ALL types (Union, nested dataclasses, torch.Tensor, etc.) for guaranteed
   exact reproducibility.

2. **resolved_configs.yaml** (Human Reference): Best-effort YAML conversion for
   easy inspection and diffing.

3. **experiment_config.py** (Context): Copy of your experiment file showing original
   logic and intent.

Config Building Process
-----------------------

At first run without checkpoint:

1. configure_robot_and_simulator() - customize robot & sim
2. env_config() - build environment config
3. agent_config() - build agent config
4. Apply CLI overrides (--overrides) if provided
5. Save all to resolved_configs.pt

Important
---------
CLI overrides during training are PERMANENT! They are saved to resolved_configs.pt
and used in future resumes. For temporary overrides, use a new experiment name.

Create Config Only Mode
-----------------------
Use ``--create-config-only`` to generate config files without training. This is useful
for migrating old policy checkpoints when the config system API changes:

Generate new configs compatible with current code::

    python protomotions/train_agent.py \\
        --robot-name g1 --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp.py \\
        --experiment-name my_migrated_experiment \\
        --motion-file /path/to/motion.pt \\
        --num-envs 4096 --batch-size 16384 \\
        --create-config-only

Example
-------
>>> # Training with custom configuration
>>> # PYTHON_PATH protomotions/train_agent.py \\
>>> #     +exp=full_body_tracker/transformer_flat_terrain \\
>>> #     +robot=smpl \\
>>> #     +simulator=isaacgym \\
>>> #     motion_file=data/motions/amass_train.pt \\
>>> #     num_envs=2048 \\
>>> #     agent.config.batch_size=4096 \\
>>> #     +experiment_name=my_tracker
"""

import os
import time
import sys
import json

os.environ["WANDB_DISABLE_SENTRY"] = "true"  # Must be first environment variable
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_DISABLE_CODE"] = "true"

_WBC_STABILITY_ENV_DEFAULTS = {
    "PG_TIMEOUT_SEC": "3600",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "1200",
    "TORCH_NCCL_ENABLE_MONITORING": "1",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "1048576",
    # NCCL 2.26.2 DDP deadlock fix (2026-07-13). Root cause (native-stack forensics):
    # actor and critic are SEPARATE DDP modules with separate reducer buckets. The
    # critic bucket's FIRST all-reduce lazy-connects its NCCL channel ON THE HOT PATH
    # (the actor bucket was warmed by the actor step; the critic's was not). That lazy
    # connect spawns NCCL's nonblocking group-launch worker thread and joins it, hitting
    # a join-on-recycled-thread hang: ncclCommGetAsyncError -> ncclGroupJobComplete ->
    # ncclAsyncJobComplete -> std::thread::join on a dead pthread (state=ncclSuccess but
    # the collective is never enqueued). Reproducibly wedged one rank at the first critic
    # backward of epoch 0/1 on the 36864-env warm-start; NOT memory pressure, NOT a
    # graph/size mismatch, NOT GPU hardware (ECC/Xid clean; actor all-reduce on the same
    # comm already succeeded). RUNTIME_CONNECT=0 eager-connects ALL channels at
    # ncclCommInitRank (init time, off the hot path) -> removes the trigger.
    # USE_COMM_NONBLOCKING=0 forces BLOCKING comms so the nonblocking group-launch worker
    # thread never exists -> removes the racy join. Verified: 36864/85GB trains cleanly
    # past epoch 0/1 (was: hang there twice). A definitive underlying fix is NCCL >=2.27
    # (fixes "a group launch of multiple communicators"); these env defaults are the
    # zero-rebuild, spot-durable mitigation.
    "NCCL_RUNTIME_CONNECT": "0",
    "TORCH_NCCL_USE_COMM_NONBLOCKING": "0",
}


def _set_wbc_stability_env_defaults() -> None:
    """Seed process-group/NCCL stability defaults before Fabric initializes."""

    for name, value in _WBC_STABILITY_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)


def _raise_nproc_soft_limit() -> None:
    """Raise RLIMIT_NPROC soft limit to min(desired, hard) at startup.

    fix: RLIMIT_NPROC starvation wedges multi-rank Isaac+NCCL training.
    Ranks that inherit the bare SLURM step default (soft=4096) exhaust the
    per-UID thread budget during startup/epoch0 burst thread creation --
    pthread_create returns EAGAIN inside NCCL, the failing rank SIGABRTs
    holding process-group state, and every peer wedges in its next
    collective (the ncclSystemError / "pthread_join failed" /
    futex_wait_queue family; see wbc_push/briefs/rank_stall_rca.*.md
    round-2). Launcher-side `ulimit -u` is fragile: it silently no-ops when
    the request exceeds the hard cap and does not reliably survive
    setsid/spawn boundaries (both observed live). Setting it here, in the
    training process itself before simulator/Fabric setup, is immune to
    launch plumbing. Never request above the hard cap -- clamp to it.
    Override the target via NPROC_SOFT_LIMIT; opt out with
    NPROC_SOFT_LIMIT=0.
    """

    import resource

    try:
        desired = int(os.environ.get("NPROC_SOFT_LIMIT", "16384"))
    except ValueError:
        desired = 16384
    if desired <= 0:
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        target = desired if hard == resource.RLIM_INFINITY else min(desired, hard)
        if soft != resource.RLIM_INFINITY and target > soft:
            resource.setrlimit(resource.RLIMIT_NPROC, (target, hard))
            print(
                f"[train_agent] raised RLIMIT_NPROC soft {soft} -> {target} (hard={hard})"
            )
    except (ValueError, OSError) as exc:
        print(f"[train_agent] WARN could not raise RLIMIT_NPROC: {exc}")

"""
## Quick Start

When you train a model, all configurations are automatically saved to the experiment directory for exact reproducibility:

```bash
# Training (automatic config saving)
python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name my_experiment \
    --motion-file /path/to/motion.pt \
    --num-envs 4096 \
    --batch-size 16384

# Results in:
results/my_experiment/
├── config.yaml              # CLI arguments
├── resolved_configs.pt      # Full config objects (pickled)
├── resolved_configs.yaml    # Human-readable configs (best-effort)
├── experiment_config.py     # Copy of mlp.py
└── last.ckpt               # Model checkpoint
```
## Why This Approach?

### Problem
Config dataclasses can have complex types (Union, nested dataclasses, torch.Tensors) that JSON/YAML can't handle, plus experiments often inherit from base configs that may change over time.

### Solution
**Three files for different purposes:**

1. **`resolved_configs.pt`** (Primary)
   - Full Python objects saved with pickle
   - Handles ALL types (Union, nested, torch.Tensor, etc.)
   - Guaranteed exact reproducibility
   - Not human-readable

2. **`resolved_configs.yaml`** (Human Reference)
   - Best-effort YAML conversion
   - Easy to inspect and diff
   - May fail for complex types (non-critical)
   - Human-readable

3. **`experiment_config.py`** (Context)
   - Copy of your experiment file
   - Shows original logic and intent
   - Useful for understanding decisions

Config System, at 1st run without ckpt

Config Building (from experiment file):
1. configure_robot_and_simulator() - customize robot & sim
2. env_config() - build environment config
3. agent_config() - build agent config
4. Apply CLI overrides (--overrides) if provided
5. Save all to resolved_configs.pt

IMPORTANT: CLI overrides during training are PERMANENT!
They are saved to resolved_configs.pt and used in future resumes.
For temporary overrides, use a new experiment name.
"""


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Train reinforcement learning agent with configurable parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--robot-name",
        type=str,
        required=True,
        help="Name of the robot (e.g., 'g1', 'smpl')",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton', 'genesis')",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        required=True,
        help="Number of parallel environments to run",
    )
    parser.add_argument(
        "--batch-size", type=int, required=True, help="Batch size for training"
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        required=True,
        help="Path to motion file for training",
    )
    parser.add_argument(
        "--experiment-path",
        type=str,
        required=True,
        help="File path to experiment configuration (e.g., 'examples/train/mimic/mimic_mlp.py')",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Name of the experiment for logging and checkpointing",
    )

    # Optional arguments
    parser.add_argument(
        "--scenes-file", type=str, default=None, help="Path to scenes file (optional)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from",
    )
    parser.add_argument(
        "--warmstart-training-state",
        action="store_true",
        default=False,
        help=(
            "On a WARM start (--checkpoint into a fresh logdir), also restore the "
            "optimizer/Adam-moment + advantage-normalization state from the checkpoint "
            "(load_training_state=True) instead of starting with fresh Adam. Requires the "
            "checkpoint to contain actor_optimizer/critic_optimizer. Epoch stays 0 and the "
            "evaluator is not restored unless present in the checkpoint. No effect on resume/fresh."
        ),
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--use-slurm",
        action="store_true",
        default=False,
        help="Enable SLURM autoresume functionality",
    )
    parser.add_argument(
        "--ngpu", type=int, default=1, help="Number of GPUs to use for training"
    )
    parser.add_argument(
        "--nodes", type=int, default=1, help="Number of nodes for distributed training"
    )
    parser.add_argument(
        "--headless",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Run simulation in headless mode",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--torch-deterministic",
        action="store_true",
        default=False,
        help="Enable deterministic PyTorch operations",
    )
    parser.add_argument(
        "--training-max-steps",
        type=int,
        default=10000000000000,
        help="Maximum number of training steps. Default to 'loads of steps'.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in format key=value (e.g., env.max_episode_length=1000 simulator.num_envs=4096)",
    )
    parser.add_argument(
        "--create-config-only",
        action="store_true",
        default=False,
        help="Only create and save config files without training. "
        "Useful for migrating old policy checkpoints when config system API changes - "
        "generate new configs that are compatible with current code, then load old weights.",
    )

    return parser


# Parse arguments first (argparse is safe, doesn't import torch)
import argparse  # noqa: E402
from protomotions.utils.cli_utils import parse_bool  # noqa: E402

parser = create_parser()
args, unknown_args = parser.parse_known_args()
_set_wbc_stability_env_defaults()
_raise_nproc_soft_limit()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch
# This also returns AppLauncher if using isaaclab, None otherwise
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else including torch
from pathlib import Path  # noqa: E402
import logging  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
import importlib.util  # noqa: E402
import shutil  # noqa: E402
import wandb  # noqa: E402
from lightning.pytorch.loggers import WandbLogger  # noqa: E402
import torch  # noqa: E402
from utils.torch_utils import seeding  # noqa: E402
from dataclasses import asdict  # noqa: E402
from protomotions.utils.config_utils import clean_dict_for_storage, make_json_serializable  # noqa: E402


log = logging.getLogger(__name__)


def detect_checkpoint_mode(args, save_dir):
    """
    Detect checkpoint mode: resume, warm start, or fresh.

    Returns:
        tuple: (mode, checkpoint_path, wandb_id)
            mode: "resume", "warm_start", or "fresh"
            checkpoint_path: Path to checkpoint or None
            wandb_id: Wandb ID for resume or None
    """
    pre_existing_checkpoint = save_dir / "last.ckpt"
    checkpoint_config_path = save_dir / "config.yaml"

    # Priority 1: Resume - continuing same run
    if pre_existing_checkpoint.exists():
        log.info(f"RESUME: Found checkpoint in save_dir: {pre_existing_checkpoint}")

        # Load wandb_id
        wandb_id = None
        if checkpoint_config_path.exists():
            log.info(f"Loading saved args from {checkpoint_config_path}")
            with open(checkpoint_config_path, "r") as file:
                checkpoint_config = json.load(file)

            # Update args with checkpoint config.
            # Skip transient CLI-only flags that must NOT be restored from a
            # saved run: `create_config_only` (a saved config minted via
            # --create-config-only bakes this True; restoring it re-triggers the
            # create-config-only branch on every resume -> save_configs ->
            # json.dump(args) crashes on the resume checkpoint PosixPath) and
            # `checkpoint` (resume sets its own checkpoint path below).
            _SKIP_RESTORE = {"wandb_id", "create_config_only", "checkpoint"}
            for key, value in checkpoint_config.items():
                if key not in _SKIP_RESTORE:
                    setattr(args, key, value)
            wandb_id = checkpoint_config.get("wandb_id", None)
        else:
            raise FileNotFoundError(
                f"Config file not found at {checkpoint_config_path}"
            )

        return "resume", pre_existing_checkpoint, wandb_id

    # Priority 2: Warm Start - new run with pretrained weights
    elif args.checkpoint is not None:
        log.info(f"WARM START: Using checkpoint for initialization: {args.checkpoint}")
        return "warm_start", Path(args.checkpoint), None

    # No checkpoint - training from scratch
    else:
        log.info("FRESH START: Training from scratch")
        return "fresh", None, None


def load_experiment_module(experiment_path):
    """
    Load the experiment module from a given path.

    Args:
        experiment_path: Path to the experiment Python file

    Returns:
        Loaded experiment module
    """
    experiment_path = Path(experiment_path)

    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {experiment_path}")

    log.info(f"Loading experiment module from: {experiment_path}")

    # Ensure the repo root is on sys.path so that experiment configs can import
    # from sibling packages (e.g. `from examples.experiments.mimic... import ...`).
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    spec = importlib.util.spec_from_file_location("experiment_module", experiment_path)
    experiment_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_module)

    return experiment_module


def prepare_inference_configs_for_save(*configs):
    """Let configs make inference bundles self-contained before saving.

    Training configs should stay faithful to the experiment setup. Inference
    configs may need to embed lightweight construction metadata for frozen
    modules whose weights are saved inside the owning checkpoint.
    """
    for config in configs:
        hook = getattr(config, "prepare_inference_config_for_save", None)
        if callable(hook):
            hook()


def save_configs(
    save_dir,
    args,
    robot_config,
    simulator_config,
    terrain_config,
    scene_lib_config,
    motion_lib_config,
    env_config,
    agent_config,
    fabric_config,
    experiment_source_path,
    file_name="resolved_configs",
):
    """
    Save all configuration files (first run only).

    Saves:
    - config.yaml (CLI args + wandb_id)
    - resolved_configs.pt (pickled config objects)
    - resolved_configs.yaml (human-readable, best-effort)
    - experiment_config.py (copy of experiment file)
    """
    checkpoint_config_path = save_dir / "config.yaml"

    # Convert args to dict and add wandb_id
    checkpoint_config = vars(args).copy()
    checkpoint_config["wandb_id"] = None

    # Try to get wandb_id from loggers
    if args.use_wandb:
        try:
            wandb_id = wandb.run.id
            log.info(f"wandb_id found: {wandb_id}")
            checkpoint_config["wandb_id"] = wandb_id
        except Exception:
            log.warning("Could not get wandb_id")

    # Save CLI args + wandb_id
    log.info(f"Saving config file to {save_dir}")
    checkpoint_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_config_path, "w") as file:
        json.dump(checkpoint_config, file, indent=2)

    # Save pickled configs (guaranteed reproducibility)
    resolved_configs_path = (save_dir / file_name).with_suffix(".pt")
    resolved_configs = {
        "robot": robot_config,
        "simulator": simulator_config,
        "terrain": terrain_config,
        "scene_lib": scene_lib_config,
        "motion_lib": motion_lib_config,
        "env": env_config,
        "agent": agent_config,
    }
    log.info(f"Saving resolved configs (pickled) to {resolved_configs_path}")
    torch.save(resolved_configs, resolved_configs_path)

    # Save YAML configs (human-readable, best-effort)
    try:
        resolved_configs_yaml_path = (save_dir / file_name).with_suffix(".yaml")
        resolved_configs_dict = {
            "robot": clean_dict_for_storage(asdict(robot_config)),
            "simulator": clean_dict_for_storage(asdict(simulator_config)),
            "terrain": clean_dict_for_storage(asdict(terrain_config)),
            "scene_lib": clean_dict_for_storage(asdict(scene_lib_config)),
            "motion_lib": clean_dict_for_storage(asdict(motion_lib_config)),
            "env": clean_dict_for_storage(asdict(env_config)),
            "agent": clean_dict_for_storage(asdict(agent_config)),
        }
        import yaml

        log.info(f"Saving resolved configs (YAML) to {resolved_configs_yaml_path}")
        with open(resolved_configs_yaml_path, "w") as file:
            yaml.dump(
                resolved_configs_dict, file, default_flow_style=False, sort_keys=False
            )
    except Exception as e:
        log.warning(f"Could not save YAML configs (non-critical): {e}")

    # Copy experiment Python file just for human reference, not used by code.
    experiment_copy_path = save_dir / "experiment_config.py"
    log.info(f"Copying experiment file to {experiment_copy_path}")
    shutil.copy(experiment_source_path, experiment_copy_path)


def try_log_hyperparams_to_wandb(
    fabric,
    robot_config,
    simulator_config,
    terrain_config,
    scene_lib_config,
    motion_lib_config,
    env_config,
    agent_config,
    fabric_config,
):
    """Try to log hyperparameters to wandb (non-critical)."""
    for logger in fabric.loggers:
        if isinstance(logger, WandbLogger):
            try:
                hyper_params = {
                    "robot": clean_dict_for_storage(asdict(robot_config)),
                    "simulator": clean_dict_for_storage(asdict(simulator_config)),
                    "terrain": clean_dict_for_storage(asdict(terrain_config)),
                    "scene_lib": clean_dict_for_storage(asdict(scene_lib_config)),
                    "motion_lib": clean_dict_for_storage(asdict(motion_lib_config)),
                    "env": clean_dict_for_storage(asdict(env_config)),
                    "agent": clean_dict_for_storage(asdict(agent_config)),
                    "fabric": clean_dict_for_storage(fabric_config.as_loggable_dict()),
                }

                log.info("Preparing configs for wandb logging...")
                serializable_params = make_json_serializable(hyper_params)
                logger.log_hyperparams(serializable_params)
                log.info("Successfully logged hyperparams to wandb")
            except Exception as e:
                log.warning(f"Could not log hyperparams to wandb (non-critical): {e}")


def main():
    global parser, args
    torch.set_float32_matmul_precision("high")

    # ===================================================================
    # 1. Setup: Detect Checkpoint Mode
    # ===================================================================
    save_dir = Path("results") / args.experiment_name
    resolved_configs_path = save_dir / "resolved_configs.pt"
    original_experiment_path = Path(args.experiment_path)

    # --create-config-only: Force fresh mode to just generate configs
    if args.create_config_only:
        log.info("CREATE CONFIG ONLY: Generating configs without training")
        mode, checkpoint_path, wandb_id = "fresh", None, None
    else:
        mode, checkpoint_path, wandb_id = detect_checkpoint_mode(args, save_dir)

    # ===================================================================
    # 2. Load Configs Based on Mode
    # ===================================================================

    if mode == "resume":
        # ===============================================================
        # RESUME: Continuing same run - load from pickle only
        # Does NOT load experiment module or rebuild configs
        # ===============================================================
        if not resolved_configs_path.exists():
            raise FileNotFoundError(
                f"Resume requires resolved_configs.pt but not found at {resolved_configs_path}\n"
                f"This may be an old checkpoint. Use --checkpoint flag for warm start instead."
            )

        log.info(f"Loading configs from {resolved_configs_path}")
        resolved_configs = torch.load(
            resolved_configs_path, map_location="cpu", weights_only=False
        )

        robot_config = resolved_configs["robot"]
        simulator_config = resolved_configs["simulator"]
        terrain_config = resolved_configs["terrain"]
        scene_lib_config = resolved_configs["scene_lib"]
        motion_lib_config = resolved_configs["motion_lib"]
        env_config = resolved_configs["env"]
        agent_config = resolved_configs["agent"]

        # PM_MIRROR_PROB (env-gated, resume-safe): online sagittal-mirror ref
        # augmentation. Unlike the reward gates below, this lives on the motion
        # manager, which IS rebuilt fresh from motion_lib_config on resume -- and
        # MotionManager.__init__ reads PM_MIRROR_PROB LIVE from os.environ,
        # preferring it over the frozen config field. So enabling mirror at a
        # resume boundary just works: export PM_MIRROR_PROB=0.5 and resume. We
        # still stamp the frozen config here so resolved_configs.pt reflects the
        # active value (keeps the human-readable snapshot honest). No-op unless
        # the env var is explicitly present.
        _mirror_env = os.environ.get("PM_MIRROR_PROB")
        if _mirror_env is not None:
            _mm_cfg = getattr(motion_lib_config, "motion_manager", None) \
                or getattr(env_config, "motion_manager", None)
            if _mm_cfg is not None and hasattr(_mm_cfg, "mirror_prob"):
                log.info(
                    f"RESUME override motion_manager.mirror_prob: "
                    f"{getattr(_mm_cfg, 'mirror_prob', None)} -> {float(_mirror_env)} "
                    f"(from PM_MIRROR_PROB; live-read also applies)"
                )
                _mm_cfg.mirror_prob = float(_mirror_env)

        # PM_WRIST_*_WEIGHT / _SIGMA (env-gated, resume-safe): resume freezes
        # reward config from the pickle, so the experiment file's env gates
        # never run here. Re-apply them on the loaded components so reward
        # ladders can ride a resume without a warm start.
        _rc = getattr(env_config, "reward_components", None) or {}
        for _comp, _wvar, _svar in (
            ("wrist_relative_body_pos", "PM_WRIST_POS_WEIGHT", "PM_WRIST_POS_SIGMA"),
            ("wrist_relative_body_ori", "PM_WRIST_ORI_WEIGHT", "PM_WRIST_ORI_SIGMA"),
        ):
            if _comp not in _rc:
                continue
            _sp = _rc[_comp].static_params
            for _var, _key in ((_wvar, "weight"), (_svar, "sigma")):
                _val = os.environ.get(_var)
                if _val:
                    log.info(
                        f"RESUME override {_comp}.{_key}: "
                        f"{_sp.get(_key)} -> {float(_val)} (from {_var})"
                    )
                    _sp[_key] = float(_val)

        # PM v2 REWARD-GATE FAMILY (env-gated, resume-safe): resume freezes the
        # reward config from the pickle so teacher.py's env gates never run on a
        # resume -- the REF-OR-SELF lift gate (and the rest of the v2 stepping /
        # anti-shuffle stack) would silently stay at their frozen values. Re-apply
        # the family onto the loaded components so gate ladders ride a resume
        # without a warm start (imprint PR #119 step-in-place: resume re-apply gap).
        # STORAGE SPLIT: weight/sigma live in component.static_params; the motion-
        # gate / geometry thresholds live as attributes on the stateful
        # component.compute_func object (e.g. FeetApexHeightReward.min_ref_speed).
        # GUARD: each override fires ONLY when its env var is EXPLICITLY PRESENT in
        # os.environ (not default-valued), so an ordinary resume (no launcher env)
        # is byte-identical to the frozen config.
        _SP, _CF = "sp", "cf"
        for _comp, _var, _loc, _key in (
            ("feet_apex_height", "PM_STEP_LIFT_WEIGHT", _SP, "weight"),
            ("feet_apex_height", "PM_STEP_LIFT_MIN_REF_SPEED", _CF, "min_ref_speed"),
            ("feet_apex_height", "PM_STEP_LIFT_MIN_SELF_SPEED", _CF, "min_self_speed"),
            ("stepping", "PM_STEPPING_WEIGHT", _SP, "weight"),
            ("stepping", "PM_STEPPING_MIN_REF_SPEED", _CF, "min_ref_speed"),
            ("micro_step_tax", "PM_ANTISHUFFLE_WEIGHT", _SP, "weight"),
            ("micro_step_tax", "PM_ANTISHUFFLE_MAX_STEP", _CF, "max_step_length"),
            ("micro_step_tax", "PM_ANTISHUFFLE_MAX_APEX", _CF, "max_apex_height"),
            ("micro_step_tax", "PM_MICRO_TAX_MIN_SWING_STEPS", _CF, "min_swing_steps"),
            ("step_budget", "PM_STEP_BUDGET_MIN_SWING_STEPS", _CF, "min_swing_steps"),
            ("step_budget", "PM_STEP_BUDGET_MAX_CREDITS", _CF, "max_credits"),
            ("step_budget", "PM_STEP_BUDGET_STREAK_CAP", _CF, "streak_cap"),
            ("step_budget", "PM_STEP_BUDGET_STREAK_DECAY_STEPS", _CF, "streak_decay_steps"),
            ("foot_slip", "PM_FOOT_SLIP_WEIGHT", _SP, "weight"),
            ("foot_slip", "PM_FOOT_SLIP_ANG_SCALE", _SP, "ang_vel_scale"),
            ("foot_slip", "PM_FOOT_SLIP_ZVEL_SCALE", _SP, "z_vel_scale"),
            ("feet_apex_height", "PM_STEP_LIFT_MIN_SWING_SEC", _CF, "min_swing_sec"),
            ("feet_apex_height", "PM_STEP_LIFT_PLACEMENT_SIGMA", _CF, "placement_sigma"),
            ("feet_apex_height", "PM_STEP_APEX_TARGET", _CF, "apex_target_height"),
            ("feet_apex_height", "PM_STEP_LIFT_RECOVERY_PAY_SCALE", _CF, "recovery_pay_scale"),
            ("foot_speed", "PM_FOOT_SPEED_WEIGHT", _SP, "weight"),
            ("foot_speed", "PM_FOOT_SPEED_MAX", _SP, "max_foot_speed"),
            ("foot_speed", "PM_FOOT_SPEED_REF_SCALE", _SP, "ref_speed_scale"),
            ("fall_penalty", "PM_FALL_PENALTY_WEIGHT", _SP, "weight"),
            ("drift_penalty", "PM_DRIFT_PENALTY_WEIGHT", _SP, "weight"),
            ("drift_penalty", "PM_DRIFT_PENALTY_THRESHOLD", _SP, "drift_threshold"),
            ("step_budget", "PM_STEP_BUDGET_WEIGHT", _SP, "weight"),
            ("dof_pos_track", "PM_DOF_POS_TRACK_WEIGHT", _SP, "weight"),
            ("dof_pos_track", "PM_DOF_POS_TRACK_SIGMA", _SP, "sigma"),
            ("global_anchor_pos", "PM_GLOBAL_POS_WEIGHT", _SP, "weight"),
            ("global_anchor_pos", "PM_GLOBAL_POS_SIGMA", _SP, "sigma"),
            ("heading_local_anchor_drift", "PM_HEADING_DRIFT_WEIGHT", _SP, "weight"),
            # DUAL-SIGMA companions (2026-08-04, static-hold precision): each
            # position-tracking Gaussian gains an OPTIONAL narrow companion
            # ``+ fine_weight * exp(-e^2 / fine_sigma^2)`` on top of the
            # UNCHANGED coarse kernel, so precision near zero error is bought
            # without shrinking capture range (see rewards/tracking.py
            # ``_dual_sigma_exp``). Frozen configs were pickled WITHOUT these
            # keys; the kernel defaults (fine_weight=0.0) keep them
            # byte-identical, so the companion activates ONLY via these rows.
            # fine_weight is RELATIVE to the component's own weight.
            # WORLD-FRAME HAND POSITION (2026-08-04 frame audit): the only
            # term that scores where the hand actually is in the WORLD, so the
            # arm is paid to cancel base sway. Every other body-position term
            # is anchor-relative and cancels pelvis translation/yaw exactly.
            ("global_wrist_pos", "PM_GLOBAL_WRIST_POS_WEIGHT", _SP, "weight"),
            ("global_wrist_pos", "PM_GLOBAL_WRIST_POS_SIGMA", _SP, "sigma"),
            ("global_wrist_pos", "PM_GLOBAL_WRIST_POS_FINE_WEIGHT", _SP, "fine_weight"),
            ("global_wrist_pos", "PM_GLOBAL_WRIST_POS_FINE_SIGMA", _SP, "fine_sigma"),
            ("relative_body_pos", "PM_REL_POS_FINE_WEIGHT", _SP, "fine_weight"),
            ("relative_body_pos", "PM_REL_POS_FINE_SIGMA", _SP, "fine_sigma"),
            # wrist_relative_body_pos is the term that actually governs the
            # HAND (anchor-relative, heading-local, body_indices = wrists).
            ("wrist_relative_body_pos", "PM_WRIST_POS_FINE_WEIGHT", _SP, "fine_weight"),
            ("wrist_relative_body_pos", "PM_WRIST_POS_FINE_SIGMA", _SP, "fine_sigma"),
            ("global_anchor_pos", "PM_GLOBAL_POS_FINE_WEIGHT", _SP, "fine_weight"),
            ("global_anchor_pos", "PM_GLOBAL_POS_FINE_SIGMA", _SP, "fine_sigma"),
            ("dof_pos_track", "PM_DOF_POS_TRACK_FINE_WEIGHT", _SP, "fine_weight"),
            ("dof_pos_track", "PM_DOF_POS_TRACK_FINE_SIGMA", _SP, "fine_sigma"),
            (
                "heading_local_anchor_drift",
                "PM_HEADING_DRIFT_FINE_WEIGHT",
                _SP,
                "fine_weight",
            ),
            (
                "heading_local_anchor_drift",
                "PM_HEADING_DRIFT_FINE_SIGMA",
                _SP,
                "fine_sigma",
            ),
            # Static-hold body VELOCITY penalty (2026-08-04): gate + weight for
            # the reference-still-gated hand/body velocity term. The component
            # itself is registered by the fresh-build gate
            # (PM_STATIC_HOLD_VEL_WEIGHT in teacher.py); these rows let a RESUME
            # retune an already-registered one.
            ("static_hold_vel", "PM_STATIC_HOLD_VEL_WEIGHT", _SP, "weight"),
            ("static_hold_vel", "PM_STATIC_HOLD_VEL_REF_GATE", _SP, "ref_speed_gate"),
            # HELD-REFERENCE JOINT-QUIET (FIX A, 2026-08-04 pause forensics):
            # exp(-vel_scale * mean_j dof_vel^2) gated on the HOLD-FIX
            # reference_still_mask. Prices the ~1.19 Hz whole-body postural
            # limit cycle that every existing term is blind to (tracking
            # Gaussians are flat below ~3 cm; the one-step action taxes see
            # deltas ~7x smaller than the amplitude at 1.2 Hz). These rows
            # RETUNE an already-registered component; a resume that is
            # ACTIVATING it for the first time is served by the injection pass
            # below (RESUME_INJECTABLE_COMPONENTS), which reads the same knobs.
            ("hold_joint_quiet", "PM_HOLD_JOINT_QUIET_WEIGHT", _SP, "weight"),
            ("hold_joint_quiet", "PM_HOLD_JOINT_QUIET_VEL_SCALE", _SP, "vel_scale"),
            # DOF-limit soft-margin proximity (2026-07-29, knees-at-full-
            # extension fix): frozen configs were pickled WITHOUT these keys;
            # the kernel defaults (soft_margin_frac=0.0) keep them byte-
            # identical, so the term ONLY activates via this re-apply row.
            ("limits_dof_pos", "PM_DOF_LIMIT_MARGIN", _SP, "soft_margin_frac"),
            ("limits_dof_pos", "PM_DOF_LIMIT_PROX_SCALE", _SP, "proximity_scale"),
        ):
            _val = os.environ.get(_var)
            if _val is None:
                continue
            if _comp not in _rc:
                # M4 (2026-07-27): an explicitly-set gate var whose component
                # is absent from the frozen config used to be dropped
                # SILENTLY -- loudly flag the no-op instead.
                # 2026-08-04: distinguish a genuine no-op from a component the
                # LATER injection pass is about to create. Claiming "NO effect"
                # for an injectable component's knobs on its first activating
                # resume would be a lie in the resume log, which is the only
                # artifact anyone reads to confirm what the reward became.
                from protomotions.envs.component_factories import (
                    RESUME_INJECTABLE_COMPONENTS as _INJECTABLE,
                )

                if _comp in _INJECTABLE:
                    log.warning(
                        f"RESUME override DEFERRED: {_var}={_val} is set and "
                        f"reward component '{_comp}' is absent from the frozen "
                        f"config -- the RESUME INJECT pass below will create it "
                        f"from the same env knobs (watch for 'RESUME INJECT "
                        f"component {_comp}')"
                    )
                else:
                    log.warning(
                        f"RESUME override SKIPPED: {_var}={_val} is set but reward "
                        f"component '{_comp}' is not in the frozen config "
                        f"(env var has NO effect on this resume)"
                    )
                continue
            # min_swing_steps / streak_cap / streak_decay_steps are INT
            # attributes on the compute_func (control-step / event counts,
            # not continuous weights/sigmas) -- the family is otherwise
            # all-float, so cast explicitly here rather than leaving a float
            # on an int-typed attr (>= comparisons still "work" but the
            # stored/logged value would misleadingly show e.g. 2.0).
            _fval = (
                int(float(_val))
                if _key in ("min_swing_steps", "streak_cap", "streak_decay_steps")
                else float(_val)
            )
            if _loc == _SP:
                _old = _rc[_comp].static_params.get(_key)
                _rc[_comp].static_params[_key] = _fval
            else:
                _cf = _rc[_comp].compute_func
                _old = getattr(_cf, _key, None)
                setattr(_cf, _key, _fval)
            # WARNING level: the resume log captures WARNING+ only, so INFO
            # override lines were invisible on the gpu3202 gate resume.
            log.warning(
                f"RESUME override {_comp}.{_key} = {_fval} (was {_old}, from {_var})"
            )
        from protomotions.envs.component_factories import (
            validate_dual_sigma_components as _validate_dual_sigma,
        )

        _validate_dual_sigma(_rc, log_fn=log.warning)
        # v5.2 placement gate needs a dynamic_var the pre-v5.2 frozen configs
        # were pickled WITHOUT: ref_rigid_body_pos. Inject it on resume when
        # the placement sigma is explicitly enabled, else the kernel silently
        # never receives ref positions and the gate no-ops (the exact class of
        # trap the re-apply loop exists to prevent).
        if (
            os.environ.get("PM_STEP_LIFT_PLACEMENT_SIGMA")
            and "feet_apex_height" in _rc
        ):
            from protomotions.envs.context_views import EnvContext as _ECtx

            _dv = _rc["feet_apex_height"].dynamic_vars
            if "ref_rigid_body_pos" not in _dv:
                _dv["ref_rigid_body_pos"] = _ECtx.mimic.ref_state.rigid_body_pos
                log.warning(
                    "RESUME override feet_apex_height.dynamic_vars += "
                    "ref_rigid_body_pos (v5.2 placement gate wiring)"
                )
            # M1 placement yaw alignment (2026-07-27): the kernel now rotates
            # each side's root-relative foot XY into its own root heading
            # frame; pre-M1 frozen configs lack the rot tensors (kernel falls
            # back to the world-frame comparison without them).
            for _rk, _rv in (
                ("rigid_body_rot", _ECtx.current.rigid_body_rot),
                ("ref_rigid_body_rot", _ECtx.mimic.ref_state.rigid_body_rot),
            ):
                if _rk not in _dv:
                    _dv[_rk] = _rv
                    log.warning(
                        f"RESUME override feet_apex_height.dynamic_vars += "
                        f"{_rk} (M1 placement yaw-alignment wiring)"
                    )
        # Heel-pop pricing (2026-07-28) needs a dynamic_var pre-fix frozen
        # configs were pickled WITHOUT: rigid_body_ang_vel. Inject it on
        # resume when the ang-scale knob is explicitly enabled, else the
        # kernel never receives foot angular velocities and the angular term
        # silently no-ops (same trap class as the placement-gate wiring).
        if (
            os.environ.get("PM_FOOT_SLIP_ANG_SCALE")
            and "foot_slip" in _rc
        ):
            from protomotions.envs.context_views import EnvContext as _ECtx2

            _dv = _rc["foot_slip"].dynamic_vars
            if "rigid_body_ang_vel" not in _dv:
                _dv["rigid_body_ang_vel"] = _ECtx2.current.rigid_body_ang_vel
                log.warning(
                    "RESUME override foot_slip.dynamic_vars += "
                    "rigid_body_ang_vel (heel-pop stance-stillness wiring)"
                )
        _alt = os.environ.get("PM_STEP_LIFT_ALTERNATE")
        if _alt is not None and "feet_apex_height" in _rc:
            _cf = _rc["feet_apex_height"].compute_func
            _newalt = _alt not in ("0", "")
            _oldalt = getattr(_cf, "require_alternation", None)
            setattr(_cf, "require_alternation", _newalt)
            if getattr(_cf, "_last_paid_foot", "MISSING") == "MISSING":
                setattr(_cf, "_last_paid_foot", None)
            log.warning(
                f"RESUME override feet_apex_height.require_alternation = "
                f"{_newalt} (was {_oldalt}, from PM_STEP_LIFT_ALTERNATE)"
            )
        # v5.5 same-foot-repeat forced overdraft (bool cast mirrors the
        # PM_STEP_LIFT_ALTERNATE pattern above). Also seed the lazily-created
        # state attrs to None on unpickled pre-v5.5 instances so __call__'s
        # getattr path starts clean.
        _balt = os.environ.get("PM_STEP_BUDGET_ALTERNATE")
        if _balt is not None and "step_budget" in _rc:
            _cf = _rc["step_budget"].compute_func
            _newbalt = _balt not in ("0", "")
            _oldbalt = getattr(_cf, "require_alternation_budget", None)
            setattr(_cf, "require_alternation_budget", _newbalt)
            for _attr in ("_last_counted_foot", "_ref_last_td_foot", "_ref_repeat"):
                if getattr(_cf, _attr, "MISSING") == "MISSING":
                    setattr(_cf, _attr, None)
            log.warning(
                f"RESUME override step_budget.require_alternation_budget = "
                f"{_newbalt} (was {_oldbalt}, from PM_STEP_BUDGET_ALTERNATE)"
            )

        # PM_ARM_{KP,KD,EFFORT}[_SHOULDER|_ELBOW|_WRIST] (env-gated, resume-safe):
        # robot_config is likewise frozen from the pickle, so the module-level
        # PM_ARM_KP gate in robot_configs/h1_2.py never fires on resume.
        # Per-group overrides beat the uniform PM_ARM_KP/KD when both are set.
        _oci = getattr(getattr(robot_config, "control", None), "override_control_info", None) or {}
        _arm_groups = {
            "SHOULDER": [".*_shoulder_(pitch|roll)_joint", ".*_shoulder_yaw_joint"],
            "ELBOW": [".*_elbow_joint"],
            "WRIST": [".*_wrist_(roll|pitch|yaw)_joint"],
        }
        _arm_touched = False
        for _grp, _patterns in _arm_groups.items():
            for _field, _base in (("stiffness", "PM_ARM_KP"), ("damping", "PM_ARM_KD"), ("effort_limit", "PM_ARM_EFFORT")):
                _val = os.environ.get(f"{_base}_{_grp}") or os.environ.get(_base)
                if not _val:
                    continue
                for _pat in _patterns:
                    if _pat in _oci:
                        _old = getattr(_oci[_pat], _field, None)
                        setattr(_oci[_pat], _field, float(_val))
                        _arm_touched = True
                        log.warning(
                            f"RESUME override arm {_pat}.{_field}: {_old} -> {float(_val)}"
                        )
        # The pickle also carries the BAKED per-DOF control_info built at the
        # original launch; initialize_control_info() is a no-op when it exists
        # (hasattr guard), so overrides above would never reach the sim. Drop
        # it to force a rebuild from the MJCF + the mutated overrides.
        _ctrl = getattr(robot_config, "control", None)
        if _arm_touched and _ctrl is not None:
            # Rebuild the baked per-DOF control_info in place (nothing on the
            # resume path calls initialize_control_info, and the sim reads
            # control.control_info directly).
            if hasattr(_ctrl, "control_info"):
                delattr(_ctrl, "control_info")
            _ctrl.initialize_control_info(robot_config.asset)
            log.warning("RESUME override arm: rebuilt control_info from MJCF + overrides")

        # PM_GAIN_DR_LOW / PM_GAIN_DR_HIGH (env-gated, resume-safe):
        # MARIONETTE-mode actuator-gain DR range widening (2026-08-04).
        # simulator_config is frozen from the pickle, so teacher.py's
        # fresh-build gate never runs on a resume -- re-apply the sampled
        # gain-scale range here. The simulator is rebuilt from
        # simulator_config every boot and GAIN-DR resamples at build, so
        # mutating the ranges is sufficient. Both stiffness and damping
        # ranges move together (single knob pair). GUARD: fires only when
        # a var is EXPLICITLY PRESENT; unset = frozen ranges byte-identical.
        # Companion runtime knobs PM_PERTURB_GAIN_EXP / PM_PERTURB_SCALE_MIN
        # are read LIVE by the simulator (no config row needed; default
        # exp=0 = coupling OFF = byte-identical).
        # PER-GROUP (2026-08-04): PM_GAIN_DR_{LOW,HIGH}_{LEGS,WAIST,ARMS},
        # PM_GAIN_DR_CONSTANT_ZETA and PM_GAIN_DR_ENV_SCALE_SOURCE ride the
        # SAME gate (one shared implementation with the fresh-build path, so
        # the two can never drift). The gate is a hard no-op when none of the
        # knobs is present.
        from protomotions.simulator.base_simulator.gain_dr_env_gates import (
            apply_gain_dr_env_overrides,
        )

        apply_gain_dr_env_overrides(
            getattr(
                getattr(simulator_config, "domain_randomization", None),
                "actuator_gain",
                None,
            ),
            log_fn=log.warning,
            label="RESUME",
        )

        # PM_ACTION_NOISE_SCALE / PM_OBS_NOISE_SCALE / PM_ANCHOR_ROT_NOISE_SCALE
        # (env-gated, resume-safe): training observation/action noise magnitude
        # scaling (2026-08-07). Same shape and the SAME shared implementation as
        # the GAIN-DR gate above, so the fresh-build twin in teacher.py and this
        # resume row can never drift. simulator_config is frozen from the
        # pickle, so teacher.py's fresh-build gate never runs on a resume --
        # re-apply here. Action noise is resampled at every simulator build and
        # observation noise is read live off
        # simulator.config.domain_randomization every step, so mutating the
        # config is sufficient on both paths. GUARD: fires only when a var is
        # EXPLICITLY PRESENT, and a scale that resolves to exactly 1.0 writes
        # nothing -- unset (or 1.0) = frozen config byte-identical.
        from protomotions.simulator.base_simulator.noise_scale_env_gates import (
            apply_noise_scale_env_overrides,
        )

        apply_noise_scale_env_overrides(
            getattr(simulator_config, "domain_randomization", None),
            log_fn=log.warning,
            label="RESUME",
        )

        # v5.4 COMPONENT INJECTION (env-gated, resume-safe): reward components
        # are env-side -- adding one changes no obs/network shape -- but the
        # re-apply family above can only PATCH components already present in
        # the frozen config. Inject the dormant contact-channel /
        # swing-timing components (contact_match, liftoff_penalty,
        # action_smooth_lme) when their PM_* weight vars are set and the
        # component is absent. hold_balance / root_gain need no row here: the
        # HOLD-FIX boot path reads HOLD_BALANCE_BONUS / ROOT_GAIN_REWARD live
        # at env construction, which is rebuilt on every resume.
        from protomotions.envs.component_factories import (
            resume_inject_reward_components,
        )

        # dof_names comes from the FROZEN robot config so a resume can resolve
        # a PM_HOLD_JOINT_QUIET_JOINTS subset against the same DOF ordering the
        # run was built with. Absent/empty => a requested subset is a hard
        # error inside the builder, never a silent all-DOF fallback.
        if resume_inject_reward_components(
            _rc,
            log_fn=log.warning,
            dof_names=getattr(
                getattr(robot_config, "kinematic_info", None), "dof_names", None
            ),
        ):
            # _rc may be a fresh dict when the frozen config had None.
            env_config.reward_components = _rc

        args.checkpoint = checkpoint_path
        experiment_module = (
            None  # Intentionally skip loading - frozen config from pickle
        )

        # Warn if user tried to use overrides during resume
        if args.overrides:
            log.warning(
                "CLI overrides provided during RESUME will be IGNORED.\n"
                "Resume uses exact configs from resolved_configs.pt.\n"
                "For a new run with modified configs, use --checkpoint for warm start instead."
            )

        log.info(
            "RESUME: Using exact configs from first run (no config building, no CLI overrides)"
        )

    elif mode in ["warm_start", "fresh"]:
        # ===============================================================
        # WARM START / FRESH: Build configs from experiment file
        # Calls: configure_robot_and_simulator() → env_config() → agent_config()
        # ===============================================================
        log.info(f"{mode.upper()}: Building configs from experiment file")

        experiment_path = original_experiment_path
        log.info(f"Using original experiment path: {experiment_path}")
        args.checkpoint = checkpoint_path if mode == "warm_start" else None

        experiment_module = load_experiment_module(experiment_path)
        
        # Allow experiment files to add custom CLI arguments
        additional_args_fn = getattr(experiment_module, "additional_experiment_arguments", None)
        if additional_args_fn:
            additional_args_fn(parser)
        
        args = parser.parse_args()

        # Get required config functions
        terrain_config_fn = getattr(experiment_module, "terrain_config")
        scene_lib_config_fn = getattr(experiment_module, "scene_lib_config")
        motion_lib_config_fn = getattr(experiment_module, "motion_lib_config")
        env_config_fn = getattr(experiment_module, "env_config")

        # Get optional config functions
        configure_robot_and_simulator_fn = getattr(experiment_module, "configure_robot_and_simulator", None)
        agent_config_fn = getattr(experiment_module, "agent_config", None)

        from protomotions.utils.config_builder import build_standard_configs

        configs = build_standard_configs(
            args=args,
            terrain_config_fn=terrain_config_fn,
            scene_lib_config_fn=scene_lib_config_fn,
            motion_lib_config_fn=motion_lib_config_fn,
            env_config_fn=env_config_fn,
            configure_robot_and_simulator_fn=configure_robot_and_simulator_fn,
            agent_config_fn=agent_config_fn,
        )
        robot_config = configs["robot"]
        simulator_config = configs["simulator"]
        terrain_config = configs["terrain"]
        scene_lib_config = configs["scene_lib"]
        motion_lib_config = configs["motion_lib"]
        env_config = configs["env"]
        agent_config = configs["agent"]

        # Apply CLI overrides (highest priority)
        # NOTE: These overrides are saved to resolved_configs.pt and become permanent!
        # True resume will use these overridden values.
        if args.overrides:
            from protomotions.utils.config_utils import (
                apply_config_overrides,
                parse_cli_overrides,
            )

            cli_overrides = parse_cli_overrides(args.overrides)
            if cli_overrides:
                log.info(
                    f"Applying {len(cli_overrides)} CLI override(s) - these will be saved to resolved_configs.pt"
                )
                apply_config_overrides(
                    cli_overrides,
                    env_config,
                    simulator_config,
                    robot_config,
                    agent_config,
                    terrain_config=terrain_config,
                    motion_lib_config=motion_lib_config,
                    scene_lib_config=scene_lib_config,
                )

    # ===================================================================
    # 2b. Create Config Only Mode: Save configs and exit early
    # ===================================================================
    if args.create_config_only:
        _handle_create_config_only(
            args,
            save_dir,
            original_experiment_path,
            experiment_module,
            robot_config,
            simulator_config,
            terrain_config,
            scene_lib_config,
            motion_lib_config,
            env_config,
            agent_config,
        )
        return

    # ===================================================================
    # 3. Fabric Configuration: Loggers, Callbacks, Distributed Setup
    # ===================================================================
    loggers = [
        {"_target_": "lightning.fabric.loggers.TensorBoardLogger", "root_dir": save_dir}
    ]

    if args.use_wandb:
        loggers.append(
            {
                "_target_": "lightning.pytorch.loggers.WandbLogger",
                "name": args.experiment_name,
                "save_dir": save_dir,
                "project": "physical_animation",
                "tags": None,
                "group": None,
                "id": wandb_id,
                "entity": None,
                "resume": "allow",
            }
        )

    callbacks = []
    if args.use_slurm:
        callbacks.append(
            {
                "_target_": "agents.callbacks.slurm_autoresume_srun.AutoResumeCallbackSrun",
                "autoresume_after": 12600,
            }
        )

    from protomotions.utils.fabric_config import FabricConfig
    from lightning.fabric import Fabric

    fabric_config = FabricConfig(
        devices=args.ngpu,
        num_nodes=args.nodes,
        loggers=loggers,
        callbacks=callbacks,
    )
    print(fabric_config.as_loggable_dict())
    fabric: Fabric = Fabric(**fabric_config.as_kwargs())
    fabric.launch()

    # PM_CUDA_MEM_FRACTION (env-gated, prod-safety cap for co-located forks):
    # hard-cap THIS process's share of the CUDA device fabric pinned it to
    # (fabric.device), BEFORE any heavy sim/env allocation, so a fork that would
    # exceed its VRAM budget OOMs itself instead of a co-located production rank
    # sharing the GPU. No-op unless PM_CUDA_MEM_FRACTION is set.
    _pm_mem_frac = os.environ.get("PM_CUDA_MEM_FRACTION")
    if _pm_mem_frac and fabric.device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(_pm_mem_frac), device=fabric.device
        )

    # ===================================================================
    # 4. Environment Setup: IsaacLab, Seeding
    # ===================================================================
    # Setup IsaacLab simulation_app if using IsaacLab simulator
    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(fabric.device)}
        _stacked_on_one_gpu = os.environ.get("PM_STACK_RANKS_ON_GPU0") == "1"
        # PM_STACK_ACROSS_GPUS with S>1 (imprint #116/#117) also co-locates
        # multiple ranks on each physical GPU (S ranks/GPU across N GPUs via a
        # gloo group, world_size N*S). Like the single-GPU stacking hack, these
        # ranks must NOT use AppLauncher's distributed mode (which pins the
        # device to cuda:LOCAL_RANK, and LOCAL_RANK runs 0..N*S-1 -> points
        # PhysX at nonexistent GPU indices -> initialize_physics busy-waits
        # forever). Instead every rank boots plain single-GPU on the device the
        # fabric strategy already pinned it to (fabric.device = cuda:(rank//S)).
        # S=1 is one-rank-per-GPU with no co-location, so it keeps the standard
        # distributed path below (device = cuda:LOCAL_RANK is correct there).
        _stacked_across_gpus = os.environ.get("PM_STACK_ACROSS_GPUS") == "1" and (
            int(os.environ.get("PM_STACK_NRANKS", "1")) > 1
        )
        _colocated_ranks = _stacked_on_one_gpu or _stacked_across_gpus
        if fabric.world_size > 1 and not _colocated_ranks:
            # This is needed when running with SLURM.
            # When launching multi-GPU/node jobs without SLURM, or differently, maybe this needs to be adapted accordingly.
            app_launcher_flags["distributed"] = True
            os.environ["LOCAL_RANK"] = str(fabric.local_rank)
            os.environ["RANK"] = str(fabric.global_rank)
            # FORK-BOMB FIX (imprint fork-bomb incident 2026-07-23): the Kit
            # carb.tasking scheduler sizes its worker pool to *hardware
            # concurrency* (~nproc) PER RANK, and TBB_THREAD_COUNT / OMP_* env
            # vars do NOT bound it. With N ranks-per-node all booting Kit, that
            # is N*nproc runnable threads (8*112 ~ 900 load on gpu3202) => the
            # node fork-storm that OOM/SIGKILLed the 8x1x32768 DDP benchmark.
            # This branch (one rank/GPU, world_size>1) previously skipped the
            # cap that the co-located branch applies -> apply it here too.
            _tcap = os.environ.get("OMP_NUM_THREADS")
            if _tcap:
                os.environ.setdefault("PXR_WORK_THREAD_LIMIT", _tcap)
                # carb.tasking needs >=1 worker; a hard 1 can deadlock its
                # nested task-wait under high STACK. Decouple carb from OMP via
                # PM_CARB_THREAD_COUNT (fallback _tcap) so the launcher can
                # starve OMP/TBB/PXR to 1 for STACK>=4 while keeping carb at a
                # safe floor of 2 (imprint fork-bomb 2026-07-24, 8x5x8192).
                _carb = os.environ.get("PM_CARB_THREAD_COUNT", _tcap)
                sys.argv.append(
                    f"--/plugins/carb.tasking.plugin/threadCount={_carb}"
                )
            # Stagger Kit/PhysX boot per rank so the N ranks on a node do not
            # warm-start PhysX simultaneously (concurrent thread + PhysX-tensor
            # memory spike). PM_RANK_STAGGER_SEC is the DDP-path knob; the older
            # NFS_STAGGER_SEC (de-collide concurrent NFS motion-lib reads) is
            # honored too. Opt-in: default 0 = off, behavior unchanged.
            _stagger = max(
                float(os.environ.get("NFS_STAGGER_SEC", "0") or "0"),
                float(os.environ.get("PM_RANK_STAGGER_SEC", "0") or "0"),
            )
            if _stagger > 0:
                time.sleep(fabric.local_rank * _stagger)
        elif fabric.world_size > 1 and _colocated_ranks:
            # MPS-stacked mode (PM_STACK_RANKS_ON_GPU0=1): every rank of this
            # gloo group shares ONE physical GPU behind one MPS daemon, and the
            # process is masked to exactly one visible device
            # (CUDA_VISIBLE_DEVICES=0 within the daemon set); fabric pins every
            # rank's device to cuda:0. AppLauncher's distributed mode must NOT
            # be used here: it overrides device/physics_gpu/active_gpu to
            # cuda:<LOCAL_RANK> (isaaclab app_launcher.py _resolve_device_
            # settings), so co-located rank>=1 would point PhysX at a GPU that
            # does not exist behind the mask -> isaacsim simulation_manager
            # initialize_physics busy-waits forever while rank0 blocks in
            # fabric.all_gather (the 2026-07-18 8x2x8192 PhysX deadlock).
            # Instead: plain single-GPU boot on the one visible device, with
            # multi-GPU rendering off, plus the same CPU-thread caps that
            # distributed mode would have applied (from OMP_NUM_THREADS, which
            # the stacked launcher sets per-rank-count).
            # Pin to the device the fabric strategy chose for THIS rank:
            # cuda:0 for the single-GPU hack (parallel_devices=[cuda:0]*S), or
            # cuda:(rank//S) for PM_STACK_ACROSS_GPUS (parallel_devices spans
            # all N GPUs). str(fabric.device) yields the right one in both.
            app_launcher_flags["device"] = str(fabric.device)
            app_launcher_flags["multi_gpu"] = False
            os.environ["RANK"] = str(fabric.global_rank)
            # Per-rank scratch isolation: the launcher's TMPDIR/WARP_CACHE_PATH
            # are per-GPU, but co-located ranks would share them (Kit temp
            # files, warp kernel-cache builds racing on the same paths). Give
            # each rank its own subdir before Kit boots.
            import tempfile

            _rsub = f"r{fabric.local_rank}"
            _tmp = os.path.join(
                os.environ.get("TMPDIR", tempfile.gettempdir()), _rsub
            )
            os.makedirs(_tmp, exist_ok=True)
            os.environ["TMPDIR"] = _tmp
            tempfile.tempdir = None  # drop cached tempdir so the new TMPDIR wins
            if os.environ.get("WARP_CACHE_PATH"):
                _wc = os.path.join(os.environ["WARP_CACHE_PATH"], _rsub)
                os.makedirs(_wc, exist_ok=True)
                os.environ["WARP_CACHE_PATH"] = _wc
            _tcap = os.environ.get("OMP_NUM_THREADS")
            if _tcap:
                os.environ.setdefault("PXR_WORK_THREAD_LIMIT", _tcap)
                # carb.tasking needs >=1 worker; a hard 1 can deadlock its
                # nested task-wait under high STACK. Decouple carb from OMP via
                # PM_CARB_THREAD_COUNT (fallback _tcap) so the launcher can
                # starve OMP/TBB/PXR to 1 for STACK>=4 while keeping carb at a
                # safe floor of 2 (imprint fork-bomb 2026-07-24, 8x5x8192).
                _carb = os.environ.get("PM_CARB_THREAD_COUNT", _tcap)
                sys.argv.append(
                    f"--/plugins/carb.tasking.plugin/threadCount={_carb}"
                )
            _stagger = max(
                float(os.environ.get("NFS_STAGGER_SEC", "0") or "0"),
                float(os.environ.get("PM_RANK_STAGGER_SEC", "0") or "0"),
            )
            if _stagger > 0:
                time.sleep(fabric.local_rank * _stagger)
        # Key the Kit-init flock by PHYSICAL GPU so co-located ranks on one GPU
        # serialize their PhysX warm-start (deadlock defense-in-depth) while
        # ranks on different GPUs boot in parallel. Set for every multi-rank
        # case (incl. one-rank-per-GPU S=1 DDP): each rank then holds its own
        # per-GPU lock uncontended, so the flock does NOT globally serialize the
        # N boots (that job is left to PM_RANK_STAGGER_SEC). cuda:(rank//S) for
        # across-GPU DDP, cuda:0 for the single-GPU hack, local_rank fallback.
        if fabric.world_size > 1:
            _dev = getattr(fabric.device, "index", None)
            os.environ["PM_KIT_LOCK_KEY"] = str(
                _dev if _dev is not None else fabric.local_rank
            )
        from protomotions.utils.kit_init_lock import kit_init_lock

        with kit_init_lock("AppLauncher/Kit boot"):
            app_launcher = AppLauncher(app_launcher_flags)
        simulator_extra_params["simulation_app"] = app_launcher.app

        # Suppress verbose PhysX/IsaacLab warnings that flood stdout.
        # These warnings (e.g., "Stiffness unsupported for articulation joints",
        # "Could not perform modify_collision_properties") are harmless but produce
        # 10K+ log lines across multi-GPU runs, causing Lustre I/O backpressure
        # and NCCL timeouts during initialization.
        # References:
        #   https://forums.developer.nvidia.com/t/isaac-sim-log-level-to-many-log-messages/308545/3
        #   https://github.com/isaac-sim/IsaacLab/issues/1691
        import omni.log

        _omni_log = omni.log.get_log()
        for _channel in ["omni.physx.plugin", "isaaclab.sim.utils"]:
            _omni_log.set_channel_enabled(
                _channel, False, omni.log.SettingBehavior.OVERRIDE
            )

    if args.seed is not None:
        rank = fabric.global_rank if fabric.global_rank is not None else 0
        fabric.seed_everything(args.seed + rank)
        seeding(args.seed + rank, torch_deterministic=args.torch_deterministic)

    # ===================================================================
    # 5. Create Environment and Agent
    # ===================================================================
    # Note: Configs are already loaded/built in section 2 based on mode
    fabric.call(
        "on_app_start",
        fabric,
        {
            "fabric_config": fabric_config,
            "robot_config": robot_config,
            "simulator_config": simulator_config,
            "env_config": env_config,
            "agent_config": agent_config,
        },
    )
    fabric.call("on_env_init_start")

    # ===================================================================
    # 5a. Convert Friction for Simulator Compatibility
    # ===================================================================
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    # ===================================================================
    # 5b. Create Components
    # ===================================================================

    from protomotions.utils.component_builder import build_all_components

    save_dir_for_weights = (
        getattr(env_config, "save_dir", None)
        if hasattr(env_config, "save_dir")
        else None
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=save_dir_for_weights,
        **simulator_extra_params,  # simulation_app for IsaacLab
    )

    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

    # ===================================================================
    # 5c. Create Environment (auto-initializes simulator)
    # ===================================================================
    from protomotions.envs.base_env.env import BaseEnv

    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )
    fabric.call("on_env_init_end")

    from protomotions.agents.base_agent.agent import BaseAgent

    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(config=agent_config, env=env, fabric=fabric)

    agent.setup()
    agent.fabric.strategy.barrier()
    agent.load(
        args.checkpoint,
        load_training_state=(
            mode == "resume"
            or (mode == "warm_start" and args.warmstart_training_state)
        ),
    )

    # ===================================================================
    # 6. Save Configs (First Run Only - Warm Start or Fresh)
    # ===================================================================
    # Only save configs for warm_start or fresh modes (not resume)
    # Resume already has all configs saved from the original run
    is_first_run = mode in ["warm_start", "fresh"]

    if fabric.global_rank == 0 and is_first_run:
        if args.use_wandb:
            try_log_hyperparams_to_wandb(
                fabric,
                robot_config,
                simulator_config,
                terrain_config,
                scene_lib_config,
                motion_lib_config,
                env_config,
                agent_config,
                fabric_config,
            )

        save_configs(
            save_dir,
            args,
            robot_config,
            simulator_config,
            terrain_config,
            scene_lib_config,
            motion_lib_config,
            env_config,
            agent_config,
            fabric_config,
            experiment_source_path=original_experiment_path,
            file_name="resolved_configs",
        )

        from protomotions.utils.inference_utils import apply_all_inference_overrides
        from copy import deepcopy

        # Copy all configs to avoid eval parameters leaking into the training
        robot_config_inference = deepcopy(robot_config)
        simulator_config_inference = deepcopy(simulator_config)
        terrain_config_inference = deepcopy(terrain_config)
        scene_lib_config_inference = deepcopy(scene_lib_config)
        motion_lib_config_inference = deepcopy(motion_lib_config)
        env_config_inference = deepcopy(env_config)
        agent_config_inference = deepcopy(agent_config)
        apply_all_inference_overrides(
            robot_config_inference,
            simulator_config_inference,
            env_config_inference,
            agent_config_inference,
            terrain_config_inference,
            motion_lib_config_inference,
            scene_lib_config_inference,
            experiment_module=experiment_module,
            args=args,
        )
        prepare_inference_configs_for_save(
            robot_config_inference,
            simulator_config_inference,
            terrain_config_inference,
            scene_lib_config_inference,
            motion_lib_config_inference,
            env_config_inference,
            agent_config_inference,
        )
        save_configs(
            save_dir,
            args,
            robot_config_inference,
            simulator_config_inference,
            terrain_config_inference,
            scene_lib_config_inference,
            motion_lib_config_inference,
            env_config_inference,
            agent_config_inference,
            fabric_config,
            experiment_source_path=original_experiment_path,
            file_name="resolved_configs_inference",
        )

    agent.fabric.strategy.barrier()

    # Skip first policy update after resume to avoid training spike from full reset
    if mode == "resume":
        agent._skip_next_policy_update = True

    # ===================================================================
    # 7. Train
    # ===================================================================
    agent.fit()


def _handle_create_config_only(
    args,
    save_dir,
    experiment_source_path,
    experiment_module,
    robot_config,
    simulator_config,
    terrain_config,
    scene_lib_config,
    motion_lib_config,
    env_config,
    agent_config,
):
    """
    Handle --create-config-only mode: save configs and exit without training.

    This is useful for migrating old policy checkpoints when the config system API changes.
    Generate new configs compatible with current code, then load old weights with --checkpoint.

    Workflow:
        1. Run with --create-config-only to generate configs
        2. Run again with --checkpoint /path/to/old_weights.ckpt to train with old weights
    """
    from protomotions.utils.fabric_config import FabricConfig
    from protomotions.utils.inference_utils import apply_all_inference_overrides
    from copy import deepcopy

    # Create minimal fabric config (no loggers/callbacks needed for config-only mode)
    fabric_config = FabricConfig(
        devices=args.ngpu,
        num_nodes=args.nodes,
        loggers=[],
        callbacks=[],
    )

    # Save training configs
    save_configs(
        save_dir,
        args,
        robot_config,
        simulator_config,
        terrain_config,
        scene_lib_config,
        motion_lib_config,
        env_config,
        agent_config,
        fabric_config,
        experiment_source_path=experiment_source_path,
        file_name="resolved_configs",
    )

    # Save inference configs
    robot_config_inference = deepcopy(robot_config)
    simulator_config_inference = deepcopy(simulator_config)
    terrain_config_inference = deepcopy(terrain_config)
    scene_lib_config_inference = deepcopy(scene_lib_config)
    motion_lib_config_inference = deepcopy(motion_lib_config)
    env_config_inference = deepcopy(env_config)
    agent_config_inference = deepcopy(agent_config)
    apply_all_inference_overrides(
        robot_config_inference,
        simulator_config_inference,
        env_config_inference,
        agent_config_inference,
        terrain_config_inference,
        motion_lib_config_inference,
        scene_lib_config_inference,
        experiment_module=experiment_module,
        args=args,
    )
    prepare_inference_configs_for_save(
        robot_config_inference,
        simulator_config_inference,
        terrain_config_inference,
        scene_lib_config_inference,
        motion_lib_config_inference,
        env_config_inference,
        agent_config_inference,
    )
    save_configs(
        save_dir,
        args,
        robot_config_inference,
        simulator_config_inference,
        terrain_config_inference,
        scene_lib_config_inference,
        motion_lib_config_inference,
        env_config_inference,
        agent_config_inference,
        fabric_config,
        experiment_source_path=experiment_source_path,
        file_name="resolved_configs_inference",
    )

    log.info(f"CREATE CONFIG ONLY: Configs saved to {save_dir}")
    log.info("  - resolved_configs.pt / .yaml (training)")
    log.info("  - resolved_configs_inference.pt / .yaml (inference)")
    log.info(
        "Exiting without training. Use these configs with old policy checkpoints via --checkpoint."
    )


if __name__ == "__main__":
    main()
