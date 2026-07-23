# SPDX-License-Identifier: Apache-2.0

"""ONNX export for MaskedMimic (VAE learned-prior) student policies.

Exports a ``SupervisedAgent``/``MaskedMimicModel`` checkpoint (prior
transformer -> 64-latent -> trunk MLP -> actions) to a single unified ONNX
model **without** running a simulator, with MEAN-LATENT inference baked in
(``latent = prior_mu`` -- the recommended deploy mode; the privileged encoder
is training-only and is not exported).

The graph bakes the full obs computation (exactly the checkpoint's own
``env.observation_components`` kernels: ``max_coords_obs``,
``masked_mimic_target_poses/masks/times/poses_masks``, ``historical_pose_obs``,
``previous_actions``) so the runtime only supplies RAW context tensors:

    current/noisy rigid body pos/rot/vel/ang_vel   (world frame, xyzw quats)
    masked_mimic ref_pos/ref_rot/masks/time_offsets
    historical rigid-body ring buffer (+ ground heights)
    historical actions

Outputs: ``actions`` (raw policy output -- feed back into the historical
action buffer) and ``joint_pos_targets`` (absolute PD targets) plus
stiffness/damping targets, same contract as ``export_bm_tracker_onnx.py``.

Usage
-----
::

    python deployment/export_masked_mimic_onnx.py \
        --checkpoint <ckpt_dir>/last.ckpt \
        --output     <out_dir>/ \
        [--fixture   mm_io fixture npz from masked_mimic_rollout --dump-io]

Outputs ``<output>/unified_pipeline.onnx`` + ``unified_pipeline.yaml``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

PM_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Mock context (shape-only stand-in for EnvContext during tracing)
# ---------------------------------------------------------------------------
class _Namespace:
    pass


def build_mock_context(
    num_bodies: int,
    num_dofs: int,
    num_masked_future_steps: int,
    num_conditionable_bodies: int,
    num_history_steps: int,
    num_contact_bodies: int,
    dt: float,
):
    """Random-but-valid EnvContext stand-in for ONNX tracing."""
    import torch
    import torch.nn.functional as F

    def q(*shape):
        return F.normalize(torch.randn(*shape, 4), dim=-1)

    mock = _Namespace()

    cur = _Namespace()
    cur.rigid_body_pos = torch.randn(1, num_bodies, 3)
    cur.rigid_body_rot = q(1, num_bodies)
    cur.rigid_body_vel = torch.randn(1, num_bodies, 3)
    cur.rigid_body_ang_vel = torch.randn(1, num_bodies, 3)
    cur.dof_pos = torch.randn(1, num_dofs)
    cur.dof_vel = torch.randn(1, num_dofs)
    mock.current = cur
    # At inference observation noise is unconfigured: the noisy view aliases
    # the clean tensors (same alias convention as export_bm_tracker_onnx).
    mock.noisy = cur
    mock.noisy_ground_heights = torch.zeros(1)
    mock.ground_heights = torch.zeros(1)
    mock.body_contacts = torch.zeros(1, num_contact_bodies)

    mm = _Namespace()
    S = num_masked_future_steps
    mm.ref_pos = torch.randn(1, S, num_bodies, 3)
    mm.ref_rot = q(1, S, num_bodies)
    mm.target_bodies_masks = (
        torch.rand(1, S * num_conditionable_bodies * 2) > 0.5
    ).float()
    mm.time_offsets = torch.rand(1, S)
    mm.target_poses_masks = torch.ones(1, S)
    mock.masked_mimic = mm

    hist = _Namespace()
    H = num_history_steps
    hist.rigid_body_pos = torch.randn(1, H, num_bodies, 3)
    hist.rigid_body_rot = q(1, H, num_bodies)
    hist.rigid_body_vel = torch.randn(1, H, num_bodies, 3)
    hist.rigid_body_ang_vel = torch.randn(1, H, num_bodies, 3)
    hist.ground_heights = torch.zeros(1, H)
    hist.body_contacts = torch.zeros(1, H, num_contact_bodies)
    hist.actions = torch.randn(1, H, num_dofs)
    mock.historical = hist

    mock.dt = dt  # non-tensor -> resolved to a traced constant
    return mock


def _resolve_attr_path(path: str, obj):
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


def find_resolved_configs(checkpoint: Path) -> Path:
    for cand in (
        checkpoint.parent / "resolved_configs_inference.pt",
        checkpoint.parent.parent / "model" / "resolved_configs_inference.pt",
        checkpoint.parent.parent / "resolved_configs_inference.pt",
        checkpoint.parent / "resolved_configs.pt",
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"resolved_configs_inference.pt not found next to {checkpoint}"
    )


# ---------------------------------------------------------------------------
# Mean-latent policy wrapper (prior_mu -> trunk), no encoder / no vae noise
# ---------------------------------------------------------------------------
def build_mean_latent_policy(agent_config, checkpoint_path: Path):
    """Instantiate prior + trunk, load weights, return a TensorDict policy."""
    import torch
    from protomotions.agents.common.latent import VAE_LATENT_KEY
    from protomotions.utils.hydra_replacement import get_class

    model_cfg = agent_config.model
    prior = get_class(model_cfg.prior._target_)(config=model_cfg.prior)
    trunk = get_class(model_cfg.trunk._target_)(config=model_cfg.trunk)

    class MeanLatentMaskedMimic(torch.nn.Module):
        """prior -> mu -> trunk. out_keys ['action'] for UnifiedPipelineModule."""

        def __init__(self, prior, trunk):
            super().__init__()
            self.prior = prior
            self.trunk = trunk
            trunk_in = [k for k in trunk.in_keys if k != VAE_LATENT_KEY]
            self.in_keys = list(dict.fromkeys(list(prior.in_keys) + trunk_in))
            self.out_keys = ["action"]

        def forward(self, td):
            td = self.prior(td)
            mu = td[self.prior.out_keys[0]]
            td[VAE_LATENT_KEY] = mu
            td = self.trunk(td)
            td["action"] = td[self.trunk.out_keys[0]]
            return td

    policy = MeanLatentMaskedMimic(prior, trunk)
    policy.eval()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    prior_state = {k[len("_prior."):]: v for k, v in state.items() if k.startswith("_prior.")}
    trunk_state = {k[len("_trunk."):]: v for k, v in state.items() if k.startswith("_trunk.")}
    if not prior_state or not trunk_state:
        raise ValueError(
            "checkpoint model state dict has no _prior./_trunk. keys -- not a "
            f"MaskedMimicModel checkpoint? (keys: {list(state)[:5]} ...)"
        )
    return policy, prior_state, trunk_state


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------
def export_masked_mimic(
    checkpoint: str,
    output_dir: str,
    validate: bool = True,
    fixture: str | None = None,
) -> Path:
    import torch
    from tensordict import TensorDict

    from protomotions.utils.export_utils import (
        ActionExportModule,
        ObservationExportModule,
        UnifiedPipelineModule,
    )

    checkpoint_path = Path(checkpoint).resolve()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    resolved_path = find_resolved_configs(checkpoint_path)
    log.info(f"Loading configs from {resolved_path}")
    resolved = torch.load(resolved_path, map_location="cpu", weights_only=False)

    robot_config = resolved["robot"]
    env_config = resolved["env"]
    agent_config = resolved["agent"]
    simulator_config = resolved.get("simulator")

    num_dofs = robot_config.kinematic_info.num_dofs
    body_names = list(robot_config.kinematic_info.body_names)
    joint_names = list(robot_config.kinematic_info.dof_names)
    num_bodies = len(body_names)
    trackable = list(robot_config.trackable_bodies_subset)
    conditionable_body_ids = [body_names.index(n) for n in trackable]

    mm_cfg = env_config.control_components["masked_mimic"]
    num_masked_future_steps = int(mm_cfg.num_masked_future_steps)
    num_history_steps = int(getattr(env_config, "num_state_history_steps", 5))
    contact_bodies = getattr(robot_config, "contact_bodies", None) or body_names
    num_contact_bodies = len(contact_bodies)

    # Timing (mujoco-converted, like export_bm_tracker_onnx).
    control_dt, physics_dt, decimation = 0.02, 0.001, 20
    pd_target_max_accel = None
    if simulator_config is not None:
        try:
            from protomotions.simulator.factory import update_simulator_config_for_test

            mj_cfg = update_simulator_config_for_test(
                current_simulator_config=simulator_config,
                new_simulator="mujoco",
                robot_config=robot_config,
            )
            physics_dt = 1.0 / mj_cfg.sim.fps
            decimation = mj_cfg.sim.decimation
            control_dt = physics_dt * decimation
            accel = getattr(mj_cfg, "pd_target_max_accel", None)
            if accel is not None:
                pd_target_max_accel = float(accel)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"sim2sim timing conversion failed: {exc}")

    # ------------------------------------------------------------------
    # Model in_keys -> obs component subset
    # ------------------------------------------------------------------
    from protomotions.agents.common.latent import VAE_LATENT_KEY

    prior_in = list(agent_config.model.prior.in_keys)
    trunk_in = [k for k in agent_config.model.trunk.in_keys if k != VAE_LATENT_KEY]
    model_in_keys = list(dict.fromkeys(prior_in + trunk_in))
    log.info(f"Model inference obs keys: {model_in_keys}")

    # Training-only stochastic augmentation knobs that some obs kernels accept
    # as static_params (e.g. masked_mimic's conditioning_noise_pos/rot teleop-
    # realism jitter -- see protomotions/envs/obs/masked_mimic.py). These are
    # baked into the ONNX graph as a FROZEN torch.randn() draw at trace time
    # (ONNX has no live RNG), so leaving them enabled at export time does not
    # give "noisy deploy conditioning" -- it gives one arbitrary, permanent
    # noise offset for the model's entire deployed lifetime, and it also makes
    # the exporter's own onnxruntime-vs-pytorch parity check meaningless (the
    # pytorch reference forward and the traced/exported graph each draw an
    # independent Gaussian sample, so the "parity" diff is actually just this
    # noise's magnitude, not an export defect). Deploy/export must always use
    # the clean (noise-free) conditioning kernel: real hardware/teleop signals
    # already carry their own real-world noise, so re-injecting synthetic
    # training-time jitter on top is never wanted at inference time.
    _TRAINING_ONLY_NOISE_PARAMS = ("conditioning_noise_pos", "conditioning_noise_rot")

    obs_configs = {}
    for key in model_in_keys:
        comp = env_config.observation_components.get(key)
        if comp is None:
            raise ValueError(f"obs component '{key}' missing from env config")
        stripped = {
            p: comp.static_params[p]
            for p in _TRAINING_ONLY_NOISE_PARAMS
            if comp.static_params.get(p, 0.0)
        }
        if stripped:
            import copy

            comp = copy.copy(comp)
            comp.static_params = dict(comp.static_params)
            for p in stripped:
                comp.static_params[p] = 0.0
            log.info(
                f"obs component '{key}': zeroed training-only noise params for "
                f"deploy export -- {stripped} -> 0.0"
            )
        obs_configs[key] = comp

    # ------------------------------------------------------------------
    # Mock context + obs module
    # ------------------------------------------------------------------
    torch.manual_seed(0)
    mock = build_mock_context(
        num_bodies=num_bodies,
        num_dofs=num_dofs,
        num_masked_future_steps=num_masked_future_steps,
        num_conditionable_bodies=len(conditionable_body_ids),
        num_history_steps=num_history_steps,
        num_contact_bodies=num_contact_bodies,
        dt=control_dt,
    )

    obs_module = ObservationExportModule(obs_configs, mock, device="cpu")
    obs_module.eval()
    obs_input_keys = obs_module.get_input_keys()
    obs_output_keys = obs_module.get_output_keys()
    log.info(f"Context input keys:  {obs_input_keys}")
    log.info(f"Observation outputs: {obs_output_keys}")

    # ------------------------------------------------------------------
    # Policy (mean-latent) + weights
    # ------------------------------------------------------------------
    policy, prior_state, trunk_state = build_mean_latent_policy(
        agent_config, checkpoint_path
    )

    # Materialise LazyLinear layers with one forward pass, then load weights.
    sample_inputs = [_resolve_attr_path(k, mock) for k in obs_input_keys]
    with torch.no_grad():
        obs_out = obs_module(*sample_inputs)
        obs_td = TensorDict(
            {k: v for k, v in zip(obs_output_keys, obs_out)}, batch_size=[1]
        )
        policy(obs_td.clone())
    policy.prior.load_state_dict(prior_state, strict=True)
    policy.trunk.load_state_dict(trunk_state, strict=True)
    policy.eval()
    log.info(
        "Loaded weights: prior=%d tensors, trunk=%d tensors"
        % (len(prior_state), len(trunk_state))
    )

    missing = set(model_in_keys) - set(obs_output_keys)
    if missing:
        raise ValueError(f"obs module does not produce policy inputs: {missing}")

    # ------------------------------------------------------------------
    # Unified pipeline
    # ------------------------------------------------------------------
    action_module = ActionExportModule(env_config.action_config, device="cpu")
    action_module.eval()

    unified = UnifiedPipelineModule(
        observation_module=obs_module,
        policy_module=policy,
        action_module=action_module,
        policy_in_keys=model_in_keys,
        policy_action_key="action",
    )
    unified.cpu().eval()

    with torch.no_grad():
        actions, pd_targets, stiffness_t, damping_t = unified(*sample_inputs)
    log.info(
        f"Forward pass OK: actions={list(actions.shape)}, "
        f"pd_targets={list(pd_targets.shape)}"
    )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _sanitize(name: str) -> str:
        return name.replace(".", "_").replace("[", "_").replace("]", "_")

    onnx_input_names = [_sanitize(k) for k in obs_input_keys]
    onnx_output_names = [
        "actions",
        "joint_pos_targets",
        "stiffness_targets",
        "damping_targets",
    ]

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
            **{n: {0: "batch_size"} for n in onnx_input_names},
            **{n: {0: "batch_size"} for n in onnx_output_names},
        },
        dynamo=False,
    )
    log.info(f"ONNX exported -> {onnx_path}")

    # Read back actual names (unused inputs may be pruned / renamed).
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual_in_names = [inp.name for inp in session.get_inputs()]
    actual_out_names = [out.name for out in session.get_outputs()]

    sanitized_to_key = {_sanitize(k): k for k in obs_input_keys}
    onnx_name_to_key = {}
    for onnx_name in actual_in_names:
        base = onnx_name
        for suffix in (".1", ".2", ".3", "_1", "_2", "_3"):
            if base.endswith(suffix) and base[: -len(suffix)] in sanitized_to_key:
                base = base[: -len(suffix)]
                break
        if base in sanitized_to_key:
            onnx_name_to_key[onnx_name] = sanitized_to_key[base]
        else:
            log.warning(f"Cannot map ONNX input '{onnx_name}' to a semantic key")

    dropped = [k for k in obs_input_keys if k not in onnx_name_to_key.values()]
    if dropped:
        log.info(f"Inputs pruned from the graph (unused): {dropped}")

    # ------------------------------------------------------------------
    # Validate (random mock inputs)
    # ------------------------------------------------------------------
    max_diff_overall = None
    if validate:
        import numpy as np

        key_to_tensor = {k: t for k, t in zip(obs_input_keys, sample_inputs)}
        ort_inputs = {
            n: key_to_tensor[onnx_name_to_key[n]].detach().numpy().astype(np.float32)
            for n in actual_in_names
            if n in onnx_name_to_key
        }
        ort_out = session.run(actual_out_names, ort_inputs)
        pt_out = [
            actions.detach().numpy(),
            pd_targets.detach().numpy(),
            stiffness_t.detach().numpy(),
            damping_t.detach().numpy(),
        ]
        max_diff_overall = 0.0
        for i, name in enumerate(onnx_output_names):
            diff = float(np.abs(ort_out[i] - pt_out[i]).max())
            max_diff_overall = max(max_diff_overall, diff)
            status = "OK" if diff < 1e-4 else "WARN"
            log.info(f"  {status}  {name}: max_diff = {diff:.2e}")
        log.info("Validation (random mock inputs) complete")

    # ------------------------------------------------------------------
    # Optional: fixture replay validation (stored real env inputs)
    # ------------------------------------------------------------------
    fixture_stats = None
    if fixture:
        import numpy as np

        fx = np.load(fixture, allow_pickle=True)
        steps = json.loads(str(fx["meta"]))["steps"]
        deltas = []
        for t in range(steps):
            feed = {}
            ok = True
            for n in actual_in_names:
                key = onnx_name_to_key.get(n)
                # dump-io stores context arrays as "ctx/<path>" with '/'
                # encoded as '__' (dots kept): e.g. ctx__noisy.rigid_body_pos
                arr_key = ("ctx__" + key) if key else None
                if arr_key is None or arr_key not in fx:
                    log.warning(f"fixture missing '{arr_key}' for ONNX input '{n}'")
                    ok = False
                    break
                feed[n] = fx[arr_key][t : t + 1].astype(np.float32)
            if not ok:
                log.warning("fixture missing a context array; skipping replay")
                break
            out = session.run(["actions"], feed)[0]
            deltas.append(np.abs(out[0] - fx["actions"][t]).max())
        if deltas:
            fixture_stats = {
                "steps": len(deltas),
                "max_action_delta": float(np.max(deltas)),
                "mean_action_delta": float(np.mean(deltas)),
            }
            log.info(
                "Fixture replay: %d steps, max action delta %.3e, mean %.3e"
                % (len(deltas), fixture_stats["max_action_delta"], fixture_stats["mean_action_delta"])
            )

    # ------------------------------------------------------------------
    # YAML metadata
    # ------------------------------------------------------------------
    stiffness_vals = [
        float(robot_config.control.control_info[j].stiffness) for j in joint_names
    ]
    damping_vals = [
        float(robot_config.control.control_info[j].damping) for j in joint_names
    ]
    try:
        effort_limits = [
            float(robot_config.control.control_info[j].effort) for j in joint_names
        ]
    except (AttributeError, KeyError):
        effort_limits = None

    asset_root = Path(getattr(robot_config.asset, "asset_root", ""))
    if not asset_root.is_absolute():
        asset_root = PM_ROOT / asset_root
    mjcf_abs = asset_root / robot_config.asset.asset_file_name

    input_shapes = {
        k: list(v.shape) for k, v in zip(obs_input_keys, sample_inputs)
    }

    yaml_content = {
        "type": "masked_mimic_unified_pipeline",
        "dt": control_dt,
        "joint_names": joint_names,
        "body_names": body_names,
        "default_joint_stiffness": stiffness_vals,
        "default_joint_damping": damping_vals,
        "policy_outputs": [
            {"name": n, "key": n, "shape": [1, num_dofs], "joint_names": joint_names}
            for n in onnx_output_names
        ],
        "_runtime": {
            "onnx_in_names": actual_in_names,
            "onnx_out_names": actual_out_names,
            "onnx_name_to_in_key": onnx_name_to_key,
            "obs_context_keys": obs_input_keys,
            "input_shapes": input_shapes,
            "pruned_context_keys": dropped,
        },
        "robot": {
            "mjcf_path": str(mjcf_abs),
            "num_bodies": num_bodies,
            "num_dofs": num_dofs,
            "root_body_name": body_names[0],
            "body_names": body_names,
            "joint_names": joint_names,
            "num_contact_bodies": num_contact_bodies,
        },
        "control": {
            "stiffness": stiffness_vals,
            "damping": damping_vals,
            "effort_limits": effort_limits,
            "pd_target_max_accel": pd_target_max_accel,
            "action_ema_alpha": 1.0,
        },
        "timing": {
            "control_dt": control_dt,
            "physics_dt": physics_dt,
            "decimation": decimation,
        },
        "masked_mimic": {
            "trackable_bodies_subset": trackable,
            "conditionable_body_ids": conditionable_body_ids,
            "num_masked_future_steps": num_masked_future_steps,
            "num_history_steps": num_history_steps,
            "time_alpha": float(mm_cfg.time_alpha),
            "time_beta": float(mm_cfg.time_beta),
            "historical_pose_obs_steps": 5,
        },
        "metadata": {
            "checkpoint": str(checkpoint_path),
            "inference_mode": "mean_latent (latent = prior_mu)",
            "validation_max_diff": max_diff_overall,
            "fixture_replay": fixture_stats,
        },
    }

    import yaml

    yaml_path = output_path / "unified_pipeline.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=None, sort_keys=False)
    log.info(f"YAML metadata -> {yaml_path}")

    return onnx_path


def _parse_args():
    p = argparse.ArgumentParser(
        description="Export a MaskedMimic student policy to ONNX (mean-latent)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="Path to *.ckpt")
    p.add_argument("--output", default=None, help="Output dir (default: <ckpt_dir>/compiled_models)")
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--fixture", default=None,
                   help="mm_io fixture npz (masked_mimic_rollout --dump-io) "
                        "for stored-input replay validation")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = args.output or str(Path(args.checkpoint).parent / "compiled_models")
    onnx_file = export_masked_mimic(
        checkpoint=args.checkpoint,
        output_dir=out,
        validate=not args.no_validate,
        fixture=args.fixture,
    )
    log.info(f"\nDone!  Model exported to: {onnx_file}")
