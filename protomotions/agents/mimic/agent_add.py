# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re

import torch

from protomotions.agents.amp.agent import AMP
from protomotions.envs.obs.humanoid import compute_humanoid_max_coords_observations
from protomotions.utils.rotations import calc_heading
from lightning.fabric import Fabric
from typing import Optional
from pathlib import Path
from protomotions.envs.base_env.env import BaseEnv

log = logging.getLogger(__name__)

# Channels produced by ``compute_root_displacement_features``: root xy error
# in the current heading frame (2) + wrapped heading error (1).  How many are
# APPENDED to ``mimic_target_poses_diff`` is config-dependent:
# ``add_root_displacement_features`` appends the heading channel only;
# ``add_root_xy_features`` additionally appends the xy channels.  Track D
# final design (user decision 2026-07-10) ships BOTH OFF: xy+heading pressure
# lives in the task-reward channel (root_xy_displacement_rew /
# root_heading_rew factories) — training-time-only privileged signals,
# decoupled from deploy-time odometry quality. These flags remain as A/B
# options.
NUM_ROOT_DISPLACEMENT_FEATURES = 3


def compute_root_displacement_features(
    ref_root_pos: torch.Tensor,
    ref_root_rot: torch.Tensor,
    current_root_pos: torch.Tensor,
    current_root_rot: torch.Tensor,
    w_last: bool = True,
) -> torch.Tensor:
    """Differential root-displacement features for the ADD discriminator.

    Track D teacher-retrain objective 1 (minimize xy + heading displacement
    relative to the motion): explicit error channels so the ADD discriminator
    can auto-balance global drift toward zero.  Like the existing
    ``mimic_target_poses_diff`` (ref pose minus current pose), these are
    reference-minus-actual differentials that are exactly zero when tracking
    is perfect — matching ADD's zero-vector positive samples.

    Args:
        ref_root_pos: Reference root positions [num_envs, 3].
        ref_root_rot: Reference root rotations [num_envs, 4].
        current_root_pos: Current root positions [num_envs, 3].
        current_root_rot: Current root rotations [num_envs, 4].
        w_last: Quaternion layout (xyzw when True).

    Returns:
        [num_envs, 3] tensor: xy error expressed in the current root heading
        frame (2) followed by the wrapped heading error in radians (1).
    """
    xy_err_world = ref_root_pos[:, :2] - current_root_pos[:, :2]
    cur_heading = calc_heading(current_root_rot, w_last)
    cos_h = torch.cos(cur_heading)
    sin_h = torch.sin(cur_heading)
    # Rotate the world-frame error by -heading into the heading-local frame.
    xy_err_local = torch.stack(
        [
            cos_h * xy_err_world[:, 0] + sin_h * xy_err_world[:, 1],
            -sin_h * xy_err_world[:, 0] + cos_h * xy_err_world[:, 1],
        ],
        dim=-1,
    )
    ref_heading = calc_heading(ref_root_rot, w_last)
    heading_err = ref_heading - cur_heading
    # Wrap to [-pi, pi).
    heading_err = torch.remainder(heading_err + torch.pi, 2 * torch.pi) - torch.pi
    return torch.cat([xy_err_local, heading_err.unsqueeze(-1)], dim=-1)


def build_diff_feature_scale_vector(
    body_names,
    scales_spec: dict,
    num_appended_root_channels: int,
    device,
) -> torch.Tensor:
    """Per-feature scale vector for the ADD tracking differential.

    Feature layout (``compute_humanoid_max_coords_observations`` with
    ``local_obs=False, root_height_obs=True, observe_contacts=False`` — see
    ``protomotions/envs/obs/humanoid.py:229``):

        [root_h (1) | body pos, bodies 1..B-1 (3 each) | body rot tan-norm,
         bodies 0..B-1 (6 each) | body lin vel (3 each) | body ang vel
         (3 each) | appended root channels (0, 1 or 3)]

    ``scales_spec`` maps BODY-NAME REGEX -> scale (first matching pattern in
    insertion order wins; unmatched bodies scale 1.0). The special key
    ``"root_displacement"`` scales the appended root channels (when enabled);
    the root_h channel takes body 0's scale.

    Scaling is a SALIENCY BIAS on ADD's auto-balancing, not a hard weight —
    the discriminator still adapts. NOTE: with ``normalize_obs=True`` on the
    discriminator input, a static per-feature scale is asymptotically
    absorbed by the per-feature running normalizer (the bias is strongest
    early in training); set ``normalize_obs=False`` on the discriminator to
    make it permanent. Zero positives stay zero under any scaling.
    """
    body_scales = []
    patterns = [(k, float(v)) for k, v in scales_spec.items()
                if k != "root_displacement"]
    for name in body_names:
        s = 1.0
        for pat, val in patterns:
            if re.search(pat, name):
                s = val
                break
        body_scales.append(s)
    b = len(body_names)
    root_disp_scale = float(scales_spec.get("root_displacement", 1.0))

    parts = [torch.tensor([body_scales[0]])]                            # root_h
    parts.append(torch.tensor(body_scales[1:]).repeat_interleave(3))    # pos
    parts.append(torch.tensor(body_scales).repeat_interleave(6))        # rot
    parts.append(torch.tensor(body_scales).repeat_interleave(3))        # vel
    parts.append(torch.tensor(body_scales).repeat_interleave(3))        # angvel
    if num_appended_root_channels > 0:
        parts.append(torch.full((num_appended_root_channels,), root_disp_scale))
    vec = torch.cat(parts).to(device=device, dtype=torch.float32)
    expected = 1 + 3 * (b - 1) + 6 * b + 3 * b + 3 * b + num_appended_root_channels
    assert vec.shape[0] == expected, (vec.shape, expected)
    return vec


class MimicADD(AMP):
    def __init__(
        self, fabric: Fabric, env: BaseEnv, config, root_dir: Optional[Path] = None
    ):
        super().__init__(fabric, env, config, root_dir)

    # -----------------------------
    # Config-gated Track D extensions (all default OFF -> stock behavior)
    # -----------------------------
    @property
    def _root_displacement_features_enabled(self) -> bool:
        """A/B option (default OFF): append root-error channels to the diff."""
        return bool(
            getattr(getattr(self, "config", None), "add_root_displacement_features", False)
        )

    @property
    def _root_xy_features_enabled(self) -> bool:
        """A/B sub-option: also append the heading-frame xy channels (legacy
        3-channel form); otherwise only the heading channel is appended."""
        return bool(
            getattr(getattr(self, "config", None), "add_root_xy_features", False)
        )

    @property
    def _num_appended_root_channels(self) -> int:
        if not self._root_displacement_features_enabled:
            return 0
        return NUM_ROOT_DISPLACEMENT_FEATURES if self._root_xy_features_enabled else 1

    @property
    def _global_diff_bodies(self):
        """A/B option (default OFF — user decision 2026-07-10): bodies whose
        position diff keeps the FULL world-frame error (root displacement
        included). Rejected as the Track D default because a heavily-weighted
        GLOBAL wrist channel couples the top-priority precision signal to
        deploy-time frame-estimate quality (odom slip -> phantom offsets);
        root-stripped hand error depends only on proprioception."""
        return getattr(getattr(self, "config", None), "global_diff_bodies", None)

    @property
    def _diff_feature_scales_spec(self):
        return getattr(getattr(self, "config", None), "add_diff_feature_scales", None)

    def _env_body_names(self):
        robot_config = getattr(self.env, "robot_config", None)
        if robot_config is None:
            return None
        return list(robot_config.kinematic_info.body_names)

    def _global_diff_body_mask(self, num_bodies: int, device) -> torch.Tensor:
        """Bool mask [num_bodies] of bodies whose position diff stays GLOBAL."""
        cached = getattr(self, "_global_body_mask_cache", None)
        if cached is not None and cached.shape[0] == num_bodies:
            return cached.to(device)
        names = self._env_body_names()
        assert names is not None and len(names) == num_bodies, (
            "global_diff_bodies requires env.robot_config body names matching "
            f"the state tensor ({num_bodies} bodies)"
        )
        mask = torch.zeros(num_bodies, dtype=torch.bool)
        for i, n in enumerate(names):
            for pat in self._global_diff_bodies:
                if re.search(pat, n):
                    mask[i] = True
                    break
        assert mask.any(), (
            f"global_diff_bodies {self._global_diff_bodies} matched no body in "
            f"{names}"
        )
        self._global_body_mask_cache = mask
        return mask.to(device)

    def _diff_feature_scale_vector(self, total_dim: int, device):
        cached = getattr(self, "_diff_scale_vec_cache", None)
        if cached is not None and cached.shape[0] == total_dim:
            return cached.to(device)
        names = self._env_body_names()
        assert names is not None, (
            "add_diff_feature_scales requires env.robot_config body names"
        )
        vec = build_diff_feature_scale_vector(
            names,
            self._diff_feature_scales_spec,
            self._num_appended_root_channels,
            device,
        )
        assert vec.shape[0] == total_dim, (
            "add_diff_feature_scales: layout mismatch — expected diff dim "
            f"{vec.shape[0]} from the max-coords layout, got {total_dim}. "
            "The scale vector assumes local_obs=False, root_height_obs=True, "
            "observe_contacts=False."
        )
        self._diff_scale_vec_cache = vec
        return vec

    # -----------------------------
    # Environment Interaction and Data Updates
    # -----------------------------
    def add_agent_info_to_obs(self, obs):
        obs = super().add_agent_info_to_obs(obs)

        motion_times = self.env.motion_manager.motion_times
        motion_ids = self.env.motion_manager.motion_ids
        ref_state = self.env.motion_lib.get_motion_state(motion_ids, motion_times)

        ref_state_gt = ref_state.rigid_body_pos.reshape(self.num_envs, -1, 3)
        ref_state_gt += (
            self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
                ref_state_gt
            )
        )
        ref_ground_heights = self.env.terrain.get_ground_heights(
            ref_state_gt[:, 0]
        ).clone()

        current_state = self.env.simulator.get_bodies_state()
        ground_heights = self.env.terrain.get_ground_heights(
            current_state.rigid_body_pos[:, 0]
        ).clone()

        # ADD uses local_obs=False for tracking diff observations
        local_obs = False
        root_height_obs = True
        observe_contacts = False  # ADD does not yet support contact based conditioning

        # Empty contact flags since observe_contacts is False
        empty_contacts = torch.zeros(
            self.num_envs, 0, dtype=torch.bool, device=ref_state_gt.device
        )

        ref_pose = compute_humanoid_max_coords_observations(
            body_pos=ref_state_gt,
            body_rot=ref_state.rigid_body_rot,
            body_vel=ref_state.rigid_body_vel,
            body_ang_vel=ref_state.rigid_body_ang_vel,
            ground_height=ref_ground_heights,
            body_contacts=empty_contacts,
            local_obs=local_obs,
            root_height_obs=root_height_obs,
            observe_contacts=observe_contacts,
            w_last=True,
        )

        current_pose = compute_humanoid_max_coords_observations(
            body_pos=current_state.rigid_body_pos,
            body_rot=current_state.rigid_body_rot,
            body_vel=current_state.rigid_body_vel,
            body_ang_vel=current_state.rigid_body_ang_vel,
            ground_height=ground_heights,
            body_contacts=empty_contacts,
            local_obs=local_obs,
            root_height_obs=root_height_obs,
            observe_contacts=observe_contacts,
            w_last=True,
        )

        tracking_diff_obs = ref_pose - current_pose
        tracking_diff_obs = tracking_diff_obs.view(self.num_envs, -1)

        if self._global_diff_bodies:
            # A/B option (default OFF): rebuild the body-position diff block
            # so the listed bodies keep the FULL world-frame position error
            # (root displacement included — drift + articulation together),
            # while every other body is root-stripped (articulation-relative
            # error only). Layout: the pos block covers bodies 1..B-1 at
            # channels [1 : 1+3*(B-1)].
            ref_pos = ref_state_gt
            cur_pos = current_state.rigid_body_pos.reshape(self.num_envs, -1, 3)
            num_bodies = cur_pos.shape[1]
            mask = self._global_diff_body_mask(num_bodies, cur_pos.device)
            glob_diff = ref_pos - cur_pos
            rel_diff = (ref_pos - ref_pos[:, :1]) - (cur_pos - cur_pos[:, :1])
            pos_diff = torch.where(mask.view(1, -1, 1), glob_diff, rel_diff)
            pos_block_end = 1 + 3 * (num_bodies - 1)
            assert tracking_diff_obs.shape[-1] >= pos_block_end, (
                tracking_diff_obs.shape, num_bodies
            )
            tracking_diff_obs = tracking_diff_obs.clone()
            tracking_diff_obs[:, 1:pos_block_end] = pos_diff[:, 1:].reshape(
                self.num_envs, -1
            )

        if self._root_displacement_features_enabled:
            # A/B option (default OFF): append root-error channels. Heading
            # only by default; add_root_xy_features restores the xy channels.
            root_displacement_features = compute_root_displacement_features(
                ref_root_pos=ref_state_gt[:, 0],
                ref_root_rot=ref_state.rigid_body_rot.reshape(self.num_envs, -1, 4)[:, 0],
                current_root_pos=current_state.rigid_body_pos[:, 0],
                current_root_rot=current_state.rigid_body_rot[:, 0],
                w_last=True,
            )
            if not self._root_xy_features_enabled:
                root_displacement_features = root_displacement_features[:, 2:3]
            tracking_diff_obs = torch.cat(
                [tracking_diff_obs, root_displacement_features], dim=-1
            )

        if self._diff_feature_scales_spec:
            # Track D per-body saliency bias (wrists > head > rest).
            scale_vec = self._diff_feature_scale_vector(
                tracking_diff_obs.shape[-1], tracking_diff_obs.device
            )
            tracking_diff_obs = tracking_diff_obs * scale_vec

        obs["mimic_target_poses_diff"] = tracking_diff_obs
        # Cache the differential width so expert (zero) samples can be built
        # without re-deriving it from env internals (see get_expert_disc_obs).
        self._tracking_diff_dim = int(tracking_diff_obs.shape[-1])

        # Track D observability: accumulate per-channel |feature| (PRE disc
        # normalizer, post-scaling) + the grace-mask fraction for the
        # per-epoch disc-input / grace TB scalars (cheap: one abs-mean per
        # collection step).
        with torch.no_grad():
            absmean = tracking_diff_obs.detach().abs().mean(dim=0)
            acc = getattr(self, "_diff_absmean_acc", None)
            if acc is None or acc.shape != absmean.shape:
                self._diff_absmean_acc = absmean.clone()
                self._diff_absmean_n = 1
            else:
                self._diff_absmean_acc += absmean
                self._diff_absmean_n += 1
            grace_fn = getattr(self.env.simulator, "get_action_rate_grace_mask", None)
            if grace_fn is not None:
                mask = grace_fn()
                if mask is not None:
                    self._grace_frac_acc = (
                        getattr(self, "_grace_frac_acc", 0.0)
                        + float(mask.float().mean())
                    )
                    self._grace_frac_n = getattr(self, "_grace_frac_n", 0) + 1
        return obs

    # -----------------------------
    # Track D comprehensive logging (2026-07-10): reward economics + DR force
    # events as cheap per-epoch TB scalars — nothing about the run's internal
    # economics should be invisible afterward (the L2C2/action-rate lesson).
    # -----------------------------
    def post_epoch_logging(self, training_log_dict):
        try:
            self._add_reward_economics_logging(training_log_dict)
        except Exception as exc:  # noqa: BLE001 — logging must never kill training
            log.debug("reward economics logging skipped: %s", exc)
        try:
            self._add_dr_force_logging(training_log_dict)
        except Exception as exc:  # noqa: BLE001
            log.debug("dr force logging skipped: %s", exc)
        try:
            self._add_disc_input_group_logging(training_log_dict)
        except Exception as exc:  # noqa: BLE001
            log.debug("disc input group logging skipped: %s", exc)
        super().post_epoch_logging(training_log_dict)

    _DIFF_GROUP_PATTERNS = (
        ("wrists", r".*_wrist_yaw_link"),
        ("wrist_aux", r".*_wrist_(roll|pitch)_link"),
        ("head", r".*head.*"),
        ("feet", r".*_ankle_roll_link"),
    )

    def _diff_channel_group_indices(self, dim: int):
        """Channel index lists per body group over the diff layout (cached)."""
        cached = getattr(self, "_diff_group_idx_cache", None)
        if cached is not None and cached[0] == dim:
            return cached[1]
        names = self._env_body_names()
        if names is None:
            return None
        b = len(names)
        layout_dim = 1 + 3 * (b - 1) + 6 * b + 3 * b + 3 * b
        appended = dim - layout_dim
        if appended < 0:
            return None

        def group_of(name):
            for gname, pat in self._DIFF_GROUP_PATTERNS:
                if re.search(pat, name):
                    return gname
            return "other"

        groups = ["other"]  # root_h
        for n in names[1:]:
            groups += [group_of(n)] * 3
        for block in (6, 3, 3):
            for n in names:
                groups += [group_of(n)] * block
        groups += ["appended_root"] * appended
        idx = {}
        for i, g in enumerate(groups):
            idx.setdefault(g, []).append(i)
        idx = {g: torch.tensor(ii, dtype=torch.long) for g, ii in idx.items()}
        self._diff_group_idx_cache = (dim, idx)
        return idx

    def _add_disc_input_group_logging(self, training_log_dict):
        """Per-body-group disc-input |feature| PRE and POST normalizer.

        The continuously-logged saliency curve: pre-normalizer group means
        show the scaled raw error magnitudes; post-normalizer (pre / running
        sigma) shows what the discriminator effectively sees — i.e. how much
        of the wrists-3x bias the running normalizer has absorbed so far.
        """
        acc = getattr(self, "_diff_absmean_acc", None)
        n = getattr(self, "_diff_absmean_n", 0)
        if acc is None or n == 0:
            return
        pre = acc / n
        self._diff_absmean_acc = None
        self._diff_absmean_n = 0
        idx = self._diff_channel_group_indices(pre.shape[0])
        if idx is None:
            return
        # Disc running normalizer sigma (post = pre / sigma).
        sigma = None
        try:
            sd = self.model.state_dict()
            var = sd.get("_discriminator.models.0.norm.running_obs_norm.var")
            if var is not None and var.shape == pre.shape:
                sigma = torch.sqrt(var.to(pre.device) + 1e-5)
        except Exception:  # noqa: BLE001
            sigma = None
        for g, ii in idx.items():
            ii = ii.to(pre.device)
            training_log_dict[f"disc_input/{g}_absmean_pre"] = float(
                pre[ii].mean())
            if sigma is not None:
                training_log_dict[f"disc_input/{g}_absmean_post"] = float(
                    (pre[ii] / sigma[ii]).mean())
        # Grace-window activation fraction (action-rate suspension).
        gn = getattr(self, "_grace_frac_n", 0)
        if gn > 0:
            training_log_dict["dr_force/action_rate_grace_frac"] = (
                getattr(self, "_grace_frac_acc", 0.0) / gn
            )
            self._grace_frac_acc = 0.0
            self._grace_frac_n = 0

    def _add_reward_economics_logging(self, training_log_dict):
        """Per-term reward magnitudes vs the dominant (disc) term.

        Emits ``reward_ratios/<term>_vs_disc`` = |scaled env term per-step
        mean| / |weighted disc reward per-step mean| for every scaled env
        reward component, plus ``reward_ratios/task_total_vs_disc`` — the
        standing scale-drift monitor.
        """
        buf = getattr(self, "experience_buffer", None)
        if buf is None or not hasattr(buf, "amp_rewards"):
            return
        disc_w = float(self.config.amp_parameters.discriminator_reward_w)
        disc_mean = abs(float(buf.amp_rewards.mean())) * disc_w
        training_log_dict["reward_ratios/disc_per_step_abs"] = disc_mean
        denom = max(disc_mean, 1e-9)
        env_means = self.episode_env_tensors.mean()
        task_total = 0.0
        for key, value in env_means.items():
            if not key.startswith("scaled_r/"):
                continue
            name = key[len("scaled_r/"):]
            v = abs(float(value))
            task_total += float(value)
            training_log_dict[f"reward_ratios/{name}_vs_disc"] = v / denom
        training_log_dict["reward_ratios/task_total_vs_disc"] = (
            abs(task_total) * float(self.config.task_reward_w) / denom
        )

    def _add_dr_force_logging(self, training_log_dict):
        """Persistent-force / wrench DR event snapshot (per scheduler entry).

        For each active wrench-class scheduler: fraction of envs currently
        loaded, mean/max applied force magnitude over loaded envs, and the
        configured cap — keyed by direction mode + first body name so the
        curriculum stage's DR scale is visible directly in TB.
        """
        sim = getattr(self.env, "simulator", None)
        if not getattr(sim, "_wrench_enabled", False):
            return
        for sched in sim._wrench_scheds:
            cfg = sched["cfg"]
            entry_names = sched.get("entry_names") or (cfg.body_names or ["body"])
            mode = getattr(cfg, "direction_mode", "uniform")
            tag = f"dr_force/{mode}_{entry_names[0]}"
            active = sched["active"]
            frac = float(active.float().mean())
            training_log_dict[f"{tag}/active_frac"] = frac
            training_log_dict[f"{tag}/cap_n"] = float(cfg.force_magnitude_range[1])
            if bool(active.any()):
                mags = sched["forces"][active].norm(dim=-1)
                mags = mags[mags > 0]
                if mags.numel():
                    training_log_dict[f"{tag}/applied_mag_mean_n"] = float(mags.mean())
                    training_log_dict[f"{tag}/applied_mag_max_n"] = float(mags.max())

    def get_expert_disc_obs(self, num_samples: int):
        if getattr(getattr(self, "config", None), "reference_obs_components", True):
            expert_disc_obs = super().get_expert_disc_obs(num_samples)
        else:
            # ADD positive samples are pure zero differentials: when the
            # discriminator consumes only ``mimic_target_poses_diff`` there is
            # nothing to compute from the motion lib, so the base-AMP
            # requirement for reference_obs_components does not apply.
            expert_disc_obs = {}
        cached = getattr(self, "_tracking_diff_dim", None)
        if cached is not None:
            # Set by add_agent_info_to_obs during collection (always runs
            # before dataset augmentation) — already includes any appended
            # root channels and per-feature scaling (zeros stay zero).
            obs_dim = cached
        else:
            obs_manager = getattr(self.env, "observation_manager", None)
            hist = getattr(obs_manager, "observation_history_buffers", {}) if obs_manager else {}
            if "max_coords_obs" in hist:
                obs_dim = hist["max_coords_obs"].data.shape[-1]
            else:
                obs_dim = expert_disc_obs.get("max_coords_obs", expert_disc_obs.get("historical_max_coords_obs", torch.empty(0))).shape[-1] // 8
            # Expert (positive) samples are zero differentials — extend by
            # the appended root channels to stay dimension-aligned.
            obs_dim += self._num_appended_root_channels
        tracking_diff_obs = torch.zeros(
            [num_samples, obs_dim],
            device=self.device,
        )
        expert_disc_obs["mimic_target_poses_diff"] = tracking_diff_obs

        return expert_disc_obs
