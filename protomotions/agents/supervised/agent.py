# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervised rollout imitation agent.

This agent collects rollouts with the student policy, labels those states with
an expert policy, and optimizes a configured supervision loss. Algorithms such
as MaskedMimic are experiment/model configurations of this generic loop.
"""

import os

import torch
from torch import Tensor
from tensordict import TensorDict
import logging

from protomotions.utils.config_utils import load_resolved_configs_from_checkpoint
from protomotions.utils.hydra_replacement import get_class
from typing import Tuple, Dict
from pathlib import Path

from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.agents.common.common import weight_init_trainable
from protomotions.agents.common.supervision import compute_supervision_loss
from protomotions.agents.optimizer.factory import instantiate_optimizer
from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.supervised.config import RolloutActor
from protomotions.agents.supervised.expert_utils import get_expert_actor_in_keys
from protomotions.agents.utils.normalization import RunningMeanStd

log = logging.getLogger(__name__)


class SupervisedAgent(BaseAgent):
    """Student/expert rollout agent for supervised distillation.

    The agent collects TensorDict rollouts, writes configured model outputs into
    the experience buffer, and optimizes ``SupervisionLossConfig``. Models and
    experiment files define which keys are predictions and labels.
    """

    model: BaseModel

    def create_model(self):
        model_cls = get_class(self.config.model._target_)
        model: BaseModel = model_cls(config=self.config.model)
        if not getattr(model, "skip_default_weight_init", False):
            model.apply(weight_init_trainable)

        # Optionally load a pre-trained expert model if provided.
        # Note: Expert observation components are loaded in the experiment file
        # and prefixed with "expert_" for use during distillation training.
        expert_model_path = self.config.expert_model_path
        if expert_model_path is not None:
            log.info(f"Loading expert model from: {expert_model_path}")

            checkpoint_path = Path(expert_model_path)
            assert (
                checkpoint_path.exists()
            ), f"Could not find expert model at {checkpoint_path}"

            resolved_configs = load_resolved_configs_from_checkpoint(checkpoint_path)

            self.expert_env_config = resolved_configs["env"]
            expert_agent_config: PPOAgentConfig = resolved_configs["agent"]

            # Create the expert model
            ExpertModelConfig = get_class(expert_agent_config.model._target_)
            expert_model: BaseModel = ExpertModelConfig(
                config=expert_agent_config.model
            )

            # Move model to device BEFORE materializing lazy modules
            expert_model = expert_model.to(self.device)
            expert_model.reset_rollout_context(
                num_envs=self.num_envs,
                device=self.device,
            )

            # Once model is created, we pass fabric to the RunningMeanStd modules.
            # This allows the modules to internally handle distributed aggregation of normalization moments.
            def pass_fabric_to_running_mean_std(module):
                if isinstance(module, RunningMeanStd):
                    module.fabric = self.fabric

            expert_model.apply(pass_fabric_to_running_mean_std)

            expert_actor = self._external_expert_module_from(expert_model)
            expert_actor_in_keys = get_expert_actor_in_keys(expert_agent_config)
            if not expert_actor_in_keys:
                expert_actor_in_keys = list(getattr(expert_actor, "in_keys", []))

            log.info("Materializing expert actor lazy modules...")
            # External experts are frozen inference modules. Only the actor is
            # needed to label actions; materializing the full actor-critic model
            # can require critic-only observations that the distillation env does
            # not provide.
            expert_model.eval()
            with torch.no_grad():
                dummy_obs = self.env.get_obs()
                # Build expert obs tensordict (strips "expert_" prefix from keys)
                dummy_obs_td = self.obs_dict_to_tensordict(dummy_obs)
                dummy_expert_obs_td = self._build_expert_obs_td(
                    dummy_obs_td, expert_actor_in_keys
                )
                if os.environ.get("PM_MM_DEBUG_SYNC") == "1":
                    torch.cuda.synchronize(self.device)
                    log.info("[mm-debug-sync] pre expert materialize OK")
                self._materialize_frozen_external_expert(
                    expert_actor,
                    dummy_expert_obs_td,
                )
                if os.environ.get("PM_MM_DEBUG_SYNC") == "1":
                    torch.cuda.synchronize(self.device)
                    log.info("[mm-debug-sync] expert materialize OK")

            # Load weights before any distributed wrapper changes module keys.
            pre_trained_expert = torch.load(
                str(checkpoint_path),
                map_location=self.device,
                weights_only=False,
            )
            self._load_external_expert_state(
                expert_model,
                pre_trained_expert["model"],
            )
            for param in expert_model.parameters():
                param.requires_grad = False
            if os.environ.get("PM_MM_DEBUG_SYNC") == "1":
                torch.cuda.synchronize(self.device)
                devs = {str(t.device) for t in expert_model.state_dict().values() if hasattr(t, "device")}
                log.info(f"[mm-debug-sync] expert weights loaded OK; expert tensor devices={devs}")

            # Keep the external expert as a plain frozen module. The trainable
            # student is wrapped by create_optimizers(); the expert only labels
            # rollouts and does not need gradient synchronization.
            self.expert_model = expert_model
            self.expert_actor = expert_actor
            self.expert_actor_in_keys = expert_actor_in_keys
            self.expert_model.eval()
        else:
            self.expert_model = None
            self.expert_actor = None
            self.expert_actor_in_keys = []

        return model

    @staticmethod
    def _external_expert_module_from(expert_model):
        wrapped_module = getattr(expert_model, "module", None)
        if wrapped_module is not None and (
            callable(wrapped_module)
            or hasattr(wrapped_module, "_actor")
            or hasattr(wrapped_module, "actor")
        ):
            expert_model = wrapped_module
        return getattr(
            expert_model,
            "_actor",
            getattr(expert_model, "actor", expert_model),
        )

    def _external_expert_module(self):
        expert_actor = getattr(self, "expert_actor", None)
        if expert_actor is not None:
            return expert_actor
        return self._external_expert_module_from(self.expert_model)

    def _materialize_frozen_external_expert(self, expert_actor, dummy_obs_td):
        """Materialize a frozen expert without updating distributed normalizers."""
        freeze_states = []
        for module in expert_actor.modules():
            if hasattr(module, "_freeze_running"):
                freeze_states.append((module, module._freeze_running))
                module._freeze_running = True
        try:
            _ = expert_actor(dummy_obs_td)
        finally:
            for module, freeze_running in freeze_states:
                module._freeze_running = freeze_running

    def _load_external_expert_state(self, expert_model, model_state_dict):
        expert_actor = self._external_expert_module_from(expert_model)
        for prefix in ("_actor.", "actor."):
            actor_state_dict = {
                key[len(prefix) :]: value
                for key, value in model_state_dict.items()
                if key.startswith(prefix)
            }
            if actor_state_dict:
                expert_actor.load_state_dict(actor_state_dict)
                return

        expert_model.load_state_dict(model_state_dict)

    def _build_expert_obs_td(
        self, obs_td: TensorDict, expert_in_keys: list
    ) -> TensorDict:
        """Build expert observation TensorDict by stripping 'expert_' prefix from keys.

        The experiment file adds expert observation components with "expert_" prefix
        (e.g., "expert_max_coords_obs"). This method maps those back to the keys
        the expert model expects (e.g., "max_coords_obs").

        Args:
            obs_td: Full observation TensorDict with both student and expert_* keys
            expert_in_keys: List of keys the expert model expects

        Returns:
            TensorDict with keys matching expert model's in_keys
        """
        expert_obs = {}
        for key in expert_in_keys:
            expert_key = f"expert_{key}"
            if expert_key in obs_td.keys():
                # Prefer prefixed expert observation
                expert_obs[key] = obs_td[expert_key]
            else:
                raise KeyError(
                    f"Expert model requires observation '{expert_key}' for "
                    f"expert input '{key}'. Available keys: {list(obs_td.keys())}"
                )
        return TensorDict(expert_obs, batch_size=obs_td.batch_size, device=self.device)

    def create_optimizers(self, model: BaseModel):
        optimizer = instantiate_optimizer(
            self.config.model.optimizer,
            model.optimization_module(),
        )
        self.training_model, self.supervised_optimizer = self._setup_model_optimizer(
            model,
            optimizer,
        )

    # -----------------------------
    # Training Loop and Dataset Processing
    # -----------------------------
    def register_algorithm_experience_buffer_keys(self):
        if self.expert_model is not None:
            self.experience_buffer.register_key(
                "expert_actions",
                shape=(self.env.robot_config.number_of_actions,),
            )
            # Track C action-delta matching: store the expert's PREVIOUS
            # action per step so the loss can supervise action deltas
            # (velocity gain/direction). Only registered when the term is on
            # so stock runs keep an identical buffer layout. The tracker
            # tensor persists across epochs (epoch boundaries are not episode
            # boundaries); fresh-episode samples are masked in the loss via
            # the all-zero previous_actions heuristic.
            if getattr(self.config, "expert_action_delta_weight", 0.0) > 0:
                num_actions = self.env.robot_config.number_of_actions
                self.experience_buffer.register_key(
                    "expert_prev_actions",
                    shape=(num_actions,),
                )
                self._expert_prev_actions = torch.zeros(
                    self.num_envs, num_actions, device=self.device
                )

    def register_algorithm_experience_buffer_keys_from_obs(self, obs_td: TensorDict):
        target_key = self.config.loss.target_key
        if hasattr(self.experience_buffer, target_key):
            return

        if target_key in obs_td.keys():
            value = obs_td[target_key]
        else:
            with self._eval_model_for_buffer_registration(), torch.no_grad():
                output_td = self._collect_rollout_output(obs_td.clone())
            if target_key not in output_td.keys():
                raise KeyError(
                    f"Supervised loss target_key '{target_key}' was not produced by "
                    f"the rollout output. Available keys: {list(output_td.keys())}"
                )
            value = output_td[target_key]

        self.experience_buffer.register_key(
            target_key,
            shape=value.shape[1:],
            dtype=value.dtype,
        )

    def _collect_external_expert_action(self, obs_td: TensorDict) -> torch.Tensor:
        expert_actor = self._external_expert_module()
        expert_in_keys = getattr(self, "expert_actor_in_keys", None)
        if not expert_in_keys:
            expert_in_keys = list(getattr(expert_actor, "in_keys", []))
        expert_obs_td = self._build_expert_obs_td(
            obs_td,
            expert_in_keys,
        )
        expert_output_td = expert_actor(expert_obs_td)
        if "mean_action" in expert_output_td.keys():
            return expert_output_td["mean_action"]
        if "action" in expert_output_td.keys():
            return expert_output_td["action"]
        raise KeyError(
            "External expert actor must produce either 'mean_action' or 'action'. "
            f"Available keys: {list(expert_output_td.keys())}"
        )

    def _collect_rollout_output(self, obs_td: TensorDict) -> TensorDict:
        rollout_actor = self.config.rollout_actor
        if rollout_actor not in (RolloutActor.STUDENT, RolloutActor.EXPERT):
            raise ValueError(f"Unsupported supervised rollout_actor: {rollout_actor}")

        has_external_expert = self.expert_model is not None
        if rollout_actor == RolloutActor.EXPERT and not has_external_expert:
            model_expert_rollout = getattr(
                self.model,
                "collect_expert_rollout",
                None,
            )
            if model_expert_rollout is None:
                raise ValueError(
                    "rollout_actor=EXPERT needs an expert source: set "
                    "expert_model_path for an external expert, or use a model "
                    "that defines collect_expert_rollout."
                )
            output_td = model_expert_rollout(obs_td)
        else:
            output_td = self.model(obs_td)

        if has_external_expert:
            expert_action = self._collect_external_expert_action(obs_td)
            output_td["expert_actions"] = expert_action
            if rollout_actor == RolloutActor.EXPERT:
                output_td["action"] = expert_action
                output_td["mean_action"] = expert_action

        return output_td

    def collect_rollout_step(self, obs_td: TensorDict, step):
        """Collect student action and expert label for the current state."""
        output_td = self._collect_rollout_output(obs_td)

        if self.config.rollout_actor == RolloutActor.EXPERT:
            action = output_td["action"]
        elif "privileged_action" in output_td:
            action = output_td[
                "privileged_action"
            ]  # During training, we use the privileged action
        else:
            action = output_td["action"]  # During evaluation, we use the action

        # Store model outputs
        output_keys = list(
            dict.fromkeys(list(self.model_output_keys) + [self.config.loss.target_key])
        )
        for key in output_keys:
            if key in output_td:
                self.experience_buffer.update_data(key, step, output_td[key])
            elif key not in obs_td.keys():
                raise KeyError(
                    f"Supervised rollout output did not contain required key '{key}'. "
                    f"Available keys: {list(output_td.keys())}"
                )

        if self.expert_model is not None and "expert_actions" not in output_keys:
            self.experience_buffer.update_data(
                "expert_actions", step, output_td["expert_actions"]
            )

        # Action-delta matching: write e_{t-1} for this step, then roll the
        # tracker forward to e_t for the next step.
        expert_prev = getattr(self, "_expert_prev_actions", None)
        if expert_prev is not None and "expert_actions" in output_td.keys():
            self.experience_buffer.update_data(
                "expert_prev_actions", step, expert_prev
            )
            self._expert_prev_actions = output_td["expert_actions"].detach().clone()

        output_td["action"] = action
        return output_td

    def perform_optimization_step(self, batch_dict, batch_idx) -> Dict:
        # Update model
        iter_log_dict = {}
        loss, loss_dict = self.supervised_step(batch_dict)
        iter_log_dict.update(loss_dict)
        grad_clip_dict = self._step_optimizer(
            loss=loss,
            model=self.training_model,
            optimizer=self.supervised_optimizer,
            model_name="model",
        )
        iter_log_dict.update(grad_clip_dict)

        return iter_log_dict

    # -----------------------------
    # Model Forward Pass and Loss Computation
    # -----------------------------
    def supervised_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        """Compute supervised imitation loss from a rollout batch."""
        # Convert to TensorDict and run model forward
        batch_td = TensorDict(batch_dict, batch_size=batch_dict["action"].shape[0])
        batch_td = self.training_model(batch_td)

        supervised_loss, supervised_log_dict = self._compute_supervision_loss(
            batch_td,
        )
        actions = (
            batch_td["privileged_action"]
            if "privileged_action" in batch_td.keys()
            else batch_td["action"]
        )

        extra_loss, extra_log_dict = self.calculate_extra_loss(batch_td, actions)

        model_loss, model_log_dict = self.model.compute_model_loss(
            batch_td,
            current_epoch=self.current_epoch,
            zero_loss=supervised_loss,
            log_prefix="model",
        )

        loss = supervised_loss + extra_loss + model_loss

        log_dict = {
            "supervised/loss": supervised_loss.detach(),
            "supervised/extra_loss": extra_loss.detach(),
            "supervised/model_loss": model_loss.detach(),
            "losses/supervised_loss": loss.detach(),
        }
        log_dict.update(supervised_log_dict)
        log_dict.update(model_log_dict)
        log_dict.update(extra_log_dict)

        return loss, log_dict

    def _compute_supervision_loss(self, batch_td: TensorDict) -> Tuple[Tensor, Dict]:
        """Configured supervision loss, with optional per-dim MSE weighting.

        Track C: ``action_dim_weights`` (getattr: old pickles predate the
        field) re-weights the imitation MSE per action dim (robot dof order),
        normalized by the mean weight so the total loss scale matches the
        unweighted MSE. Uniform weights reproduce F.mse_loss exactly.
        """
        loss_config = self.config.loss
        dim_weights = getattr(self.config, "action_dim_weights", None)
        if dim_weights is None or not loss_config.enabled:
            return compute_supervision_loss(batch_td, loss_config)

        from protomotions.agents.common.supervision import SupervisionLossType

        if SupervisionLossType(loss_config.loss_type) != SupervisionLossType.MSE:
            raise ValueError(
                "action_dim_weights is only supported for loss_type=mse, got "
                f"{loss_config.loss_type}"
            )

        prediction = batch_td[loss_config.prediction_key]
        target = batch_td[loss_config.target_key]
        weights = torch.as_tensor(
            dim_weights, dtype=prediction.dtype, device=prediction.device
        )
        if weights.shape != prediction.shape[-1:]:
            raise ValueError(
                f"action_dim_weights length {tuple(weights.shape)} does not "
                f"match action dim {prediction.shape[-1]}"
            )
        weights = weights / weights.mean()
        raw_loss = ((prediction - target).pow(2) * weights).mean()
        weighted_loss = raw_loss * loss_config.weight
        prefix = loss_config.log_prefix
        return weighted_loss, {
            f"{prefix}/mse": raw_loss.detach(),
            f"{prefix}/loss": weighted_loss.detach(),
        }

    def calculate_extra_loss(self, batch_dict, actions) -> Tuple[Tensor, Dict]:
        extra_loss = torch.tensor(0.0, device=self.device)
        log_dict: Dict = {}

        l2c2_weight = self.config.l2c2_weight
        if l2c2_weight > 0:
            l2c2_loss = self._calculate_l2c2_loss(batch_dict)
            extra_loss = extra_loss + l2c2_weight * l2c2_loss
            log_dict["supervised/l2c2_loss"] = l2c2_loss.detach()

        # Track C: action-rate penalty (getattr: old resolved_configs pickles
        # predate the field; treat missing as 0.0 / off).
        action_rate_weight = getattr(self.config, "action_rate_weight", 0.0)
        if action_rate_weight > 0:
            action_rate_loss = self._calculate_action_rate_loss(batch_dict, actions)
            extra_loss = extra_loss + action_rate_weight * action_rate_loss
            log_dict["supervised/action_rate_loss"] = action_rate_loss.detach()

        # v5.4 LME port: soft-L_inf over per-joint |delta action| — student-side
        # twin of the teacher's action_smooth_lme arm-flail tax. Keep the weight
        # SMALL: the BC (distillation) loss must dominate.
        lme_weight = getattr(self.config, "action_lme_weight", 0.0)
        if lme_weight > 0:
            lme_loss = self._calculate_action_lme_loss(batch_dict, actions)
            extra_loss = extra_loss + lme_weight * lme_loss
            log_dict["supervised/action_lme_loss"] = lme_loss.detach()

        # Track C: action-delta matching against the expert (velocity gain).
        delta_weight = getattr(self.config, "expert_action_delta_weight", 0.0)
        if delta_weight > 0:
            delta_loss = self._calculate_action_delta_loss(batch_dict, actions)
            extra_loss = extra_loss + delta_weight * delta_loss
            log_dict["supervised/action_delta_loss"] = delta_loss.detach()

        # FK Cartesian wrist loss. Both weights default to 0.0 and are read via
        # getattr (old resolved_configs pickles predate the fields), so an
        # unconfigured run pays ZERO FK cost and stays byte-identical.
        fk_pos_weight = getattr(self.config, "fk_wrist_pos_weight", 0.0)
        fk_ori_weight = getattr(self.config, "fk_wrist_ori_weight", 0.0)
        fk_global_weight = getattr(self.config, "fk_global_pos_weight", 0.0)
        if fk_pos_weight > 0 or fk_ori_weight > 0 or fk_global_weight > 0:
            # ONE context, ONE FK pass. compute_forward_kinematics_from_transforms
            # already produces every body's pose, so the all-body term is a slice
            # of work the wrist term was doing anyway -- its marginal cost when
            # the wrist term is already on is one subtract/square/mean.
            cached = self._fk_commanded_body_poses(batch_dict, actions)
            if fk_pos_weight > 0 or fk_ori_weight > 0:
                fk_pos_loss, fk_ori_loss = self._calculate_fk_wrist_loss(
                    batch_dict, actions, cached=cached
                )
                if fk_pos_weight > 0:
                    extra_loss = extra_loss + fk_pos_weight * fk_pos_loss
                    log_dict["supervised/fk_wrist_pos_loss"] = fk_pos_loss.detach()
                if fk_ori_weight > 0:
                    extra_loss = extra_loss + fk_ori_weight * fk_ori_loss
                    log_dict["supervised/fk_wrist_ori_loss"] = fk_ori_loss.detach()
            if fk_global_weight > 0:
                fk_global_loss = self._calculate_fk_global_loss(
                    batch_dict, actions, cached=cached
                )
                extra_loss = extra_loss + fk_global_weight * fk_global_loss
                log_dict["supervised/fk_global_pos_loss"] = fk_global_loss.detach()

        return extra_loss, log_dict

    # ------------------------------------------------------------------
    # FK Cartesian wrist loss
    # ------------------------------------------------------------------

    #: Cached, device-resident kinematic/index context for the FK wrist loss.
    #: Built ONCE (first gated call) and reused for every subsequent step.
    _fk_wrist_ctx = None

    def _fk_wrist_context(self, device: torch.device) -> Dict:
        """Build (once) everything the FK wrist loss needs per step.

        Everything here is static for the life of the run: the MJCF-derived
        ``KinematicInfo`` (already parsed by the robot config -- we never
        re-parse the MJCF), the action->joint-target transform parameters, and
        the flat indices into the reference observation. The per-step path
        therefore does pure tensor work only.
        """
        if self._fk_wrist_ctx is not None:
            return self._fk_wrist_ctx

        env = self.env
        robot_config = env.robot_config
        kinematic_info = robot_config.kinematic_info.to(device)
        body_names = list(kinematic_info.body_names)

        # The reference observation is expressed relative to the ROOT body
        # (build_sparse_target_poses subtracts body 0). The teacher's reward is
        # expressed relative to the ANCHOR body. They only agree when the anchor
        # IS the root, which is the case for every mimic robot config shipped
        # here (anchor_body_index == 0 == pelvis).
        anchor_body_index = int(getattr(robot_config, "anchor_body_index", 0) or 0)
        if anchor_body_index != 0:
            raise ValueError(
                "fk_wrist_*_weight requires anchor_body_index == 0 (root): the "
                "reference observation is ROOT-relative while the teacher's "
                f"reward is ANCHOR-relative, and this robot anchors at body "
                f"{anchor_body_index} ({body_names[anchor_body_index]})."
            )

        # --- bodies to score -------------------------------------------------
        wrist_body_names = getattr(self.config, "fk_wrist_body_names", None)
        if not wrist_body_names:
            naming = getattr(robot_config, "common_naming_to_robot_body_names", {}) or {}
            wrist_body_names = list(naming.get("all_left_hand_bodies", [])) + list(
                naming.get("all_right_hand_bodies", [])
            )
        if not wrist_body_names:
            raise ValueError(
                "fk_wrist_*_weight > 0 but no bodies to score: set "
                "fk_wrist_body_names, or give the robot config "
                "all_left_hand_bodies / all_right_hand_bodies."
            )
        missing = [n for n in wrist_body_names if n not in body_names]
        if missing:
            raise ValueError(
                f"fk_wrist_body_names {missing} are not robot bodies. "
                f"Available: {body_names}"
            )
        wrist_body_indices = torch.tensor(
            [body_names.index(n) for n in wrist_body_names],
            dtype=torch.long,
            device=device,
        )

        # --- all-body (global) term: bodies to score -------------------------
        # Default is EVERY body, matching the eval metric's body set
        # (mean_body_pos_error over all rigid bodies).
        global_body_names = getattr(self.config, "fk_global_body_names", None)
        if not global_body_names:
            global_body_names = list(body_names)
        missing_global = [n for n in global_body_names if n not in body_names]
        if missing_global:
            raise ValueError(
                f"fk_global_body_names {missing_global} are not robot bodies. "
                f"Available: {body_names}"
            )
        global_body_indices = torch.tensor(
            [body_names.index(n) for n in global_body_names],
            dtype=torch.long,
            device=device,
        )

        # --- conditionable bodies of the reference observation ---------------
        ref_key = getattr(
            self.config, "fk_wrist_ref_key", "masked_mimic_target_poses"
        )
        conditionable_body_ids = None
        obs_components = getattr(env.config, "observation_components", None) or {}
        ref_component = obs_components.get(ref_key)
        if ref_component is not None:
            conditionable_body_ids = (ref_component.static_params or {}).get(
                "conditionable_body_ids"
            )
        if conditionable_body_ids is None:
            trackable = getattr(robot_config, "trackable_bodies_subset", None)
            if not trackable:
                raise ValueError(
                    f"Could not resolve the conditionable body order for "
                    f"'{ref_key}': the observation component does not declare "
                    "conditionable_body_ids and the robot config has no "
                    "trackable_bodies_subset."
                )
            conditionable_body_ids = [body_names.index(n) for n in trackable]
        conditionable_body_ids = [int(i) for i in conditionable_body_ids]

        wrist_slots = []
        for name in wrist_body_names:
            body_id = body_names.index(name)
            if body_id not in conditionable_body_ids:
                raise ValueError(
                    f"'{name}' is not in the masked-mimic conditionable set "
                    f"{[body_names[i] for i in conditionable_body_ids]}, so "
                    f"'{ref_key}' carries no reference pose for it."
                )
            wrist_slots.append(conditionable_body_ids.index(body_id))
        wrist_slots = torch.tensor(wrist_slots, dtype=torch.long, device=device)

        # --- all-body reference layout ---------------------------------------
        # `mimic_target_poses` (build_max_coords_target_poses) is the ONLY
        # all-body reference this recipe carries. It covers EVERY rigid body
        # (no conditionable subsetting -- num_bodies comes straight off the
        # reference state), unlike masked_mimic_target_poses which carries only
        # the 6 conditionable bodies.
        #
        # Per future step it concatenates, IN THIS ORDER:
        #     target_body_pos      (3*NB)  <- the block we want
        #     target_body_pos_rel  (3*NB)  ] only when with_relative
        #     target_body_rot      (6*NB)
        #     target_rel_body_rot  (6*NB)  ] only when with_relative
        #     local_target_vel     (3*NB)  ] only when with_velocities
        #     local_target_ang_vel (3*NB)  ]
        # and the result is viewed as [envs, steps * features_per_step], i.e.
        # STEP-MAJOR. target_body_pos is always FIRST within a step, so its
        # offset is 0 regardless of the flags; only the per-step STRIDE moves.
        global_ref_key = getattr(
            self.config, "fk_global_ref_key", "mimic_target_poses"
        )
        global_ref_component = obs_components.get(global_ref_key)
        global_static = (
            global_ref_component.static_params if global_ref_component else None
        ) or {}
        # Factory defaults (mimic_target_poses_max_coords_factory) are both True.
        with_relative = bool(global_static.get("with_relative", True))
        with_velocities = bool(global_static.get("with_velocities", True))
        global_features_per_body = (
            3 + 6 + (3 + 6 if with_relative else 0) + (3 + 3 if with_velocities else 0)
        )
        if global_ref_component is None:
            log.info(
                "FK global loss: '%s' is not a declared observation component; "
                "assuming the factory defaults (with_relative=True, "
                "with_velocities=True -> %d features/body/step).",
                global_ref_key,
                global_features_per_body,
            )

        # --- action -> joint position targets --------------------------------
        # Reuse the env's OWN action pipeline so the FK input is exactly the
        # joint-position target the simulator would receive (tanh/clamp
        # transform + PD offset/scale). No second copy of that math.
        action_config = getattr(env.config, "action_config", None)
        if not action_config or "fn" not in action_config:
            raise ValueError(
                "fk_wrist_*_weight > 0 requires env.config.action_config (a PD "
                "position-target action pipeline); the FK loss is only "
                "meaningful when the action IS a joint-position target."
            )
        action_fn = action_config["fn"]
        action_params = {
            key: (value.to(device) if isinstance(value, Tensor) else value)
            for key, value in action_config.items()
            if key != "fn"
        }

        # --- root rotation source, validated ONCE against the real obs spec ---
        # The rotation block sits after root height + body positions. Both
        # offsets are derived from the robot's own body count.
        num_bodies = kinematic_info.num_bodies
        root_rot_key = getattr(
            self.config, "fk_wrist_root_rot_obs_key", "max_coords_obs"
        )
        root_rot_offset = 1 + 3 * (num_bodies - 1)
        root_rot_min_width = root_rot_offset + 12 * num_bodies
        if root_rot_key is not None:
            root_rot_component = obs_components.get(root_rot_key)
            if root_rot_component is not None:
                static_params = root_rot_component.static_params or {}
                # A width check can NEVER catch this: local_obs=False produces a
                # tensor of exactly the same width whose rotations are WORLD, not
                # heading-local, which would silently compare two different
                # frames. Reject it up front.
                if static_params.get("local_obs", True) is not True:
                    raise ValueError(
                        f"fk_wrist_root_rot_obs_key='{root_rot_key}' is built "
                        "with local_obs=False, so its body rotations are WORLD "
                        "rotations, not heading-local ones. Point the key at a "
                        "local_obs=True max-coords observation, or set it to "
                        "None to skip the heading-local correction."
                    )
            else:
                log.info(
                    "FK wrist loss: '%s' is not a declared observation "
                    "component; its layout will be validated by width only.",
                    root_rot_key,
                )
            # The root rotation should come from a CLEAN observation. `use_noisy`
            # is a FACTORY argument, not a static_param -- it selects the context
            # binding -- so the only way to detect it is the dynamic_vars path.
            # Reading a noisy frame would price the policy for jitter it cannot
            # control, so warn loudly and name the fix.
            if root_rot_component is not None:
                bindings = root_rot_component.dynamic_vars or {}
                body_rot_path = getattr(bindings.get("body_rot"), "path", "")
                if str(body_rot_path).startswith("noisy"):
                    log.warning(
                        "FK wrist loss: root rotation is being read from '%s', "
                        "which is bound to the NOISY state ('%s'). The loss "
                        "frame will carry observation noise; prefer the clean "
                        "twin (e.g. 'clean_max_coords_obs') via "
                        "fk_wrist_root_rot_obs_key.",
                        root_rot_key,
                        body_rot_path,
                    )

            # FRAME CONSISTENCY. Every reference block is rotated into the
            # heading frame of the CURRENT root as that observation sees it. If
            # the reference is built from the clean state but the root rotation
            # is read from the noisy twin (this recipe's default: mimic_target_
            # poses is use_noisy=False while max_coords_obs is use_noisy=True),
            # the two sides live in DIFFERENT frames and the residual is
            # contaminated by the obs-noise realization. Width can never catch
            # this either.
            def _is_noisy(component, param):
                if component is None:
                    return None
                path = getattr((component.dynamic_vars or {}).get(param), "path", "")
                return str(path).startswith("noisy")

            root_noisy = _is_noisy(root_rot_component, "body_rot")
            for label, component, param in (
                ("fk_wrist_ref_key", ref_component, "current_state_body_rot"),
                (
                    "fk_global_ref_key",
                    global_ref_component,
                    "current_state_body_rot",
                ),
            ):
                ref_noisy = _is_noisy(component, param)
                if root_noisy is not None and ref_noisy is not None:
                    if root_noisy != ref_noisy:
                        log.warning(
                            "FK loss FRAME MISMATCH: the root rotation comes "
                            "from '%s' (noisy=%s) but %s's reference is built "
                            "from the %s state (noisy=%s). The two sides are "
                            "rotated into different heading frames; point "
                            "fk_wrist_root_rot_obs_key at the matching twin.",
                            root_rot_key,
                            root_noisy,
                            label,
                            "noisy" if ref_noisy else "clean",
                            ref_noisy,
                        )

        self._fk_wrist_ctx = {
            "kinematic_info": kinematic_info,
            "num_bodies": num_bodies,
            "global_ref_key": global_ref_key,
            "global_body_indices": global_body_indices,
            "global_body_names": list(global_body_names),
            "global_features_per_body": global_features_per_body,
            "root_rot_key": root_rot_key,
            "root_rot_offset": root_rot_offset,
            "root_rot_min_width": root_rot_min_width,
            "num_dofs": kinematic_info.num_dofs,
            "num_conditionable": len(conditionable_body_ids),
            "wrist_body_indices": wrist_body_indices,
            "wrist_slots": wrist_slots,
            "wrist_body_names": list(wrist_body_names),
            "action_fn": action_fn,
            "action_params": action_params,
        }
        return self._fk_wrist_ctx

    def _fk_wrist_reference(self, batch_td, ctx: Dict):
        """Slice the reference wrist pose + visibility masks out of the batch.

        ``masked_mimic_target_poses`` (build_sparse_target_poses,
        include_root_relative=True) is laid out as
        ``[envs, steps, conditionable_bodies, 2, 12]`` where the leading "2"
        splits TRANSLATION features from ROTATION features, and each 12 is
        ``[body-relative (6, zero-padded), root-relative (6, zero-padded)]``.
        We take the ROOT-RELATIVE halves: positions at ``[..., 0, 6:9]`` and the
        6D tan-norm rotation at ``[..., 1, 6:12]``. Both are already in the
        HEADING-LOCAL frame of the current root.
        """
        ref_key = getattr(
            self.config, "fk_wrist_ref_key", "masked_mimic_target_poses"
        )
        mask_key = getattr(
            self.config, "fk_wrist_ref_mask_key", "masked_mimic_target_masks"
        )
        for key in (ref_key, mask_key):
            if key not in batch_td.keys():
                raise KeyError(
                    f"fk_wrist_*_weight > 0 requires '{key}' in the training "
                    f"batch. Available keys: {list(batch_td.keys())}"
                )

        num_cond = ctx["num_conditionable"]
        ref = batch_td[ref_key]
        batch_size = ref.shape[0]
        features_per_step = num_cond * 24
        if ref.shape[-1] % features_per_step != 0:
            raise ValueError(
                f"'{ref_key}' width {ref.shape[-1]} is not a multiple of "
                f"{features_per_step} (= {num_cond} conditionable bodies x 24 "
                "features). The FK wrist loss needs "
                "include_root_relative=True sparse target poses."
            )
        num_steps = ref.shape[-1] // features_per_step
        step = int(getattr(self.config, "fk_wrist_future_step", 0))
        if not 0 <= step < num_steps:
            raise ValueError(
                f"fk_wrist_future_step={step} out of range for "
                f"{num_steps} future steps in '{ref_key}'."
            )

        ref = ref.view(batch_size, num_steps, num_cond, 2, 12)
        slots = ctx["wrist_slots"]
        ref_pos = ref[:, step, :, 0, 6:9].index_select(1, slots)
        ref_rot_tan_norm = ref[:, step, :, 1, 6:12].index_select(1, slots)

        masks = batch_td[mask_key]
        expected_mask_width = num_steps * num_cond * 2
        if masks.shape[-1] != expected_mask_width:
            raise ValueError(
                f"'{mask_key}' width {masks.shape[-1]} != expected "
                f"{expected_mask_width} (steps x bodies x 2) for '{ref_key}'."
            )
        masks = masks.view(batch_size, num_steps, num_cond, 2).to(ref_pos.dtype)
        pos_mask = masks[:, step, :, 0].index_select(1, slots)
        rot_mask = masks[:, step, :, 1].index_select(1, slots)
        return ref_pos, ref_rot_tan_norm, pos_mask, rot_mask

    def _fk_wrist_root_heading_local_rot(
        self, batch_td, ctx: Dict, like: Tensor
    ) -> Tensor:
        """Root rotation in the HEADING-LOCAL frame (i.e. its roll/pitch), [B,3,3].

        Read out of the max-coords humanoid observation
        (``compute_humanoid_max_coords_observations``), whose layout is exactly

            root_h (1) | body_pos (3*(NB-1)) | body_rot_tan_norm (6*NB)
                       | body_vel (3*NB) | body_ang_vel (3*NB) | [contacts]

        so the rotation block starts at ``1 + 3*(NB-1)`` -- derived from the
        robot's body count, never a hardcoded stride. With ``local_obs=True``
        every rotation is premultiplied by ``calc_heading_quat_inv(root_rot)``,
        and body 0 is the root, so the first 6 rotation features ARE
        ``heading_inv * root_rot``. Returns identity when the key is None.
        """
        from protomotions.components.pose_lib import quaternion_to_matrix
        from protomotions.utils.rotations import tan_norm_to_quat

        key = ctx["root_rot_key"]
        if key is None:
            eye = torch.eye(3, device=like.device, dtype=like.dtype)
            return eye.expand(like.shape[0], 3, 3)

        if key not in batch_td.keys():
            raise KeyError(
                f"fk_wrist_root_rot_obs_key='{key}' is missing from the "
                f"training batch. Available keys: {list(batch_td.keys())}"
            )
        obs = batch_td[key]
        rot_offset = ctx["root_rot_offset"]
        min_width = ctx["root_rot_min_width"]
        if obs.shape[-1] < min_width:
            raise ValueError(
                f"'{key}' width {obs.shape[-1]} is not a max-coords observation "
                f"of {ctx['num_bodies']} bodies: that is "
                f"1 + 3*(NB-1) + 12*NB = {min_width} features "
                "(root height, body positions, 6D body rotations, body linear "
                "and angular velocities), plus an optional contact tail. Set "
                "fk_wrist_root_rot_obs_key to the correct key, or to None to "
                "skip the heading-local correction."
            )
        root_tan_norm = obs[..., rot_offset : rot_offset + 6]
        root_quat = tan_norm_to_quat(root_tan_norm, w_last=True)
        return quaternion_to_matrix(root_quat, w_last=True)

    def _fk_require_reference_keys(self, batch_td) -> None:
        """Assert the batch carries the reference for every ENABLED FK term."""
        required = []
        if (
            getattr(self.config, "fk_wrist_pos_weight", 0.0) > 0
            or getattr(self.config, "fk_wrist_ori_weight", 0.0) > 0
        ):
            required.append(
                getattr(
                    self.config, "fk_wrist_ref_key", "masked_mimic_target_poses"
                )
            )
            required.append(
                getattr(
                    self.config, "fk_wrist_ref_mask_key", "masked_mimic_target_masks"
                )
            )
        if getattr(self.config, "fk_global_pos_weight", 0.0) > 0:
            required.append(
                getattr(self.config, "fk_global_ref_key", "mimic_target_poses")
            )
        available = list(batch_td.keys())
        for key in required:
            if key not in available:
                raise KeyError(
                    f"FK loss requires '{key}' in the training batch. "
                    f"Available keys: {available}"
                )

    def _fk_commanded_body_poses(self, batch_td, actions: Tensor):
        """FK the predicted action into COMMANDED poses for EVERY body.

        This is the single shared FK pass behind both the wrist term and the
        all-body term: ``compute_forward_kinematics_from_transforms`` already
        returns every body's pose, so scoring 29 bodies instead of 2 costs one
        extra subtract/square/mean, not a second FK.

        Returns:
            ``(ctx, body_pos, body_rot)`` with ``body_pos`` [B, NB, 3] and
            ``body_rot`` [B, NB, 3, 3], both expressed ROOT-RELATIVE and
            HEADING-LOCAL (see ``_calculate_fk_wrist_loss`` for why that frame).
        """
        from protomotions.components.pose_lib import (
            compute_forward_kinematics_from_transforms,
            extract_transforms_from_qpos_non_root,
        )

        ctx = self._fk_wrist_context(actions.device)
        if actions.shape[-1] != ctx["num_dofs"]:
            raise ValueError(
                f"FK loss expects the action to be a {ctx['num_dofs']}-dof "
                f"joint-position target, got action dim {actions.shape[-1]}."
            )
        # Validate the enabled terms' REFERENCE keys before touching anything
        # else, so a misconfigured run names its primary contract rather than
        # failing on the frame-correction observation it reads on the way there.
        self._fk_require_reference_keys(batch_td)

        # 1) action -> joint position targets (env's own pipeline, differentiable)
        joint_targets = ctx["action_fn"](actions, **ctx["action_params"])[
            "processed_action"
        ]

        # 2) FK in the pelvis body frame (root at origin, identity root rotation)
        kinematic_info = ctx["kinematic_info"]
        joint_rot_mats = extract_transforms_from_qpos_non_root(
            kinematic_info, joint_targets
        )
        root_pos = joint_targets.new_zeros(joint_targets.shape[0], 3)
        body_pos_pelvis, body_rot_pelvis = compute_forward_kinematics_from_transforms(
            kinematic_info, root_pos, joint_rot_mats
        )

        # 3) pelvis body frame -> heading-local frame, the reference's frame
        root_rot = self._fk_wrist_root_heading_local_rot(
            batch_td, ctx, body_pos_pelvis
        )  # [B, 3, 3]
        body_pos = torch.einsum("bij,bnj->bni", root_rot, body_pos_pelvis)
        body_rot = torch.einsum("bij,bnjk->bnik", root_rot, body_rot_pelvis)
        return ctx, body_pos, body_rot

    def _calculate_fk_global_loss(self, batch_td, actions: Tensor, cached=None):
        """All-body FK Cartesian tracking loss -- the eval metric's body set.

        The gating eval criterion is ``mean_body_pos_error`` (threshold 0.25 m,
        ``protomotions/envs/terminations/tracking.py``), a mean over EVERY rigid
        body. The BC action-MSE never sees Cartesian body error, and the wrist
        term scores 2 of 29 bodies; this term puts gradient on the whole set.

        REFERENCE: ``mimic_target_poses`` (``build_max_coords_target_poses``).
        This is the only genuinely ALL-BODY reference the recipe carries --
        it takes its body count straight off the reference state with no
        conditionable subsetting, unlike ``masked_mimic_target_poses``, which
        holds just the 6 conditionable bodies. We read its FIRST block,
        ``target_body_pos``.

        FRAME: root-relative, HEADING-LOCAL -- identical to the wrist term and
        to ``compute_relative_body_pos_rew``. ``target_body_pos`` is literally
        ``quat_rotate(calc_heading_quat_inv(current_root_rot),
        ref_body_pos - current_root_pos)``, so both sides of the residual are
        measured from the pelvis in the robot's own yaw-aligned frame.

        This is NOT the eval metric's frame (that one is world) and cannot be:
        FK of a joint-position target says nothing about where the root IS, so
        world error is not a function of the action. Root-relative error is
        exactly the share of the world metric the action can move -- get the
        body configuration right and the controllable part of
        ``mean_body_pos_error`` follows. The root's own drift is left to the
        rest of the objective.

        NOTE ON THE ROOT BODY: with the default all-body set, body 0 (pelvis)
        is included to match the metric's body set, but FK always places it at
        the origin while its reference is the root tracking error. It therefore
        contributes a CONSTANT with exactly zero gradient -- harmless, but it
        inflates the logged value and dilutes the mean by 1/NB. Exclude it via
        ``fk_global_body_names`` if you want the logged number to be purely the
        quantity being optimized.

        Same honest limitation as the wrist term: FK(action) is the COMMANDED
        pose, not the ACHIEVED one -- no PD tracking error, gravity droop or
        contact. It is a surrogate for ``mean_body_pos_error``, not that metric.
        """
        if cached is None:
            cached = self._fk_commanded_body_poses(batch_td, actions)
        ctx, body_pos, _body_rot = cached

        ref_pos = self._fk_global_reference(batch_td, ctx).detach()
        idx = ctx["global_body_indices"]
        commanded = body_pos.index_select(1, idx)

        # Mean over bodies of the squared position residual, in metres^2 --
        # the same form as the wrist term, so the two weights are comparable.
        return (commanded - ref_pos).pow(2).sum(dim=-1).mean()

    def _fk_global_reference(self, batch_td, ctx: Dict) -> Tensor:
        """Slice ``target_body_pos`` for the scored bodies out of the batch.

        Layout (``build_max_coords_target_poses``, viewed [envs, steps, blocks]
        then flattened): STEP-MAJOR, and ``target_body_pos`` is always the FIRST
        block within a step, so its offset inside a step is 0 whatever the
        with_relative / with_velocities flags are. Only the per-step stride
        moves, and that is derived from the component's own static params and
        the robot's body count -- never a hardcoded number.
        """
        key = ctx["global_ref_key"]
        if key not in batch_td.keys():
            raise KeyError(
                f"fk_global_pos_weight > 0 requires '{key}' in the training "
                f"batch (the all-body reference). Available keys: "
                f"{list(batch_td.keys())}"
            )
        ref = batch_td[key]
        num_bodies = ctx["num_bodies"]
        features_per_step = ctx["global_features_per_body"] * num_bodies
        if ref.shape[-1] % features_per_step != 0:
            raise ValueError(
                f"'{key}' width {ref.shape[-1]} is not a multiple of "
                f"{features_per_step} (= {ctx['global_features_per_body']} "
                f"features/body x {num_bodies} bodies). Check that the "
                "component's with_relative / with_velocities flags match the "
                "observation actually being produced."
            )
        num_steps = ref.shape[-1] // features_per_step
        step = int(getattr(self.config, "fk_global_future_step", 0))
        if not 0 <= step < num_steps:
            raise ValueError(
                f"fk_global_future_step={step} out of range for {num_steps} "
                f"future steps in '{key}'."
            )
        start = step * features_per_step
        target_body_pos = ref[..., start : start + 3 * num_bodies]
        target_body_pos = target_body_pos.view(ref.shape[0], num_bodies, 3)
        return target_body_pos.index_select(1, ctx["global_body_indices"])

    def _calculate_fk_wrist_loss(self, batch_td, actions: Tensor, cached=None):
        """TRUE FK Cartesian wrist loss on the predicted action.

        The student's action IS a PD joint-position target, so running forward
        kinematics on it yields the COMMANDED wrist pose, differentiable w.r.t.
        the action. The teacher buys its wrist accuracy from
        ``relative_body_pos_rew_factory(body_indices=wrist_indices)`` /
        ``relative_body_ori_rew_factory``; rewards carry no gradient, so this is
        the BC-side stand-in for them.

        FRAME (matches ``compute_relative_body_pos_rew``): anchor-relative,
        HEADING-LOCAL -- wrist position measured from the pelvis, expressed in
        the robot's own yaw-aligned frame. NOT world. World would be dominated
        by root drift (a metre of pelvis translation error would swamp the ~13 cm
        wrist deficit we are trying to close), and the teacher's reward never
        sees world coordinates either. Concretely:
          * FK is run with ``root_pos = 0`` and identity root rotation, so its
            output is already anchor-relative in the PELVIS BODY frame;
          * that is then rotated by the root's heading-local rotation (its
            roll/pitch, read from ``max_coords_obs``) to land in the same
            heading-local frame the reference observation uses.

        HONEST LIMITATION: FK(action) is the COMMANDED wrist pose, not the
        ACHIEVED one. It ignores PD tracking error, gravity droop, joint
        friction/armature and contact, so a perfectly-zero loss here does NOT
        mean a perfectly-zero MuJoCo ``wrist_err``. The term shapes the whole arm
        chain toward the reference in Cartesian space -- which plain 27-dof
        action MSE does not do, since MSE weights every joint equally regardless
        of its Cartesian lever arm -- but it is a surrogate for, not identical
        to, the achieved-pose metric.

        Returns:
            ``(pos_loss, ori_loss)``, each a scalar. Both are computed whenever
            either weight is on; the caller adds only the enabled ones.
        """
        if cached is None:
            cached = self._fk_commanded_body_poses(batch_td, actions)
        ctx, body_pos, body_rot = cached

        # Reference first: it names the term's primary contract.
        ref_pos, ref_rot_tan_norm, pos_mask, rot_mask = self._fk_wrist_reference(
            batch_td, ctx
        )

        wrist_idx = ctx["wrist_body_indices"]
        wrist_pos = body_pos.index_select(1, wrist_idx)  # [B, W, 3]
        wrist_rot = body_rot.index_select(1, wrist_idx)  # [B, W, 3, 3]

        ref_pos = ref_pos.detach()
        ref_rot_tan_norm = ref_rot_tan_norm.detach()

        pos_sq_err = (wrist_pos - ref_pos).pow(2).sum(dim=-1)  # [B, W], metres^2
        pos_loss = (pos_sq_err * pos_mask).sum() / pos_mask.sum().clamp_min(1.0)

        # tan-norm = (R @ x_hat, R @ z_hat) = columns 0 and 2 of the rotation
        # matrix -- exactly what quat_to_tan_norm produces for the reference.
        wrist_tan_norm = torch.cat(
            (wrist_rot[..., :, 0], wrist_rot[..., :, 2]), dim=-1
        )
        ori_sq_err = (wrist_tan_norm - ref_rot_tan_norm).pow(2).sum(dim=-1)
        ori_loss = (ori_sq_err * rot_mask).sum() / rot_mask.sum().clamp_min(1.0)

        return pos_loss, ori_loss

    def _previous_actions_from_batch(self, batch_td, actions: Tensor) -> Tensor:
        if "previous_actions" not in batch_td.keys():
            raise KeyError(
                "This loss term requires a 'previous_actions' key in the "
                f"training batch. Available keys: {list(batch_td.keys())}"
            )
        previous_actions = batch_td["previous_actions"]
        if previous_actions.shape != actions.shape:
            # flattened action history (history_steps > 1), most recent first
            previous_actions = previous_actions.reshape(actions.shape[0], -1)[
                :, : actions.shape[-1]
            ]
        return previous_actions

    def _calculate_action_delta_loss(self, batch_td, actions: Tensor) -> Tensor:
        """Match the student's per-step action delta to the expert's.

        L = mean( dimw * ((a_t - a_{t-1}) - (e_t - e_{t-1}))^2 ) over samples
        with a valid previous step. previous_actions is exactly all-zero right
        after an episode reset (StateHistoryBuffer zeroes the action history),
        which also covers the tracker's stale e_{t-1} on those samples — both
        are masked out together.
        """
        for key in ("expert_actions", "expert_prev_actions"):
            if key not in batch_td.keys():
                raise KeyError(
                    f"expert_action_delta_weight > 0 requires '{key}' in the "
                    "training batch (registered by the supervised agent when "
                    "the term is enabled). Available keys: "
                    f"{list(batch_td.keys())}"
                )
        previous_actions = self._previous_actions_from_batch(batch_td, actions)
        expert_actions = batch_td["expert_actions"]
        expert_prev_actions = batch_td["expert_prev_actions"]

        delta_err = (
            (actions - previous_actions.detach())
            - (expert_actions - expert_prev_actions).detach()
        ).pow(2)

        dim_weights = getattr(self.config, "action_dim_weights", None)
        if dim_weights is not None:
            weights = torch.as_tensor(
                dim_weights, dtype=delta_err.dtype, device=delta_err.device
            )
            if weights.shape != delta_err.shape[-1:]:
                raise ValueError(
                    f"action_dim_weights length {tuple(weights.shape)} does "
                    f"not match action dim {delta_err.shape[-1]}"
                )
            delta_err = delta_err * (weights / weights.mean())

        valid = (previous_actions.abs().sum(dim=-1, keepdim=True) > 0).to(
            delta_err.dtype
        )
        denom = (valid.sum() * delta_err.shape[-1]).clamp_min(1.0)
        return (delta_err * valid).sum() / denom

    def _calculate_action_rate_loss(self, batch_td, actions: Tensor) -> Tensor:
        """Temporal smoothness: mean((prediction_t - action_{t-1})^2).

        ``actions`` is the supervised prediction for step t (privileged_action
        when present). ``previous_actions`` is the env obs component holding
        the action applied at t-1 (previous_actions_factory(history_steps=1)),
        so the difference is the per-step action rate of the student.
        """
        previous_actions = self._previous_actions_from_batch(batch_td, actions)
        return (actions - previous_actions.detach()).pow(2).mean()

    def _calculate_action_lme_loss(self, batch_td, actions: Tensor) -> Tensor:
        """Log-Mean-Exp (soft L_infinity) over per-joint |action delta|.

        Student port of the teacher's v5.4 ``action_smooth_lme`` reward: prices
        the single most violent joint (the arm-flail/chatter axis) that the
        mean-flavored action_rate term dilutes across all DOFs. No perturbation
        grace window here — supervised batches carry no perturbation schedule.
        lme = (1/beta) * log(mean(exp(beta * |delta|))) per sample, meaned.
        """
        import math as _math

        previous_actions = self._previous_actions_from_batch(batch_td, actions)
        beta = float(getattr(self.config, "action_lme_beta", 3.0))
        delta = (actions - previous_actions.detach()).abs()
        return (
            (torch.logsumexp(beta * delta, dim=-1) - _math.log(delta.shape[-1]))
            .div(beta)
            .mean()
        )

    def _calculate_l2c2_loss(self, batch_td: TensorDict) -> Tensor:
        """L2C2 Lipschitz-ratio regularizer ported from the PPO actor path."""
        obs_pairs = self.config.l2c2_obs_pairs
        if not obs_pairs:
            raise ValueError(
                "l2c2_weight > 0 requires at least one l2c2_obs_pairs entry."
            )

        prediction_key = self.config.loss.prediction_key
        if prediction_key not in batch_td.keys():
            raise KeyError(
                f"L2C2 prediction key '{prediction_key}' is missing. "
                f"Available keys: {list(batch_td.keys())}"
            )

        clean_td = batch_td.clone()
        prediction = batch_td[prediction_key]
        input_ss = prediction.new_zeros(())
        input_n = 0

        for noisy_key, clean_key in obs_pairs.items():
            if noisy_key not in batch_td.keys():
                raise KeyError(
                    f"L2C2 noisy observation key '{noisy_key}' is missing. "
                    f"Available keys: {list(batch_td.keys())}"
                )
            if clean_key not in batch_td.keys():
                raise KeyError(
                    f"L2C2 clean observation key '{clean_key}' is missing. "
                    f"Available keys: {list(batch_td.keys())}"
                )

            noisy_obs = batch_td[noisy_key]
            clean_obs = batch_td[clean_key]
            if noisy_obs.shape != clean_obs.shape:
                raise ValueError(
                    f"L2C2 observation pair '{noisy_key}'/'{clean_key}' has "
                    f"mismatched shapes: {tuple(noisy_obs.shape)} vs "
                    f"{tuple(clean_obs.shape)}"
                )

            clean_td[noisy_key] = clean_obs
            diff = noisy_obs - clean_obs
            input_ss = input_ss + diff.pow(2).sum()
            input_n += diff.numel()

        if input_n == 0:
            raise ValueError("l2c2_obs_pairs must reference non-empty tensors.")

        input_dist = (input_ss / input_n).detach()
        clean_td = self.training_model(clean_td)
        if prediction_key not in clean_td.keys():
            raise KeyError(
                f"L2C2 clean forward did not produce prediction key '{prediction_key}'. "
                f"Available keys: {list(clean_td.keys())}"
            )

        output_dist = (prediction - clean_td[prediction_key]).pow(2).mean()
        # Track C optional MSE form: plain MSE(pred_noisy, pred_clean), no
        # ratio, no clamp (getattr: old pickles predate the field).
        if getattr(self.config, "l2c2_mse_form", False):
            return output_dist
        # Stability (2026-07-08 v2 divergence RCA): the raw Lipschitz ratio
        # exploded to inf by ep10 (TB supervised/l2c2_loss 0.25 -> 3.5 -> inf)
        # and poisoned the weights. Two guards: (a) floor the input distance
        # at 1e-4 (near-zero noisy-clean gaps — e.g. post-reset noisy-cache
        # fallback steps — must not turn the ratio into a 1e8-scale loss);
        # (b) clamp the ratio itself so one bad batch cannot dominate the
        # gradient. Neither guard binds in the healthy regime observed
        # upstream (ratio O(0.1-1)).
        return (output_dist / input_dist.clamp_min(1e-4)).clamp_max(10.0)

    # -----------------------------
    # State Saving and Restoration
    # -----------------------------
    def get_state_dict(self, state_dict):
        state_dict = super().get_state_dict(state_dict)
        state_dict["supervised_optimizer"] = self.supervised_optimizer.state_dict()
        return state_dict

    def _load_training_state(self, state_dict):
        super()._load_training_state(state_dict)
        optimizer_state = state_dict.get(
            "supervised_optimizer",
            state_dict.get("maskedmimic_optimizer"),
        )
        if optimizer_state is None:
            raise KeyError("supervised_optimizer")
        self.supervised_optimizer.load_state_dict(optimizer_state)
