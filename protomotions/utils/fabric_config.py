# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for Lightning Fabric distributed training."""

import os
from datetime import timedelta
from typing import Dict, Any, Union, Optional, List
from omegaconf import DictConfig
from dataclasses import dataclass, field, fields
from lightning import fabric

from protomotions.utils.hydra_replacement import instantiate


def _default_ddp_strategy() -> fabric.strategies.DDPStrategy:
    """Build the default DDPStrategy with a configurable process-group timeout.

    2026-07-04 crash-rootcause fix: PyTorch's default 30-min (1800s) collective
    timeout was firing on BaseAgent.__init__'s one-time world-size all_gather
    whenever a rank's Isaac-env/motion-lib construction (documented to
    legitimately take 25+ min under NFS contention) pushed past 30 min,
    aborting the entire 8-rank job even though nothing was actually hung.
    PG_TIMEOUT_SEC (env, default 3600s = 1h) raises that ceiling above the
    known JIT/NFS-load variance without weakening real-hang detection (a
    genuine deadlock still eventually aborts, just later).
    """
    timeout_sec = int(os.environ.get("PG_TIMEOUT_SEC", "3600"))
    # 2026-07-07 8-rank stall root-cause fix: find_unused_parameters=True.
    # py-spy evidence (wbc_push/eval_artifacts/gpu2255_stall1_pyspy_20260707.txt
    # and ddp7_stall_pyspy_20260707.txt; identical signature before AND after
    # the rank-uniform Transformer mask fix 6f3037f): one rank futex-parked
    # forever inside _engine_run_backward -- its DDP reducer never sees grads
    # for some params, so its final gradient bucket all-reduce is never
    # launched -- while every peer spins in a CUDA stream sync at
    # handle_model_grad_clipping waiting on bucket all-reduces that require
    # the parked rank's participation. That is the canonical
    # rank-divergent-graph hang of find_unused_parameters=False: any
    # batch-content-dependent divergence in which params receive grads
    # (MaskedMimic's per-rank stochastic masking makes this reachable)
    # deadlocks with no error and no timeout attribution.
    # find_unused_parameters=True has the reducer mark unfired params ready
    # so backward always terminates; cost is one graph traversal per step.
    # Opt out via DDP_FIND_UNUSED_PARAMETERS=0 for graphs proven static.
    #
    # 2026-07-07 RCA adjudication (wbc_push/briefs/rank_stall_rca.*.md):
    # the layer below the hang is parameter REUSE, not just unused params --
    # MaskedMimicModel.forward() invokes self._trunk TWICE per iteration
    # (prior latent + privileged/encoder latent decode,
    # masked_mimic_model.py:133/152). Vanilla DDP cannot bucket reused
    # params safely: with find_unused_parameters=False that raced into the
    # silent futex stall; with find_unused_parameters=True alone it became
    # the deterministic all-rank "RuntimeError: Expected to mark a variable
    # ready only once" (ddp7 attempt-2 log). static_graph=True is the
    # PyTorch-documented mode for graphs with reused (and unused) params
    # that are stable across iterations: the reducer learns the true hook
    # schedule on iteration 1 and stops mis-firing on intermediate hooks.
    # Opt out via DDP_STATIC_GRAPH=0 if a future model genuinely changes
    # its graph across iterations (torch then errors loudly, not silently).
    find_unused = os.environ.get("DDP_FIND_UNUSED_PARAMETERS", "1") == "1"
    static_graph = os.environ.get("DDP_STATIC_GRAPH", "1") == "1"
    # 2026-07-13 night13 Gate A attempt-3 Epoch-1 deadlock fix:
    # broadcast_buffers defaults to True in torch DDP, which issues a
    # per-forward _sync_module_buffers BROADCAST collective in DDP._pre_forward.
    # py-spy (gateA_attempt3_deadlock_pyspy_20260713.txt) caught the 8-rank
    # fleet drifted a full minibatch apart across the PPO num_mini_epochs loop:
    # rank2 parked in that _distributed_broadcast_coalesced buffer-broadcast at
    # the START of minibatch M's actor_step (ppo/agent.py:444), 5 ranks already
    # PAST it (actor_step:473), and 2 ranks a step BEHIND still spinning on
    # minibatch M-1's gradient-bucket all-reduce at handle_model_grad_clipping.
    # The buffer-broadcast is a SECOND per-forward collective independent of the
    # (already static_graph-stabilised) gradient reducer; under the mini-epoch
    # loop its ordering desynced vs the grad all-reduce -> NCCL cross-collective
    # mismatch -> hard deadlock, GPU 7x100%/1x0%. The broadcast buffers are the
    # obs RunningMeanStd normaliser stats, which do NOT need re-broadcasting
    # every optimize forward (they are updated during rollout, not optimize),
    # so disabling it removes the offending collective with no behavioural cost.
    # Env-overridable to re-enable if a future model needs buffer sync.
    broadcast_buffers = os.environ.get("DDP_BROADCAST_BUFFERS", "0") == "1"
    # Single-GPU multi-rank stacking hack (imprint issue #92, "protomotions
    # single-GPU-DDP hack"): NCCL's bootstrap refuses two ranks sharing one
    # physical GPU inside the same communicator ("Duplicate GPU detected"),
    # even under MPS. Gated opt-in: when PM_STACK_RANKS_ON_GPU0=1, pin all
    # PM_STACK_NRANKS ranks' parallel_devices to cuda:0 (already masked to
    # the intended physical GPU via CUDA_VISIBLE_DEVICES at the process
    # level) and switch the process-group backend to gloo, which has no such
    # same-device restriction. Verified in #92: 2 stacked ranks + MPS gave
    # ~2.3x aggregate throughput over a single monolithic rank at equal total
    # envs (sim-bound Newton workload parallelizes across ranks while MPS
    # overlaps their GPU kernels). Kept off by default (still NCCL).
    if os.environ.get("PM_STACK_RANKS_ON_GPU0") == "1":
        import torch

        n = int(os.environ.get("PM_STACK_NRANKS", "2"))
        return fabric.strategies.DDPStrategy(
            parallel_devices=[torch.device("cuda", 0)] * n,
            process_group_backend="gloo",
            timeout=timedelta(seconds=timeout_sec),
            find_unused_parameters=find_unused,
            static_graph=static_graph,
            broadcast_buffers=broadcast_buffers,
        )
    # TRUE cross-GPU rank-stacking (imprint issues #116/#117): the prod
    # launchers spawned N independent single-GPU train_agent.py processes
    # (per-GPU MASTER_PORT + per-GPU --experiment-name + --ngpu 1) that never
    # all-reduce gradients across GPUs -> N divergent models, only gpu0 ever
    # evaluated. This mode builds ONE DDP process-group spanning all N physical
    # GPUs with PM_STACK_NRANKS=S ranks pinned to EACH GPU:
    #   parallel_devices = [cuda:0]*S + [cuda:1]*S + ... + [cuda:(N-1)]*S
    #   world_size        = N * S
    # so a single rendezvous / --experiment-name yields one data-parallel model.
    # N is derived from PM_NGPU, else the count of CUDA_VISIBLE_DEVICES entries,
    # else torch.cuda.device_count() (the launcher exports PM_NGPU=N and passes
    # --ngpu N*S so Lightning's `devices` int equals world_size).
    # The backend MUST be gloo when S>1: NCCL's bootstrap refuses two ranks
    # sharing one physical GPU inside one communicator ("Duplicate GPU
    # detected"), even under MPS -- the same restriction that forced gloo for the
    # single-GPU PM_STACK_RANKS_ON_GPU0 hack (imprint #92). S=1 is plain
    # one-rank-per-GPU cross-GPU DDP with no co-location, so it keeps the faster
    # NCCL backend by falling through to the default strategy below.
    if os.environ.get("PM_STACK_ACROSS_GPUS") == "1":
        s = int(os.environ.get("PM_STACK_NRANKS", "1"))
        if s > 1:
            import torch

            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            if os.environ.get("PM_NGPU"):
                n = int(os.environ["PM_NGPU"])
            elif cvd:
                n = len([x for x in cvd.split(",") if x.strip() != ""])
            else:
                n = torch.cuda.device_count()
            parallel_devices = [
                torch.device("cuda", g) for g in range(n) for _ in range(s)
            ]
            return fabric.strategies.DDPStrategy(
                parallel_devices=parallel_devices,
                process_group_backend="gloo",
                timeout=timedelta(seconds=timeout_sec),
                find_unused_parameters=find_unused,
                static_graph=static_graph,
                broadcast_buffers=broadcast_buffers,
            )
        # s == 1: plain cross-GPU (one rank per GPU) -> NCCL default below.
    return fabric.strategies.DDPStrategy(
        timeout=timedelta(seconds=timeout_sec),
        find_unused_parameters=find_unused,
        static_graph=static_graph,
        broadcast_buffers=broadcast_buffers,
    )


@dataclass
class FabricConfig:
    """Configuration for Lightning Fabric distributed training."""

    accelerator: str = field(
        default="gpu",
        metadata={"help": "Hardware accelerator: 'gpu', 'cpu', 'tpu', 'auto'."}
    )
    devices: Union[int, str] = field(
        default=1,
        metadata={"help": "Number of devices or 'auto' for all available."}
    )
    num_nodes: Union[int, str] = field(
        default=1,
        metadata={"help": "Number of nodes for distributed training.", "min": 1}
    )
    strategy: Union[Dict, fabric.strategies.Strategy] = field(
        default_factory=_default_ddp_strategy,
        metadata={"help": "Distributed training strategy (DDP, FSDP, etc)."}
    )
    precision: Union[str, int] = field(
        # FABRIC_PRECISION env hook (2026-07-08): lets a launcher opt a run into
        # bf16-mixed (H100-native, fp32 master weights, no loss scaling needed)
        # without touching call sites. Default unchanged (fp32).
        default_factory=lambda: os.environ.get("FABRIC_PRECISION", "32-true"),
        metadata={"help": "Training precision: '32-true', '16-mixed', 'bf16-mixed'. Env override: FABRIC_PRECISION."}
    )
    loggers: Optional[List[Union[Dict, fabric.loggers.Logger]]] = field(
        default=None,
        metadata={"help": "List of logging backends (WandB, TensorBoard, etc)."}
    )
    callbacks: Optional[List[Union[Dict, Any]]] = field(
        default=None,
        metadata={"help": "List of training callbacks."}
    )

    def __post_init__(self):
        # Single-GPU rank-stacking hack (companion to _default_ddp_strategy's
        # PM_STACK_RANKS_ON_GPU0 branch): the stacked DDPStrategy already pins
        # world_size via parallel_devices=[cuda:0]*n, but Lightning's connector
        # still calls CUDAAccelerator.parse_devices(devices) on the int `devices`
        # (= args.ngpu = n) and rejects gpu-ids [0..n-1] against the single
        # visible GPU ("You requested gpu: [0..n-1] But your machine only has:
        # [0]"). Coerce devices to "auto" so parse_devices resolves to the 1
        # visible GPU while the strategy's parallel_devices keeps world_size=n.
        # PM_STACK_ACROSS_GPUS with S>1 has the same Lightning device-parsing
        # problem: `devices` = args.ngpu = N*S, but the machine only has N
        # visible GPUs, so CUDAAccelerator.parse_devices([0..N*S-1]) rejects the
        # request. The strategy's parallel_devices (length N*S) already fixes
        # world_size, so coerce devices to "auto" (resolves to the N visible
        # GPUs). S=1 keeps devices=N (== visible GPU count, parses cleanly).
        _across = os.environ.get("PM_STACK_ACROSS_GPUS") == "1" and (
            int(os.environ.get("PM_STACK_NRANKS", "1")) > 1
        )
        if os.environ.get("PM_STACK_RANKS_ON_GPU0") == "1" or _across:
            self.devices = "auto"
        if self.strategy is not None and (
            isinstance(self.strategy, dict) or isinstance(self.strategy, DictConfig)
        ):
            self.strategy = instantiate(self.strategy)
        if self.loggers is not None:
            loggers = []
            for logger in self.loggers:
                if isinstance(logger, dict) or isinstance(logger, DictConfig):
                    loggers.append(instantiate(logger))
                else:
                    loggers.append(logger)
            self.loggers = loggers
        if self.callbacks is not None:
            callbacks = []
            for callback in self.callbacks:
                if isinstance(callback, dict) or isinstance(callback, DictConfig):
                    callbacks.append(instantiate(callback))
                else:
                    callbacks.append(callback)
            self.callbacks = callbacks

    def as_kwargs(self) -> Dict[str, Any]:
        """Return Fabric constructor kwargs without deep-copying live objects."""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def as_loggable_dict(self) -> Dict[str, Any]:
        """Return a safe summary for logs without touching logger internals."""

        def summarize(value: Any) -> Any:
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, (list, tuple)):
                return [summarize(item) for item in value]
            return value.__class__.__name__

        return {field.name: summarize(getattr(self, field.name)) for field in fields(self)}
