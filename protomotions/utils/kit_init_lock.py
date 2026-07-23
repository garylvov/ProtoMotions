# Serialization of IsaacSim/Kit + PhysX initialization across ranks that are
# co-located on ONE physical GPU under NVIDIA MPS (PM_STACK_RANKS_ON_GPU0=1).
#
# Why: two Kit instances booting / warm-starting PhysX concurrently on the same
# GPU behind one MPS daemon is an untested path; the 2026-07-18 8x2x8192 PhysX
# deadlock showed one co-located rank stuck busy-waiting in isaacsim
# simulation_manager.initialize_physics while its sibling waited in
# fabric.all_gather. The primary root cause was AppLauncher's distributed-mode
# device override (fixed in train_agent.py), but Kit/PhysX init is additionally
# serialized here as defense in depth.
#
# Knobs:
#   PM_SERIALIZE_KIT_INIT=0  -> disable the lock (default: enabled when ranks
#                               are co-located on a GPU: PM_STACK_RANKS_ON_GPU0=1
#                               OR PM_STACK_ACROSS_GPUS=1; a no-op otherwise).
#   Lockfile is keyed by PM_KIT_LOCK_KEY (the physical GPU index, set by
#   train_agent for co-located ranks) if present, else CUDA_MPS_PIPE_DIRECTORY
#   basename, so only ranks sharing ONE physical GPU serialize against each
#   other; ranks on different GPUs proceed in parallel.

import contextlib
import fcntl
import os
import time


def _enabled() -> bool:
    if os.environ.get("PM_SERIALIZE_KIT_INIT", "1") == "0":
        return False
    # Engage whenever multiple Kit/PhysX ranks may be co-located on one physical
    # GPU: the single-GPU stacking hack (PM_STACK_RANKS_ON_GPU0) OR the
    # across-GPU DDP path with S>1 ranks/GPU (PM_STACK_ACROSS_GPUS). No-op for
    # ordinary one-rank-per-GPU runs (nothing to serialize).
    return (
        os.environ.get("PM_STACK_RANKS_ON_GPU0") == "1"
        or os.environ.get("PM_STACK_ACROSS_GPUS") == "1"
    )


@contextlib.contextmanager
def kit_init_lock(tag: str):
    """Hold an exclusive flock for a Kit/PhysX init critical section.

    No-op unless stacked-on-one-GPU mode is active (see module docstring).
    Reentrant-safe across sections because the lock is released in `finally`;
    do not nest two sections in one rank.
    """
    if not _enabled():
        yield
        return
    key = os.environ.get("PM_KIT_LOCK_KEY", "")
    if not key:
        key = os.path.basename(
            os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "").rstrip("/")
        )
    key = key or "nogpu"
    path = f"/tmp/kit_init_{key}.lock"
    f = open(path, "w")
    t0 = time.time()
    print(f"[kit-init-lock] pid={os.getpid()} waiting on {path} ({tag})", flush=True)
    fcntl.flock(f, fcntl.LOCK_EX)
    print(
        f"[kit-init-lock] pid={os.getpid()} acquired {path} ({tag}) "
        f"after {time.time() - t0:.1f}s",
        flush=True,
    )
    try:
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()
        print(f"[kit-init-lock] pid={os.getpid()} released {path} ({tag})", flush=True)
