# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motion library for managing reference motion data.

This module provides the MotionLib class which stores and manages collections of
motion clips for use in motion tracking and imitation learning. It supports efficient
loading, sampling, and interpolation of motion data from various formats (.motion, .yaml, .pt).

Key Classes:
    - MotionLib: Main motion library class for motion management

Key Features:
    - Load motions from .motion files, YAML configs, or packaged .pt files
    - Weighted sampling of motions
    - Frame interpolation for smooth motion queries
    - Batched access for parallel environments
    - Distributed training support
"""

import json
import logging
import os
import re
from typing import Optional, Tuple
from easydict import EasyDict
from pathlib import Path

import torch
import yaml

from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    StateConversion,
)
from protomotions.utils.rotations import quat_to_exp_map
from dataclasses import dataclass, field

from protomotions.utils.motion_interpolation_utils import (
    interpolate_pos,
    interpolate_quat,
    calc_frame_blend,
)

log = logging.getLogger(__name__)

# --- Per-rank motion-lib sharding (env-gated, default off) -------------------
#
# When MOTION_LIB_SHARD_PER_RANK=1 and torch.distributed is initialized with
# world_size > 1, each DDP rank loads only an interleaved shard of a packaged
# (.pt) motion library (motions[rank::world_size]) instead of holding the full
# pack GPU-resident. Sampling weights are renormalized per class (class
# boundaries from the pack's "<pack>.mix.json" sidecar) so that each rank's
# per-class sampling mass fractions exactly match the global pack.
_MOTION_SHARD_ENV = "MOTION_LIB_SHARD_PER_RANK"
_MOTION_SHARD_SIDECAR_ENV = "MOTION_LIB_SHARD_SIDECAR"

# Fields indexed by frame along dim 0 (concatenated across motions).
_SHARD_FRAME_FIELDS = (
    "gts",
    "grs",
    "gvs",
    "gavs",
    "dvs",
    "dps",
    "contacts",
    "lrs",
    "goal_states",
)
# Fields indexed by motion along dim 0.
_SHARD_MOTION_FIELDS = ("motion_lengths", "motion_dt", "motion_num_frames")


# --- Optional fp16 quantization of the big per-frame pose tensors -----------
#
# PM_MOTIONLIB_FP16=1 (opt-in, default off): store the static per-frame pose
# tensors (positions, rotations, velocities) as fp16 GPU-resident, roughly
# halving their VRAM footprint (~3 GiB on a 6 GiB pack like amass_good.pt).
# Deliberately excludes everything timing/indexing/cumulative: motion_dt,
# length_starts, motion_num_frames, motion_lengths, motion_weights, contacts,
# lrs, goal_states are never touched. Quaternions (grs) are renormalized
# immediately after the cast since fp16 rounding can nudge ||q|| off 1.0;
# get_motion_state_exact_frame upcasts back to fp32 on read so downstream
# interpolation/exp-map math is unaffected.
_MOTIONLIB_FP16_ENV = "PM_MOTIONLIB_FP16"
_FP16_QUANTIZE_FIELDS = ("gts", "grs", "gvs", "gavs", "dps", "dvs")
_FP16_QUAT_FIELDS = ("grs",)


def _motionlib_fp16_enabled() -> bool:
    return os.environ.get(_MOTIONLIB_FP16_ENV, "0") == "1"


def _quantize_motion_tensors_fp16(motion_lib) -> None:
    """In-place fp16 downcast of the big static per-frame pose tensors.

    Only touches fields in _FP16_QUANTIZE_FIELDS (positions/rotations/
    velocities); quaternion fields are renormalized post-cast. No-op unless
    PM_MOTIONLIB_FP16=1.
    """
    if not _motionlib_fp16_enabled():
        return
    freed_cuda = False
    for lib_field in _FP16_QUANTIZE_FIELDS:
        tensor = getattr(motion_lib, lib_field, None)
        if tensor is None or not torch.is_tensor(tensor):
            continue
        if tensor.dtype == torch.float16:
            continue  # already quantized (e.g. re-entrant load path)
        if tensor.is_cuda:
            freed_cuda = True
        tensor = tensor.to(torch.float16)
        if lib_field in _FP16_QUAT_FIELDS:
            norm = tensor.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
            tensor = (tensor.float() / norm).to(torch.float16)
        setattr(motion_lib, lib_field, tensor)
    # Return the freed fp32 source blocks to CUDA. setattr above drops the last
    # reference to each fp32 original, but the caching allocator keeps those
    # blocks RESERVED (not returned to the driver). On a large pack they linger
    # as ~13.9 GiB of reserved-but-unallocated, non-contiguous fragments that
    # cannot satisfy the later single large contiguous rollout-buffer allocation
    # (~12-16 GiB) -> spurious CUDA OOM at epoch 0 despite ample nominal free
    # VRAM, even though fp16 lowered the *allocated* motion footprint. Dropping
    # the cache here is what actually realizes the fp16 saving on-GPU and lets
    # PhysX (which allocates outside the torch allocator) reuse the space.
    if freed_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"[motionlib-fp16] quantized {_FP16_QUANTIZE_FIELDS} to fp16 "
        f"(quat fields {_FP16_QUAT_FIELDS} renormalized; cuda cache dropped={freed_cuda})"
    )


# --- Smoothed-contacts disk cache -------------------------------------------
#
# MotionLib.smooth_contacts() runs a per-motion conv1d over the (possibly
# per-rank) contacts tensor on every env construction. For large packs this
# costs several minutes at every startup. We cache the smoothed tensor to a
# sidecar next to the pack -- keyed by (pack size+mtime, smoothing window, and
# when sharded world_size+rank) -- and skip the recompute on a validated hit.
# The cache is best-effort: any error falls back to the normal recompute path,
# so it can never break loading. Disable with MOTION_LIB_SMOOTHED_CACHE_DISABLE=1.
_SMOOTHED_CACHE_DISABLE_ENV = "MOTION_LIB_SMOOTHED_CACHE_DISABLE"


def _smoothed_contacts_cache_disabled() -> bool:
    return os.environ.get(_SMOOTHED_CACHE_DISABLE_ENV, "0") not in (
        "0",
        "",
        "false",
        "False",
    )


def _raw_pack_path(motion_file):
    """Strip the '#shard{r}of{ws}' suffix a sharded load appends to motion_file."""
    if not motion_file:
        return None
    return str(motion_file).split("#shard")[0]


def _pack_signature(pack_path):
    st = os.stat(pack_path)
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _smoothed_contacts_sidecar_path(pack_path, window_size, world_size=None, rank=None):
    if world_size is not None and rank is not None:
        return (
            f"{pack_path}.smoothed_contacts_w{window_size}"
            f".ws{world_size}.rank{rank}.pt"
        )
    return f"{pack_path}.smoothed_contacts_w{window_size}.pt"


def _smooth_contacts_core(
    contacts, length_starts, motion_num_frames, num_motions, window_size, device
):
    """Per-motion moving-average smoothing that respects motion boundaries.

    Pure re-expression of the original smooth_contacts convolution loop; returns
    a new float32 tensor clamped to [0, 1] (does not mutate the input).
    """
    kernel = (
        torch.ones(1, 1, window_size, device=device, dtype=torch.float32)
        / window_size
    )
    padding = window_size // 2
    smoothed = torch.zeros_like(contacts, dtype=torch.float32)
    for motion_idx in range(num_motions):
        start_idx = int(length_starts[motion_idx].item())
        num_frames = int(motion_num_frames[motion_idx].item())
        end_idx = start_idx + num_frames
        motion_contacts = contacts[start_idx:end_idx].float()
        contacts_for_conv = motion_contacts.t().unsqueeze(1)
        padded_contacts = torch.nn.functional.pad(
            contacts_for_conv, (padding, padding), mode="replicate"
        )
        smoothed_motion = torch.nn.functional.conv1d(
            padded_contacts, kernel, padding=0
        )
        smoothed[start_idx:end_idx] = smoothed_motion.squeeze(1).t()
    return torch.clamp(smoothed, 0.0, 1.0)


def motion_lib_shard_enabled() -> bool:
    """True when per-rank motion-lib sharding is requested via env var."""
    return os.environ.get(_MOTION_SHARD_ENV, "0") == "1"


def load_shard_class_boundaries(sidecar_path, expected_num_motions):
    """Load per-class motion-index boundaries from a pack's .mix.json sidecar.

    The pack builder (wbc_push/scripts/training/motion_aug/build_track_d_train_pack.py)
    concatenates the base corpus followed by each augmentation class in the
    sidecar's ``mix`` insertion order, so cumulative ``kept`` counts give exact
    contiguous [start, end) index ranges per class.

    Returns:
        List of (class_name, start, end) tuples, or None if the sidecar is
        missing/unusable (callers fall back to plain interleave, no renorm).
    """
    if sidecar_path is None or not os.path.isfile(sidecar_path):
        return None
    try:
        with open(sidecar_path, "r") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        log.warning(f"[motion-shard] failed to read sidecar {sidecar_path}: {e}")
        return None
    mix = data.get("mix")
    if not mix:
        log.warning(f"[motion-shard] sidecar {sidecar_path} has no 'mix' section")
        return None
    boundaries = []
    start = 0
    for name, info in mix.items():
        kept = int(info["kept"])
        boundaries.append((name, start, start + kept))
        start += kept
    if start != expected_num_motions:
        log.warning(
            f"[motion-shard] sidecar {sidecar_path} class counts sum to {start} "
            f"but pack has {expected_num_motions} motions; ignoring sidecar"
        )
        return None
    sidecar_n = data.get("num_motions")
    if sidecar_n is not None and int(sidecar_n) != expected_num_motions:
        log.warning(
            f"[motion-shard] sidecar num_motions={sidecar_n} != pack "
            f"{expected_num_motions}; ignoring sidecar"
        )
        return None
    return boundaries


def compute_rank_shard(
    num_motions, motion_weights, rank, world_size, class_boundaries=None
):
    """Compute a rank's interleaved motion shard and renormalized weights.

    Pure function (no torch.distributed) so it is unit-testable on CPU.

    Shard split: ``local_indices = arange(rank, num_motions, world_size)``.
    Weight handling: with ``class_boundaries``, each class's local weights are
    scaled by (global_class_mass / local_class_mass) so every rank's per-class
    sampling mass fractions exactly equal the global pack's (torch.multinomial
    normalizes, so only relative mass matters). Without boundaries, weights are
    plain-sliced (approximate mass preservation only).

    Args:
        num_motions: Global motion count N.
        motion_weights: Global weight tensor [N].
        rank: This rank's index in [0, world_size).
        world_size: Number of ranks.
        class_boundaries: Optional list of (name, start, end) from
            load_shard_class_boundaries.

    Returns:
        (local_indices [n_local] LongTensor, local_weights [n_local] float32,
         report dict with per-class counts/mass fractions for dump-verification)
    """
    assert 0 <= rank < world_size, f"rank {rank} not in [0, {world_size})"
    assert num_motions >= world_size, (
        f"cannot shard {num_motions} motions across {world_size} ranks"
    )
    local_indices = torch.arange(rank, num_motions, world_size, dtype=torch.long)
    w_global = motion_weights.detach().to(dtype=torch.float64, device="cpu")
    assert w_global.shape[0] == num_motions, (
        f"weights len {w_global.shape[0]} != num_motions {num_motions}"
    )
    local_w = w_global[local_indices].clone()

    report = {
        "rank": rank,
        "world_size": world_size,
        "num_motions_global": int(num_motions),
        "num_motions_local": int(local_indices.numel()),
        "renormalized": class_boundaries is not None,
        "classes": [],
    }

    if class_boundaries is not None:
        global_total = float(w_global.sum())
        assert global_total > 0, "global sampling mass is zero"
        for name, start, end in class_boundaries:
            global_mass = float(w_global[start:end].sum())
            in_class = (local_indices >= start) & (local_indices < end)
            n_local = int(in_class.sum())
            assert n_local > 0, (
                f"[motion-shard] rank {rank}: class '{name}' has 0 motions in "
                f"shard (class size {end - start}, world_size {world_size})"
            )
            local_mass = float(local_w[in_class].sum())
            assert local_mass > 0, (
                f"[motion-shard] rank {rank}: class '{name}' has zero local "
                f"sampling mass before renorm"
            )
            local_w[in_class] *= global_mass / local_mass
            report["classes"].append(
                {
                    "name": name,
                    "global_count": end - start,
                    "local_count": n_local,
                    "global_mass_fraction": global_mass / global_total,
                }
            )
        # After renorm each class's local mass equals its global mass, so local
        # fractions match global fractions exactly (up to float64 rounding).
    return local_indices, local_w.to(torch.float32), report


# --- Runtime target-mass reweighting (env-gated via sidecar presence) --------
#
# When a "<pack>.mix_target.json" sidecar sits next to a packaged (.pt) motion
# library, its per-class target sampling-mass fractions REPLACE the pack's
# native (as-baked) per-class fractions, without touching pack bytes. This is
# applied to the GLOBAL weight tensor before any per-rank sharding — a
# per-rank (local-denominator) rescale would break the cross-rank per-class
# mass-equality invariant checked by _shard_launch_asserts' assert C, because
# each rank's local class composition (counts) differs from the global one.
_MIX_TARGET_SIDECAR_SUFFIX = ".mix_target.json"


def load_mix_target(sidecar_path, class_boundaries):
    """Load and validate a pack's "<pack>.mix_target.json" sidecar.

    Format: {"target_mass": {"<class_name>": <fraction>, ...}}. The class
    names must exactly match those in ``class_boundaries`` (from the pack's
    .mix.json), and the fractions must sum to 1.0 within 1e-6.

    Args:
        sidecar_path: Path to the "<pack>.mix_target.json" file.
        class_boundaries: List of (name, start, end) from
            load_shard_class_boundaries; defines the expected class names.

    Returns:
        Dict mapping class name -> target global mass fraction, or None if
        the sidecar file does not exist.

    Raises:
        ValueError: If the sidecar exists but is malformed (missing
            'target_mass' key, class-name mismatch, or fractions not
            summing to 1.0 within 1e-6). Errors loudly rather than silently
            falling back, since a silently-ignored target-mass request would
            train on the wrong mix.
    """
    if not os.path.isfile(sidecar_path):
        return None
    with open(sidecar_path, "r") as f:
        data = json.load(f)
    target = data.get("target_mass")
    if not target:
        raise ValueError(
            f"[mix-target] {sidecar_path} is missing a non-empty 'target_mass' key"
        )
    boundary_names = {name for name, _, _ in class_boundaries}
    target_names = set(target.keys())
    if boundary_names != target_names:
        raise ValueError(
            f"[mix-target] {sidecar_path} class names {sorted(target_names)} "
            f"do not match pack classes {sorted(boundary_names)} "
            "(from the pack's .mix.json)"
        )
    target_fractions = {name: float(frac) for name, frac in target.items()}
    total = sum(target_fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"[mix-target] {sidecar_path} 'target_mass' fractions sum to "
            f"{total!r}, expected 1.0 +/- 1e-6"
        )
    return target_fractions


def rescale_to_target_mass(motion_weights, class_boundaries, target_fractions):
    """Rescale a global weight tensor so each class's global mass fraction
    equals its target fraction, via a pure per-class scalar multiplier.

    Pure function (no I/O, no torch.distributed) so it is unit-testable and
    safe to call before any per-rank sharding.

    Derivation: let ``total`` be the pre-rescale sum of all weights, and
    ``mass_i`` the pre-rescale sum of class i's weights. Scaling class i's
    weights by ``multiplier_i = target_i * total / mass_i`` makes its
    post-rescale mass exactly ``target_i * total``. Summing over classes,
    the post-rescale total is ``total * sum(target_i) == total`` (since
    target fractions sum to 1.0), so the post-rescale total is UNCHANGED and
    each class's post-rescale global fraction is exactly ``target_i``.

    Args:
        motion_weights: Global weight tensor [N] (any float dtype/device).
        class_boundaries: List of (name, start, end) from
            load_shard_class_boundaries.
        target_fractions: Dict name -> target global mass fraction (from
            load_mix_target); must cover every class in class_boundaries.

    Returns:
        Rescaled weight tensor, same shape and dtype as ``motion_weights``.
    """
    orig_dtype = motion_weights.dtype
    w = motion_weights.detach().to(dtype=torch.float64, device="cpu")
    total = float(w.sum())
    assert total > 0, "[mix-target] global sampling mass is zero, cannot rescale"
    out = w.clone()
    for name, start, end in class_boundaries:
        assert name in target_fractions, (
            f"[mix-target] class '{name}' has no target fraction (should have "
            "been caught by load_mix_target validation)"
        )
        mass = float(w[start:end].sum())
        assert mass > 0, (
            f"[mix-target] class '{name}' has zero pre-rescale mass; cannot "
            "rescale a zero-mass class to a nonzero target"
        )
        multiplier = target_fractions[name] * total / mass
        out[start:end] = w[start:end] * multiplier
    return out.to(dtype=orig_dtype, device=motion_weights.device)


# Mapping from MotionLib (packaged motion) field names to RobotState (single motion/sim state) field names
_motion_field_mapping = {
    "gts": "rigid_body_pos",
    "grs": "rigid_body_rot",
    "gavs": "rigid_body_ang_vel",
    "gvs": "rigid_body_vel",
    "dvs": "dof_vel",
    "dps": "dof_pos",
}


@dataclass
class MotionLibConfig:
    """Configuration for motion library."""

    _target_: str = "protomotions.components.motion_lib.MotionLib"
    motion_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to motion file (.pt, .yaml, or .motion). None for empty library."
        },
    )
    motion_subset: Optional[str] = field(
        default=None,
        metadata={
            "help": "Load only a subset of the motions in a packaged .pt, so a large "
            "pack can be reviewed on a small GPU. Either a half-open index range "
            "'start:end' (e.g. '0:1000', '1000:2000') or an explicit comma-separated "
            "list (e.g. '5,10,15'). The slice happens on CPU via a memory-mapped read, "
            "so only the selected motions are ever moved to the device. None loads all."
        },
    )
    get_motion_state_use_blend: bool = field(
        default=True,
        metadata={
            "help": "Use interpolation for smooth motion queries between frames."
        },
    )
    max_seconds: Optional[float] = field(
        default=None,
        metadata={
            "help": "Clip motions longer than this to a random duration in "
            "[max_seconds - clip_delta, max_seconds]. None disables clipping."
        },
    )
    clip_delta: float = field(
        default=3.0,
        metadata={
            "help": "Maximum random reduction (seconds) below max_seconds when clipping. "
            "Clip target = uniform(max_seconds - clip_delta, max_seconds)."
        },
    )
    clip_seed: int = field(
        default=42,
        metadata={"help": "RNG seed for reproducible per-motion clip lengths."},
    )


class MotionLib:
    """Motion library for managing and sampling reference motion data.

    Stores and manages a collection of motion clips for use in imitation learning.
    Supports efficient sampling, interpolation, and batched access to motion data.
    The library can load from individual .motion files, YAML descriptors, or pre-packaged .pt files.

    **Motion Data Stored:**

    - Rigid body positions, rotations, velocities, angular velocities
    - DOF positions and velocities
    - Contact information
    - Motion metadata (lengths, FPS, weights)

    **Example:**

    .. code-block:: python

        config = MotionLibConfig(motion_file="data/motions/walk.pt")
        motion_lib = MotionLib(config, device="cuda")
        motion_ids = motion_lib.sample_motions(num_samples=1024)
        state = motion_lib.get_motion_state(motion_ids, motion_times)
    """

    # List all the tensor fields that need to be saved/loaded
    gts: torch.Tensor
    grs: torch.Tensor
    gvs: torch.Tensor
    gavs: torch.Tensor
    dvs: torch.Tensor
    dps: torch.Tensor
    length_starts: torch.Tensor
    motion_lengths: torch.Tensor
    motion_dt: torch.Tensor
    motion_num_frames: torch.Tensor
    motion_weights: torch.Tensor
    contacts: torch.Tensor

    motion_files: Tuple[str]

    # Optional fields
    lrs: Optional[torch.Tensor] = (
        None  # maybe also has local_rigid_body_rot for interpolation, see hack below
    )
    goal_states: Optional[torch.Tensor] = None  # per-frame binary mask for goal poses

    # Get all field names defined at class level
    _fields = list(__annotations__.keys())

    def __init__(
        self,
        config: "MotionLibConfig",
        device: str = "cpu",
    ):
        """Initialize MotionLib from config.

        Creates either a populated motion library (if config.motion_file is set) or
        an empty motion library (if config.motion_file is None) following Null Object pattern.

        Args:
            config: MotionLibConfig (always required, motion_file can be None for empty)
            device: PyTorch device
        """
        super().__init__()

        self.config = config
        self.device = device

        # Handle empty motion library (Null Object pattern)
        if config.motion_file is None:
            print("Creating empty MotionLib (no motion data)")
            self._create_empty()
            return

        self.get_motion_state_use_blend = config.get_motion_state_use_blend
        self.different_motion_files_across_ranks = False
        self.sharded_across_ranks = False
        self.global_motion_ids = None
        self.shard_rank = None
        self.shard_world_size = None

        motion_file = config.motion_file

        if str(motion_file).split(".")[-1] == "pt":
            print("Loading motions from packaged file which is faster")
            motion_file = self.process_packaged_motion_file_name_multi_gpu(motion_file)
            if motion_lib_shard_enabled():
                self.load_from_file_sharded(motion_file)
            else:
                self.load_from_file(motion_file)
        else:
            assert not motion_lib_shard_enabled(), (
                f"{_MOTION_SHARD_ENV}=1 requires a packaged .pt motion file, "
                f"got: {motion_file}"
            )
            print(
                "Loading motions from yaml/npy file or Directory of motions which is slower"
            )
            self._load_motions(motion_file)

        self.motion_file = motion_file
        if self.sharded_across_ranks:
            # Per-rank suffix makes env-checkpoint task ids unique per rank and
            # lets MotionManager.load_state_dict's name-equality guard reject
            # stale (unsharded / differently-sharded) motion_weights on resume.
            self.motion_file = (
                f"{motion_file}#shard{self.shard_rank}of{self.shard_world_size}"
            )

    def _create_empty(self):
        """Create an empty motion library with no motions."""
        self.get_motion_state_use_blend = False
        self.different_motion_files_across_ranks = False
        self.sharded_across_ranks = False
        self.global_motion_ids = None
        self.shard_rank = None
        self.shard_world_size = None
        self.motion_file = None

        # Create empty tensors
        self.gts = torch.empty(0, 0, 3, device=self.device)
        self.grs = torch.empty(0, 0, 4, device=self.device)
        self.gvs = torch.empty(0, 0, 3, device=self.device)
        self.gavs = torch.empty(0, 0, 3, device=self.device)
        self.dvs = torch.empty(0, 0, device=self.device)
        self.dps = torch.empty(0, 0, device=self.device)
        self.length_starts = torch.empty(0, dtype=torch.long, device=self.device)
        self.motion_lengths = torch.empty(0, device=self.device)
        self.motion_dt = torch.empty(0, device=self.device)
        self.motion_num_frames = torch.empty(0, dtype=torch.long, device=self.device)
        self.motion_weights = torch.empty(0, device=self.device)
        self.contacts = torch.empty(0, 0, device=self.device)
        self.motion_files = ()
        self.lrs = None
        self.goal_states = None

    @classmethod
    def empty(cls, device: str = "cpu"):
        """Create an empty MotionLib with no motion data.

        Factory method for creating empty motion libraries in a concise way.

        Args:
            device: PyTorch device

        Returns:
            Empty MotionLib instance
        """
        return cls(config=MotionLibConfig(motion_file=None), device=device)

    def num_motions(self):
        """Returns the number of motions in the state.

        Returns:
            int: The number of motions.
        """
        return len(self.motion_lengths)

    def get_total_length(self):
        """Returns the total length of all motions.

        Returns:
            int: The total length of all motions.
        """
        return sum(self.motion_lengths)

    def get_motion_length(self, motion_ids):
        """Returns the length of the specified motion(s).

        Args:
            motion_ids: The IDs of the motions to get the length of.

        Returns:
            Tensor: The length of the specified motion(s).
            If motion_ids is None, returns the length of all motions.
        """

        if motion_ids is None:
            return self.motion_lengths
        else:
            return self.motion_lengths[motion_ids]

    def get_motion_num_frames(self, motion_ids):
        """Returns the number of frames of the specified motion(s).

        Args:
            motion_ids: The IDs of the motions to get the number of frames of.

        Returns:
            Tensor: The number of frames of the specified motion(s).
            If motion_ids is None, returns the number of frames of all motions.
        """

        if motion_ids is None:
            return self.motion_num_frames
        else:
            return self.motion_num_frames[motion_ids]

    def process_packaged_motion_file_name_multi_gpu(self, motion_file):
        if "slurmrank" not in motion_file:
            return motion_file

        assert torch.distributed.is_initialized(), (
            "slurmrank motion files require distributed training "
            "(torch.distributed must be initialized)"
        )
        rank = torch.distributed.get_rank()

        # Discover matching files: replace "slurmrank" with a regex wildcard
        # e.g. "chunk_slurmrank.pt" -> "chunk_.*.pt" matches chunk_00.pt, chunk_1.pt, etc.
        folder = Path(motion_file).parent
        pattern = re.compile(
            "^" + re.escape(Path(motion_file).name).replace("slurmrank", "(.+)") + "$"
        )
        matches = sorted(
            (f.name for f in folder.iterdir() if pattern.match(f.name)),
            key=lambda name: int(pattern.match(name).group(1)),
        )
        assert matches, (
            f"No files matching slurmrank pattern in {folder}. "
            f"Expected files like: {Path(motion_file).name.replace('slurmrank', '*')}"
        )

        selected = matches[rank % len(matches)]
        motion_file = str(folder / selected)

        self.different_motion_files_across_ranks = True
        print(
            f"Rank {rank} loading motion file: {selected} "
            f"({len(matches)} files found, rank % {len(matches)} = {rank % len(matches)})"
        )

        return motion_file

    def _calc_frame_blend_from_id_and_time(self, motion_ids, motion_times):
        motion_len = self.motion_lengths[motion_ids]
        motion_times = motion_times.clip(min=0).clip(
            max=motion_len
        )  # Making sure time is in bounds
        num_frames = self.motion_num_frames[motion_ids]
        dt = self.motion_dt[motion_ids]

        return calc_frame_blend(motion_times, motion_len, num_frames, dt)

    def _calc_closest_frame(self, motion_ids, motion_times):
        motion_len = self.motion_lengths[motion_ids]
        motion_times = motion_times.clip(min=0).clip(
            max=motion_len
        )  # Making sure time is in bounds
        num_frames = self.motion_num_frames[motion_ids]
        frame_idx = torch.round(motion_times / motion_len * (num_frames - 1)).long()
        return frame_idx

    def has_goal_states(self) -> bool:
        """Check if this motion library has goal state annotations."""
        return self.goal_states is not None

    def get_goal_state_times(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Return the time of the goal-state frame for each motion.

        The goal state is the middle frame of the static interaction segment,
        identified during pre-processing and stored as a per-frame binary mask.

        Args:
            motion_ids: Tensor of motion IDs.

        Returns:
            Tensor of goal-state times. Returns -1 for motions without goal states.

        Raises:
            RuntimeError: If goal_states is not loaded (caller should check
            has_goal_states() first).
        """
        if self.goal_states is None:
            raise RuntimeError(
                "goal_states not available in this motion library. "
                "Run scripts/samp_detect_goal_states.py to annotate."
            )

        goal_times = torch.full(
            (len(motion_ids),), -1.0, device=self.device, dtype=torch.float32
        )
        for i, mid in enumerate(motion_ids):
            start = self.length_starts[mid]
            n_frames = self.motion_num_frames[mid]
            mask = self.goal_states[start : start + n_frames]
            goal_indices = torch.nonzero(mask, as_tuple=False)
            if len(goal_indices) > 0:
                median_idx = goal_indices[len(goal_indices) // 2].item()
                goal_times[i] = (
                    median_idx / max(n_frames.item() - 1, 1)
                ) * self.motion_lengths[mid]
        return goal_times

    def get_goal_state_times_batched(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Vectorized version of get_goal_state_times using precomputed cache.

        On first call, precomputes goal_state_time for every motion and caches it.
        Subsequent calls are a simple index lookup.

        Args:
            motion_ids: Tensor of motion IDs.

        Returns:
            Tensor of goal-state times. -1 for motions without goal states.
        """
        if not hasattr(self, "_goal_state_time_cache"):
            all_ids = torch.arange(self.num_motions(), device=self.device)
            self._goal_state_time_cache = self.get_goal_state_times(all_ids)
        return self._goal_state_time_cache[motion_ids]

    def get_motion_state(
        self, motion_ids, motion_times, joint_3d_format="exp_map", mirror_flags=None
    ) -> RobotState:
        """Sample interpolated reference state.

        mirror_flags: optional bool tensor aligned 1:1 with ``motion_ids`` rows.
        When provided (and mirror maps are attached via ``set_mirror_maps``), the
        selected rows are sagittal-mirrored in place AFTER blending -- a rigid
        reflection is linear, so mirror-after-blend == blend-of-mirrors, and
        doing it once post-blend is cheaper. ``None`` (the default) is a total
        no-op, so ``PM_MIRROR_PROB=0`` stays byte-identical.
        """
        frame_idx0, frame_idx1, blend = self._calc_frame_blend_from_id_and_time(
            motion_ids, motion_times
        )

        motion_state_0: RobotState = self.get_motion_state_exact_frame(
            motion_ids, frame_idx0
        )

        motion_state_1: RobotState = self.get_motion_state_exact_frame(
            motion_ids, frame_idx1
        )

        pos_keys = [
            "rigid_body_pos",
            "rigid_body_vel",
            "rigid_body_ang_vel",
            "dof_vel",
        ]

        rot_keys = ["rigid_body_rot"]

        for key in pos_keys:
            motion_state_0[key] = interpolate_pos(
                motion_state_0[key], motion_state_1[key], blend
            )

        for key in rot_keys:
            motion_state_0[key] = interpolate_quat(
                motion_state_0[key], motion_state_1[key], blend
            )

        # TODO: HACK: assume when local_rigid_body_rot is not None, all joints are exp_map
        # will use local_rigid_body_rot for interpolation
        if motion_state_0.local_rigid_body_rot is not None:
            # lr: (num_envs, num_bodies, 4)
            lr = interpolate_quat(
                motion_state_0.local_rigid_body_rot,
                motion_state_1.local_rigid_body_rot,
                blend,
            )
            b, j, _ = lr.shape
            lr = lr[:, 1:, :].reshape(
                -1, 4
            )  # (num_envs * num_bodies - 1, 4), excluding root
            assert (
                motion_state_0.dof_pos.shape[1] == (j - 1) * 3
            ), "dof_pos shape mismatch"
            motion_state_0.dof_pos = quat_to_exp_map(lr, w_last=True).reshape(
                b, (j - 1) * 3
            )
        else:
            motion_state_0.dof_pos = interpolate_pos(
                motion_state_0.dof_pos, motion_state_1.dof_pos, blend
            )

        # Blend contacts: use OR for boolean, average for float (smoothed contacts)
        if motion_state_0.rigid_body_contacts is not None:
            if motion_state_0.rigid_body_contacts.dtype == torch.bool:
                motion_state_0.rigid_body_contacts = (
                    motion_state_0.rigid_body_contacts
                    | motion_state_1.rigid_body_contacts
                )
            else:
                # For smoothed (float) contacts, take the average between frames
                motion_state_0.rigid_body_contacts = (
                    motion_state_0.rigid_body_contacts
                    + motion_state_1.rigid_body_contacts
                ) / 2.0

        # Online sagittal mirror augmentation (ref-side). No-op unless mirror
        # maps are attached AND a per-row flag tensor is supplied.
        if mirror_flags is not None and self._mirror_maps is not None:
            from protomotions.components.motion_mirror import mirror_robot_state

            mirror_robot_state(motion_state_0, mirror_flags, self._mirror_maps)

        return motion_state_0

    # ---- Online mirror augmentation plumbing --------------------------------
    _mirror_maps = None

    def set_mirror_maps(self, mirror_maps) -> None:
        """Attach precomputed sagittal-mirror maps (built once, on device).

        Set by the env after robot_config is available. When unset,
        ``get_motion_state(..., mirror_flags=...)`` is a no-op regardless of the
        flags, so the default path is byte-identical.
        """
        self._mirror_maps = mirror_maps

    def get_motion_state_exact_frame(
        self,
        motion_ids,
        frame_indices,
    ) -> RobotState:
        """
        Retrieves motion states at exact frame indices without any blending.

        Args:
            motion_ids: Tensor of motion IDs to sample from
            frame_indices: Tensor of integer frame indices

        Returns:
            RobotState: The robot state at the specified frames
        """

        # Get global indices by adding offsets
        fl = frame_indices + self.length_starts[motion_ids]

        # Create a dict with keys from motion_field_mapping values
        motion_data = {}
        for lib_field, motion_attr in _motion_field_mapping.items():
            field_data = getattr(self, lib_field)
            if field_data is not None:
                sampled = field_data[fl].clone()
                if sampled.dtype == torch.float16:
                    # PM_MOTIONLIB_FP16 storage: upcast to fp32 for downstream
                    # interpolation/exp-map math (never keep fp16 past sampling).
                    sampled = sampled.float()
                motion_data[motion_attr] = sampled

        if self.lrs is not None:
            local_rigid_body_rot = self.lrs[fl].clone()
        else:
            local_rigid_body_rot = None

        # Create and return the motion state
        motion_state = RobotState.from_dict(
            motion_data, state_conversion=StateConversion.COMMON
        )
        motion_state.local_rigid_body_rot = local_rigid_body_rot
        motion_state.rigid_body_contacts = (
            self.contacts[fl].clone() if self.contacts is not None else None
        )

        return motion_state

    def _load_motions(self, motion_file):
        import random

        motions = []
        motion_lengths = []
        motion_dt = []
        motion_num_frames = []

        motion_files, motion_weights = self._fetch_motion_files(motion_file)

        num_motion_files = len(motion_files)

        clip_rng = None
        if self.config.max_seconds is not None:
            clip_rng = random.Random(self.config.clip_seed)

        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print(curr_file)
            print(
                "Loading {:d}/{:d} motion files: {:s}".format(
                    f + 1, num_motion_files, curr_file
                )
            )

            curr_motion = torch.load(curr_file, weights_only=False)
            curr_motion = RobotState.from_dict(
                curr_motion, state_conversion=StateConversion.COMMON
            )

            if (
                clip_rng is not None
                and curr_motion.motion_length > self.config.max_seconds
            ):
                target = clip_rng.uniform(
                    self.config.max_seconds - self.config.clip_delta,
                    self.config.max_seconds,
                )
                max_frames = min(
                    int(target * curr_motion.fps) + 1, curr_motion.motion_num_frames
                )
                curr_motion = curr_motion[:max_frames]
                print(
                    f"    Clipped to {curr_motion.motion_length:.2f}s ({max_frames} frames)"
                )

            motions.append(curr_motion)
            motion_lengths.append(curr_motion.motion_length)
            motion_dt.append(curr_motion.motion_dt)
            motion_num_frames.append(curr_motion.motion_num_frames)

        # Process the motions using the field mapping
        for lib_field, motion_attr in _motion_field_mapping.items():
            tp = (
                torch.bool
                if getattr(motions[0], motion_attr).dtype == torch.bool
                else torch.float32
            )
            setattr(
                self,
                lib_field,
                torch.cat([getattr(m, motion_attr) for m in motions], dim=0).to(
                    dtype=tp, device=self.device
                ),
            )

        # Optionally pack contacts if present in motion data
        if motions[0].rigid_body_contacts is not None:
            tp = (
                torch.bool
                if motions[0].rigid_body_contacts.dtype == torch.bool
                else torch.float32
            )
            self.contacts = torch.cat(
                [m.rigid_body_contacts for m in motions], dim=0
            ).to(dtype=tp, device=self.device)
        else:
            self.contacts = None

        # If all contact labels are zero, discard them so downstream consumers
        # fail loudly instead of silently training on meaningless data.
        if (
            self.contacts is not None
            and self.contacts.numel() > 0
            and not self.contacts.any()
        ):
            log.warning(
                "All contact labels in motion library are zero. "
                "Discarding contacts — any reward/component that reads ref contact labels "
                "will raise an error. Re-run motion conversion with contact detection enabled, "
                "or remove contact-based rewards from the experiment config."
            )
            self.contacts = None

        # optionally pack local_rigid_body_rot if exists
        if motions[0].local_rigid_body_rot is not None:
            self.lrs = torch.cat(
                [getattr(m, "local_rigid_body_rot") for m in motions], dim=0
            ).to(dtype=torch.float32, device=self.device)

        # Handle other fields that don't come directly from the motion objects
        self.motion_num_frames = torch.tensor(
            motion_num_frames, dtype=torch.long, device=self.device
        )
        lengths_shifted = self.motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        self.motion_weights = torch.tensor(
            motion_weights, dtype=torch.float32, device=self.device
        )

        self.motion_lengths = torch.tensor(
            motion_lengths, dtype=torch.float32, device=self.device
        )
        self.motion_dt = torch.tensor(
            motion_dt, dtype=torch.float32, device=self.device
        )

        self.motion_files = tuple(motion_files)  # for saving to packed pt file

        _quantize_motion_tensors_fp16(self)

        num_motions = len(motions)
        total_len = sum(motion_lengths)
        print(
            "Loaded {:d} motions with a total length of {:.3f}s.".format(
                num_motions, total_len
            )
        )

        return

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if ext == ".yaml":
            dir_name = os.path.dirname(motion_file)

            motion_files = []
            motion_weights = []

            with open(os.path.join(os.getcwd(), motion_file), "r") as f:
                motion_config = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

            for motion_entry in motion_config.motions:
                curr_file = motion_entry.file
                curr_file = os.path.join(dir_name, curr_file)
                motion_files.append(curr_file)
                motion_weights.append(motion_entry.get("weight", 1.0))

        elif ext == ".npz" or ext == ".motion":
            motion_files = [motion_file]
            motion_weights = [1.0]
        else:
            # this should be a directory of motions
            motion_path = Path(motion_file)
            assert (
                motion_path.is_dir()
            ), "Motion file must be yaml, npz, motion, or a directory"

            motion_files = [str(path) for path in motion_path.rglob("*.motion")]
            assert len(motion_files) > 0, "No motion files found in directory"
            motion_weights = [1.0] * len(motion_files)

        return (
            motion_files,
            motion_weights,
        )

    def save_to_file(self, file_path):
        """
        Save the motion library to a packaged file (.pt).

        Args:
            file_path: Path to save the motion library
        """

        assert str(file_path).split(".")[-1] == "pt", "Name much ends with .pt"

        file_path = Path(file_path)

        # Create a dictionary with all required tensors
        save_data = {}

        for field_name in self._fields:
            if getattr(self, field_name) is not None:
                save_data[field_name] = getattr(self, field_name)

        # Ensure directory exists
        os.makedirs(file_path.parent, exist_ok=True)

        # Save to file
        torch.save(save_data, file_path)
        print(f"Motion library saved to {file_path}")

    def load_from_file(self, file_path):
        """
        Load the motion library from a packaged file (.pt).

        Args:
            file_path: Path to the motion library file
        """
        print(f"Loading motion library from {file_path}")
        if self.config.motion_subset:
            loaded_data = self._load_subset_from_file(
                file_path, self.config.motion_subset
            )
        else:
            loaded_data = torch.load(
                file_path, map_location=self.device, weights_only=False
            )

        # Runtime target-mass reweight (unsharded-load parity with
        # load_from_file_sharded): if a "<pack>.mix_target.json" sidecar is
        # present, rescale motion_weights so per-class global mass fractions
        # match the target. Requires the pack's "<pack>.mix.json" class
        # boundaries; if that sidecar is absent, log and skip (no mix.json
        # means we cannot know which weight indices belong to which class).
        mix_target_path = str(file_path) + _MIX_TARGET_SIDECAR_SUFFIX
        if os.path.isfile(mix_target_path) and "motion_weights" in loaded_data:
            mix_sidecar_path = str(file_path) + ".mix.json"
            num_motions = int(loaded_data["motion_weights"].shape[0])
            class_boundaries = load_shard_class_boundaries(
                mix_sidecar_path, num_motions
            )
            if class_boundaries is None:
                log.warning(
                    f"[motion-mix-target] found {mix_target_path} but no "
                    f"usable class boundaries (missing/invalid "
                    f"{mix_sidecar_path}); skipping target-mass rescale"
                )
            else:
                target_fractions = load_mix_target(mix_target_path, class_boundaries)
                loaded_data["motion_weights"] = rescale_to_target_mass(
                    loaded_data["motion_weights"], class_boundaries, target_fractions
                ).to(self.device)
                print(
                    f"[motion-mix-target] applied target mass sidecar "
                    f"{mix_target_path}: {target_fractions}"
                )

        # Pre-initialize all fields to None so missing fields (e.g. contacts
        # discarded at save time) don't cause AttributeError below.
        for field in self._fields:
            if not hasattr(self, field):
                setattr(self, field, None)

        for field in loaded_data:
            setattr(self, field, loaded_data[field])

        _quantize_motion_tensors_fp16(self)

        if (
            self.contacts is not None
            and self.contacts.numel() > 0
            and not self.contacts.any()
        ):
            log.warning(
                "All contact labels in packaged motion library are zero. "
                "Discarding contacts — any component reading ref contacts will error."
            )
            self.contacts = None

    def load_from_file_sharded(self, file_path):
        """Load only this rank's interleaved shard of a packaged (.pt) library.

        Env-gated via MOTION_LIB_SHARD_PER_RANK=1. Loads the pack on CPU
        (mmap'd when possible so only the shard's pages are ever read), slices
        motions[rank::world_size], renormalizes sampling weights per class
        (sidecar: <pack>.mix.json, override via MOTION_LIB_SHARD_SIDECAR), and
        only then moves tensors to self.device. Per-rank GPU (and CPU) residency
        is therefore ~pack_size / world_size, which is what allows packs larger
        than a single rank's VRAM budget.

        After this call, motion ids everywhere on this rank are SHARD-LOCAL.
        self.global_motion_ids maps local id -> global pack id.
        """
        assert (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ), (
            f"{_MOTION_SHARD_ENV}=1 requires torch.distributed to be initialized"
        )
        assert not self.different_motion_files_across_ranks, (
            f"{_MOTION_SHARD_ENV}=1 cannot be combined with 'slurmrank' "
            "per-rank motion files"
        )
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            log.warning(
                f"[motion-shard] {_MOTION_SHARD_ENV}=1 with world_size=1; "
                "loading full library (no sharding)"
            )
            self.load_from_file(file_path)
            return

        print(
            f"[motion-shard][rank {rank}] loading shard {rank}/{world_size} "
            f"of {file_path}"
        )
        try:
            loaded_data = torch.load(
                file_path, map_location="cpu", weights_only=False, mmap=True
            )
        except Exception as e:  # mmap unsupported for some save formats
            log.warning(
                f"[motion-shard] mmap load failed ({e}); falling back to plain "
                "CPU load (transient full-pack CPU RAM per rank)"
            )
            loaded_data = torch.load(file_path, map_location="cpu", weights_only=False)

        num_motions_global = int(loaded_data["motion_num_frames"].shape[0])

        sidecar_path = os.environ.get(_MOTION_SHARD_SIDECAR_ENV) or (
            str(file_path) + ".mix.json"
        )
        class_boundaries = load_shard_class_boundaries(
            sidecar_path, num_motions_global
        )
        if class_boundaries is None:
            log.warning(
                f"[motion-shard][rank {rank}] no usable sidecar at {sidecar_path}: "
                "plain interleave WITHOUT per-class mass renormalization "
                "(global class sampling mass only approximately preserved)"
            )

        # Runtime target-mass reweight, applied to the GLOBAL weight tensor
        # BEFORE compute_rank_shard so the per-rank renorm (above) targets the
        # already-rescaled class masses. Must happen here, not post-shard.
        mix_target_path = str(file_path) + _MIX_TARGET_SIDECAR_SUFFIX
        if os.path.isfile(mix_target_path):
            if class_boundaries is None:
                log.warning(
                    f"[motion-mix-target][rank {rank}] found {mix_target_path} "
                    f"but no usable class boundaries (missing/invalid "
                    f"{sidecar_path}); skipping target-mass rescale"
                )
            else:
                target_fractions = load_mix_target(mix_target_path, class_boundaries)
                loaded_data["motion_weights"] = rescale_to_target_mass(
                    loaded_data["motion_weights"], class_boundaries, target_fractions
                )
                print(
                    f"[motion-mix-target][rank {rank}] applied target mass "
                    f"sidecar {mix_target_path}: {target_fractions}"
                )

        local_indices, local_weights, report = compute_rank_shard(
            num_motions_global,
            loaded_data["motion_weights"],
            rank,
            world_size,
            class_boundaries,
        )

        # Frame gather index for this shard (global frame indices).
        global_num_frames = loaded_data["motion_num_frames"].to(torch.long)
        global_starts = loaded_data["length_starts"].to(torch.long)
        frame_idx = torch.cat(
            [
                torch.arange(global_starts[m], global_starts[m] + global_num_frames[m])
                for m in local_indices.tolist()
            ]
        )

        # Pre-initialize all fields to None (mirrors load_from_file).
        for field_name in self._fields:
            if not hasattr(self, field_name):
                setattr(self, field_name, None)

        for field_name in self._fields:
            if field_name not in loaded_data:
                continue
            value = loaded_data[field_name]
            if field_name in _SHARD_FRAME_FIELDS:
                setattr(
                    self, field_name, value.index_select(0, frame_idx).to(self.device)
                )
            elif field_name in _SHARD_MOTION_FIELDS:
                setattr(
                    self,
                    field_name,
                    value.index_select(0, local_indices).to(self.device),
                )
            elif field_name == "motion_weights":
                self.motion_weights = local_weights.to(self.device)
            elif field_name == "motion_files":
                self.motion_files = tuple(value[m] for m in local_indices.tolist())
            elif field_name == "length_starts":
                pass  # recomputed below for the local shard
            else:
                setattr(self, field_name, value)

        # Recompute local length_starts from local frame counts.
        lengths_shifted = self.motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        del loaded_data, frame_idx

        _quantize_motion_tensors_fp16(self)

        if (
            self.contacts is not None
            and self.contacts.numel() > 0
            and not self.contacts.any()
        ):
            log.warning(
                "All contact labels in packaged motion library are zero. "
                "Discarding contacts — any component reading ref contacts will error."
            )
            self.contacts = None

        self.sharded_across_ranks = True
        self.shard_rank = rank
        self.shard_world_size = world_size
        self.global_motion_ids = local_indices.to(self.device)

        self._shard_launch_asserts(num_motions_global, class_boundaries, report)

    def _shard_launch_asserts(self, num_motions_global, class_boundaries, report):
        """Dump-verify the shard at init: log everything, hard-assert invariants.

        A. sum of per-rank motion counts across ranks == global N (all_reduce).
        B. every sidecar class present with >0 mass on this rank (checked in
           compute_rank_shard).
        C. per-class post-renorm mass fractions identical across ranks and equal
           to the global fractions (all_gather, tol 1e-5). "Global fractions"
           here means whatever `report["classes"][i]["global_mass_fraction"]`
           says — computed by compute_rank_shard from the weight tensor it was
           given. When a mix_target sidecar is active, that tensor was already
           rescaled to the target fractions before compute_rank_shard ran, so
           this assert transparently checks against the TARGET fractions in
           that case (no separate expected-value branch needed).
        D. local frame tensors consistent with motion_num_frames/length_starts.
        """
        rank, world_size = self.shard_rank, self.shard_world_size
        n_local = self.num_motions()

        # D: frame bookkeeping.
        frames_expected = int(self.motion_num_frames.sum().item())
        assert self.gts.shape[0] == frames_expected, (
            f"[motion-shard][rank {rank}] frame slice mismatch: gts has "
            f"{self.gts.shape[0]} frames, motion_num_frames sums to {frames_expected}"
        )
        assert (
            int(self.length_starts[-1].item())
            + int(self.motion_num_frames[-1].item())
            == frames_expected
        ), f"[motion-shard][rank {rank}] length_starts inconsistent"
        assert len(self.motion_files) == n_local and self.motion_weights.shape[0] == (
            n_local
        ), f"[motion-shard][rank {rank}] motion-level field length mismatch"

        # Collectives: use a CUDA tensor for NCCL, CPU otherwise.
        backend = torch.distributed.get_backend()
        coll_device = self.device if str(backend) == "nccl" else "cpu"

        # A: global motion-count conservation.
        count = torch.tensor([n_local], dtype=torch.long, device=coll_device)
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
        assert int(count.item()) == num_motions_global, (
            f"[motion-shard][rank {rank}] shards do not partition the pack: "
            f"sum of per-rank counts {int(count.item())} != {num_motions_global}"
        )

        # C: global per-class sampling-mass check.
        frac_str = "n/a (no sidecar)"
        if class_boundaries is not None:
            w = self.motion_weights.to(dtype=torch.float64, device=coll_device)
            gids = self.global_motion_ids.to(coll_device)
            total = w.sum()
            local_fracs = torch.stack(
                [
                    w[(gids >= start) & (gids < end)].sum() / total
                    for _, start, end in class_boundaries
                ]
            )
            gathered = [torch.zeros_like(local_fracs) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, local_fracs)
            global_fracs = torch.tensor(
                [c["global_mass_fraction"] for c in report["classes"]],
                dtype=torch.float64,
                device=coll_device,
            )
            max_dev = max(
                float((g - global_fracs).abs().max().item()) for g in gathered
            )
            assert max_dev < 1e-5, (
                f"[motion-shard][rank {rank}] per-class sampling-mass drift "
                f"{max_dev:.3e} >= 1e-5 across ranks"
            )
            frac_str = ", ".join(
                f"{c['name']}={f:.4f}(n={c['local_count']}/{c['global_count']})"
                for c, f in zip(
                    report["classes"], local_fracs.cpu().tolist()
                )
            ) + f" | max cross-rank mass dev {max_dev:.2e}"

        print(
            f"[motion-shard][rank {rank}] OK: {n_local} local motions of "
            f"{num_motions_global} global (world_size {world_size}, "
            f"renormalized={report['renormalized']}), "
            f"{frames_expected} local frames | class mass: {frac_str}"
        )
    @staticmethod
    def parse_motion_subset(subset: str, num_motions: int) -> list:
        """Resolve a motion_subset spec against a pack of `num_motions` motions.

        Accepts a half-open range 'start:end' or an explicit list '5,10,15'.
        Raises on an empty range or an out-of-bounds index rather than silently
        clamping, so a mistyped slice fails loudly instead of showing the wrong
        motions.
        """
        raw = subset.strip()
        if ":" in raw:
            start_str, _, end_str = raw.partition(":")
            start = int(start_str) if start_str.strip() else 0
            end = int(end_str) if end_str.strip() else num_motions
            if end <= start:
                raise ValueError(
                    f"motion_subset '{subset}' is empty: end must exceed start."
                )
            ids = list(range(start, end))
        else:
            ids = [int(x) for x in raw.split(",") if x.strip()]
            if not ids:
                raise ValueError(f"motion_subset '{subset}' selects no motions.")
        oob = [i for i in ids if i < 0 or i >= num_motions]
        if oob:
            raise ValueError(
                f"motion_subset '{subset}' is out of bounds for a pack with "
                f"{num_motions} motions (offending indices e.g. {oob[:5]})."
            )
        return ids

    def _load_subset_from_file(self, file_path, subset: str) -> dict:
        """Load only `subset` of a packaged .pt, keeping the full pack off the device.

        The pack is memory-mapped on CPU (a 12 GB pack costs ~0.4 GB RSS), the selected
        motions are gathered, and only those are moved to self.device. Loading the whole
        pack straight to the GPU is what exhausts VRAM on large corpora.
        """
        src = torch.load(file_path, map_location="cpu", weights_only=False, mmap=True)
        num_motions = len(src["motion_num_frames"])
        ids = self.parse_motion_subset(subset, num_motions)

        frame_indices = []
        for i in ids:
            start = int(src["length_starts"][i])
            frame_indices.extend(range(start, start + int(src["motion_num_frames"][i])))
        frame_indices = torch.tensor(frame_indices, dtype=torch.long)

        frame_fields = ("gts", "grs", "gvs", "gavs", "dvs", "dps", "contacts", "lrs")
        motion_fields = ("motion_num_frames", "motion_lengths", "motion_dt", "motion_weights")

        out = {}
        for key, val in src.items():
            if key == "length_starts":
                continue  # rebuilt below
            if key in frame_fields and val is not None:
                out[key] = val[frame_indices].to(self.device)
            elif key in motion_fields and val is not None:
                out[key] = val[torch.tensor(ids, dtype=torch.long)].to(self.device)
            elif key == "motion_files" and val is not None:
                out[key] = tuple(val[i] for i in ids)
            elif torch.is_tensor(val):
                out[key] = val.to(self.device)
            else:
                out[key] = val

        num_frames = out["motion_num_frames"]
        shifted = num_frames.roll(1)
        shifted[0] = 0
        out["length_starts"] = shifted.cumsum(0)

        print(
            f"  motion_subset '{subset}': loaded {len(ids)}/{num_motions} motions "
            f"({len(frame_indices)} frames) onto {self.device}"
        )
        return out

    def smooth_contacts(self, window_size: int):
        """
        Smooth binary contact labels using a moving average filter.

        This method validates that contacts are binary, then applies a uniform
        moving average convolution to produce smoothed contact probabilities in [0, 1].
        The smoothing is applied in-place, replacing self.contacts.

        IMPORTANT: Smoothing respects motion boundaries - each motion is smoothed
        independently to avoid artifacts from one motion bleeding into another.

        A validated disk cache (a sidecar next to the motion pack, keyed by pack
        size+mtime, window size, and -- when sharded -- world_size+rank) lets us
        skip the O(num_motions) convolution loop on subsequent startups. The
        cache is best-effort: any error falls back to recompute, so it can never
        break loading. Disable via MOTION_LIB_SMOOTHED_CACHE_DISABLE=1.

        Args:
            window_size: Size of the moving average window (must be positive odd number)

        Raises:
            ValueError: If contacts are not binary or window_size is invalid
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        if window_size % 2 == 0:
            raise ValueError(
                f"window_size must be odd for symmetric smoothing, got {window_size}"
            )

        if self.contacts is None:
            print(
                "Warning: No contacts to smooth (contacts are None, likely all-zero at load time)"
            )
            return

        # Fast path: load pre-smoothed contacts from the disk cache if valid.
        if self._try_load_smoothed_contacts_cache(window_size):
            return

        # Validate that contacts are binary (0/1 or boolean)
        if self.contacts.dtype == torch.bool:
            # Boolean tensors are already binary, convert to float for smoothing
            self.contacts = self.contacts.float()
        else:
            # For non-boolean tensors, validate they contain only 0 and 1
            contacts_rounded = self.contacts.round()
            is_binary = torch.allclose(self.contacts, contacts_rounded, atol=1e-5)

            if not is_binary:
                # Find non-binary values for better error message
                non_binary_mask = ~torch.isclose(
                    self.contacts, contacts_rounded, atol=1e-5
                )
                non_binary_values = self.contacts[non_binary_mask]
                raise ValueError(
                    f"Contact labels must be binary (0 or 1) before smoothing. "
                    f"Found {non_binary_mask.sum().item()} non-binary values. "
                    f"Sample non-binary values: {non_binary_values[:5].tolist()}"
                )

        print(f"Smoothing contact labels with window size {window_size}...")

        num_motions = self.num_motions()

        # Smooth each motion independently to respect motion boundaries.
        self.contacts = _smooth_contacts_core(
            self.contacts,
            self.length_starts,
            self.motion_num_frames,
            num_motions,
            window_size,
            self.device,
        )

        print(
            f"Contact smoothing complete for {num_motions} motions. Contacts are now float values in [0, 1]."
        )

        # Persist to the disk cache for future startups (best-effort).
        self._save_smoothed_contacts_cache(window_size)

    def _smoothed_contacts_cache_spec(self, window_size):
        """Resolve (sidecar_path, signature, world_size, rank) or None if the
        cache is unavailable/disabled for this instance."""
        if _smoothed_contacts_cache_disabled():
            return None
        raw = _raw_pack_path(getattr(self, "motion_file", None))
        if not raw or not os.path.isfile(raw):
            return None
        sharded = bool(getattr(self, "sharded_across_ranks", False))
        ws = int(getattr(self, "shard_world_size")) if sharded else None
        rank = int(getattr(self, "shard_rank")) if sharded else None
        try:
            sig = _pack_signature(raw)
        except OSError:
            return None
        sidecar = _smoothed_contacts_sidecar_path(raw, window_size, ws, rank)
        return sidecar, sig, ws, rank

    def _try_load_smoothed_contacts_cache(self, window_size) -> bool:
        """Return True (and set self.contacts) on a validated cache hit."""
        try:
            spec = self._smoothed_contacts_cache_spec(window_size)
            if spec is None:
                return False
            sidecar, sig, _ws, _rank = spec
            if not os.path.isfile(sidecar):
                return False
            cached = torch.load(sidecar, map_location="cpu", weights_only=False)
            if (
                not isinstance(cached, dict)
                or cached.get("window") != window_size
                or cached.get("signature") != sig
                or "contacts" not in cached
            ):
                log.warning(
                    f"[smoothed-cache] stale/invalid sidecar {sidecar}; recomputing"
                )
                return False
            cc = cached["contacts"]
            if tuple(cc.shape) != tuple(self.contacts.shape):
                log.warning(
                    f"[smoothed-cache] shape mismatch {tuple(cc.shape)} != "
                    f"{tuple(self.contacts.shape)} in {sidecar}; recomputing"
                )
                return False
            self.contacts = cc.to(device=self.device, dtype=torch.float32)
            print(
                f"[smoothed-cache] loaded pre-smoothed contacts from {sidecar} "
                "(skipped recompute)"
            )
            return True
        except Exception as e:  # never let the cache break loading
            log.warning(f"[smoothed-cache] load failed ({e}); recomputing")
            return False

    def _save_smoothed_contacts_cache(self, window_size) -> None:
        """Atomically write the smoothed contacts sidecar (best-effort)."""
        tmp = None
        try:
            spec = self._smoothed_contacts_cache_spec(window_size)
            if spec is None:
                return
            sidecar, sig, ws, rank = spec
            if os.path.isfile(sidecar):
                return
            payload = {
                "contacts": self.contacts.detach().to("cpu", torch.float32),
                "signature": sig,
                "window": window_size,
                "world_size": ws,
                "rank": rank,
                "shape": tuple(self.contacts.shape),
            }
            tmp = f"{sidecar}.tmp.{os.getpid()}"
            torch.save(payload, tmp)
            os.replace(tmp, sidecar)
            print(f"[smoothed-cache] wrote pre-smoothed contacts to {sidecar}")
        except Exception as e:  # never let the cache break loading
            log.warning(
                f"[smoothed-cache] save failed ({e}); continuing without cache"
            )
            if tmp is not None:
                try:
                    if os.path.isfile(tmp):
                        os.remove(tmp)
                except OSError:
                    pass

    def translate_all_motions_to_origin(self, target_xy: Optional[torch.Tensor] = None):
        """
        Translate all motions so their first frames start at the specified x,y position.

        Args:
            target_xy: Target x,y position as tensor [2]. If None, uses (0.0, 0.0)
        """
        if target_xy is None:
            target_xy = torch.zeros(2, device=self.device)

            # Process each motion individually
        for motion_idx in range(self.num_motions()):
            # Get the range for this motion (convert tensors to integers)
            start_idx = self.length_starts[motion_idx].item()
            length = self.motion_num_frames[
                motion_idx
            ].item()  # Use motion_num_frames instead of motion_lengths
            end_idx = start_idx + length

            # Get the first frame's root position for this motion
            first_frame_root_pos = self.gts[start_idx, 0, :]  # [3] - root body position
            current_xy = first_frame_root_pos[:2]  # [2]

            # Calculate translation needed (only in x,y, keep z unchanged)
            translation_xy = target_xy - current_xy  # [2]
            translation = torch.zeros(3, device=self.device)
            translation[:2] = translation_xy

            self.gts[start_idx:end_idx, :, :] += translation.reshape(1, 1, 3)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Motion Library utilities")
    parser.add_argument(
        "--motion-path",
        type=str,
        default="",
        help="Path to motion file (.yaml, .motion, .pt) or directory",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="motion_lib.pt",
        help="Output file path for saving motion library",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for processing (cpu or cuda)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Clip motions longer than this to a random duration in "
        "[max-seconds - clip-delta, max-seconds]. Default: disabled (no clipping).",
    )
    parser.add_argument(
        "--clip-delta",
        type=float,
        default=3.0,
        help="Maximum random reduction (seconds) below --max-seconds when clipping. "
        "Clip target = uniform(max_seconds - clip_delta, max_seconds). Default: 3.",
    )
    parser.add_argument(
        "--clip-seed",
        type=int,
        default=42,
        help="RNG seed for reproducible per-motion clip lengths. Default: 42.",
    )

    args = parser.parse_args()

    motion_file = args.motion_path

    # If the file is a YAML, verify motion files are accessible relative to YAML location
    if motion_file.endswith(".yaml"):
        yaml_dir = Path(motion_file).parent.resolve()
        with open(motion_file, "r") as f:
            motion_config = yaml.load(f, Loader=yaml.SafeLoader)

        motions = motion_config.get("motions", [])
        if motions and "file" in motions[0]:
            first_motion_path = yaml_dir / motions[0]["file"]
            if not first_motion_path.exists():
                raise FileNotFoundError(
                    f"Motion file not found: {first_motion_path}\n"
                    f"The YAML references '{motions[0]['file']}' but it doesn't exist "
                    f"relative to the YAML directory ({yaml_dir}).\n"
                    f"Did you forget to copy the YAML file to the motion directory?"
                )

    # Create and save motion library
    motion_lib = MotionLib(
        config=MotionLibConfig(
            motion_file=motion_file,
            max_seconds=args.max_seconds,
            clip_delta=args.clip_delta,
            clip_seed=args.clip_seed,
        ),
        device=args.device,
    )
    motion_lib.save_to_file(args.output_file)
