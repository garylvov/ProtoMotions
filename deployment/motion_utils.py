# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motion playback utilities for deployment.

:class:`MotionPlayer` loads **a single motion clip** and provides per-frame
state (joint positions, velocities, body rotations, etc.) at a fixed control
rate.  It accepts four input formats:

- A single ``.motion`` file (RobotState dict with ``fps``, ``dof_pos``, …).
- A packaged ``.pt`` library (multi-motion file with ``length_starts``,
  ``gts``, ``grs``, …) -- requires an explicit ``motion_index`` to select
  which clip to load.
- A **motion pack** (``{"clips": [...], "meta": {...}}``) as written by the
  eval harness -- requires an explicit ``motion_index``.  See
  :func:`unwrap_pack_clip`.
- A pre-resampled cache previously written by :meth:`cache_to_file`.

Two runtime modes
-----------------

**Interpolation mode** (first run — requires PyTorch + protomotions)
    Loads one of the raw formats above and resamples to the control rate
    using the *exact same* interpolation code as training
    (``calc_frame_blend`` + ``interpolate_pos`` + ``interpolate_quat`` from
    ``protomotions.utils.motion_interpolation_utils``).  This guarantees
    zero discrepancy with training-time motion queries.

    Call :meth:`cache_to_file` afterwards to write a pre-resampled ``.pt``
    so that future runs can skip this step entirely.

**Cached mode** (subsequent runs — PyTorch-computation-free)
    Loads a cache written by :meth:`cache_to_file`.  All queries become
    plain NumPy array indexing — no SLERP, no interpolation, no
    ``protomotions`` import.  ``torch.load`` is still used to read the
    ``.pt`` file; to eliminate even that, convert the cache to ``.npz`` and
    adjust the load call.

Cache file format
-----------------
A ``torch.save``-d dict with NumPy arrays::

    {
        "dof_pos":      np.ndarray [num_frames, num_dofs],
        "dof_vel":      np.ndarray [num_frames, num_dofs],
        "body_rot":     np.ndarray [num_frames, num_bodies, 4]  (xyzw),
        "body_pos":     np.ndarray [num_frames, num_bodies, 3],
        "body_vel":     np.ndarray [num_frames, num_bodies, 3],
        "body_ang_vel": np.ndarray [num_frames, num_bodies, 3],
        "control_dt":   float,
        "num_frames":   int,
    }

Example usage
-------------
::

    # From a single .motion file (first run, writes cache)
    player = MotionPlayer("data/motions/walk.motion", control_dt=0.02)
    player.cache_to_file("data/motions/walk.50fps.pt")

    # From a packaged .pt with explicit motion index
    player = MotionPlayer("data/motions/library.pt", motion_index=3, control_dt=0.02)

    # From a cache (no protomotions needed)
    player = MotionPlayer("data/motions/walk.50fps.pt")

    for frame in range(player.total_frames):
        state   = player.get_state_at_frame(frame)
        futures = player.get_future_references(frame, step_indices=list(range(1, 26)))
        ...
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

__all__ = [
    "MotionPlayer",
    "is_motion_pack",
    "motion_names",
    "num_motions",
    "unwrap_pack_clip",
]

# Keys present in every state/future-reference dict
_STATE_KEYS = ("dof_pos", "dof_vel", "body_rot", "body_pos", "body_vel", "body_ang_vel")

# Keys stored in cache files (superset of _STATE_KEYS plus metadata)
_CACHE_METADATA_KEYS = {"control_dt", "num_frames"}

#: Top-level key that identifies a motion pack.
_PACK_KEY = "clips"


def _is_cache_file(data: dict) -> bool:
    """Return True if *data* looks like a pre-resampled cache (not a raw motion)."""
    return "control_dt" in data and "body_rot" in data


# ---------------------------------------------------------------------------
# Motion-pack adapter
# ---------------------------------------------------------------------------
#
# The eval harness packages a whole testset as a single **motion pack**::
#
#     {"clips": [clip, clip, ...], "meta": {...}}
#
# Each ``clip`` is already resampled to a fixed control rate, so key for key it
# *is* a MotionPlayer cache dict (``dof_pos``/``body_rot``/``control_dt``/
# ``num_frames``/…) carrying extra provenance (``name``, ``category``,
# ``metrics``, …).  Written by ``scripts/wbc/eval/build_canonical_eval.py``;
# the harness's own reader is ``batch_mj_eval.load_testset``.
#
# Until this adapter existed, ``MotionPlayer`` rejected packs outright with
# "Unrecognised raw motion format", which is the whole reason the masked-mimic
# student could never be rendered on ``canonical_eval_v1.pt``.
#
# The fix is a NORMALISER rather than a branch inside the loader: a pack is
# unwrapped to the one clip that was asked for, and that clip then travels the
# ordinary cache path.  Nothing downstream of ``__init__`` learns that packs
# exist, so there is exactly one place that knows the pack layout.


def is_motion_pack(data) -> bool:
    """True iff *data* is a motion pack (a dict whose ``clips`` is a sequence).

    Deliberately structural, not name-based: the format is identified by what
    it contains, so a pack stays recognisable under any filename.
    """
    return isinstance(data, dict) and isinstance(data.get(_PACK_KEY), (list, tuple))


def unwrap_pack_clip(data: dict, motion_index: int, control_dt: float | None = None) -> dict:
    """Return clip *motion_index* of a motion pack as a plain cache dict.

    Args:
        data: the loaded motion pack.
        motion_index: which clip to select.  Required -- a pack holds many
            clips and there is no sensible default beyond 0, so an
            out-of-range index is an error rather than a clamp.
        control_dt: if given, the rate the caller intends to play at.  Pack
            clips are **already resampled**, so a mismatch cannot be honoured
            and is refused loudly instead of silently replaying at the wrong
            rate (which desynchronises every time-indexed signal downstream:
            the masked-mimic conditioning ladder, reference lookups, the
            frame pairing the metrics are defined on).

    Raises:
        IndexError: *motion_index* is outside the pack.
        ValueError: the clip is missing state arrays, or its stored
            ``control_dt`` disagrees with the requested one.
    """
    clips = data[_PACK_KEY]
    n = len(clips)
    if n == 0:
        raise ValueError("motion pack contains no clips")
    idx = int(motion_index)
    if not 0 <= idx < n:
        raise IndexError(
            f"motion_index {motion_index} is out of range for a {n}-clip motion pack"
        )
    clip = clips[idx]
    if not isinstance(clip, dict):
        raise ValueError(
            f"clip {idx} of this motion pack is a {type(clip).__name__}, not a dict; "
            "a pack clip must be a cache-shaped mapping"
        )
    missing = [k for k in _STATE_KEYS if k not in clip]
    if missing:
        raise ValueError(
            f"clip {idx} of this motion pack is missing {missing}. A pack clip must "
            f"carry every state array ({', '.join(_STATE_KEYS)}); refusing to play a "
            "clip whose missing channels would be read as zeros."
        )
    if control_dt is not None and "control_dt" in clip:
        src = float(clip["control_dt"])
        if abs(src - float(control_dt)) > 1e-9:
            raise ValueError(
                f"clip {idx} of this motion pack is stored at control_dt={src} s but "
                f"{float(control_dt)} s was requested. Pack clips are ALREADY "
                "resampled, so the request cannot be honoured; replaying them at "
                "another rate silently desynchronises every time-indexed signal. "
                "Rebuild the pack at the required rate instead."
            )
    return clip


def _as_container(source) -> dict:
    """Accept an already-loaded dict or a path to one."""
    if isinstance(source, dict):
        return source
    import torch

    return torch.load(str(source), map_location="cpu", weights_only=False)


def motion_names(source) -> List[str]:
    """Per-clip names for any supported multi-motion container.

    Callers used to reach into the file themselves with
    ``torch.load(...).get("motion_names", [])``, which returns ``[]`` for a
    motion pack -- so every rendered file fell back to ``clip<i>`` and the
    output was unlabelled.  Packs carry the name on each clip instead; this is
    the one accessor that knows both layouts.

    Returns an empty list when the container declares no names, matching the
    previous ``.get("motion_names", [])`` contract so callers that guard with
    ``if idx < len(names)`` keep working.
    """
    data = _as_container(source)
    if is_motion_pack(data):
        return [str(c.get("name", f"clip{i}")) if isinstance(c, dict) else f"clip{i}"
                for i, c in enumerate(data[_PACK_KEY])]
    names = data.get("motion_names")
    return [] if names is None else [str(n) for n in names]


def num_motions(source) -> int:
    """How many clips *source* holds (1 for a single ``.motion`` or a cache)."""
    data = _as_container(source)
    if is_motion_pack(data):
        return len(data[_PACK_KEY])
    if "length_starts" in data:
        return len(data["length_starts"])
    return 1


class MotionPlayer:
    """Lightweight player for a **single** motion clip at a fixed control rate.

    Accepts four input formats (auto-detected):

    1. **Single ``.motion`` file** — a RobotState dict saved via
       ``torch.save`` with keys ``fps``, ``dof_pos``, ``rigid_body_pos``, etc.
    2. **Packaged ``.pt`` library** — a multi-motion file with
       ``length_starts``, ``gts``, ``grs``, … keys.  You **must** pass
       ``motion_index`` to select which clip to extract.
    3. **Motion pack** — ``{"clips": [...], "meta": {...}}`` as written by the
       eval harness.  You **must** pass ``motion_index``.  Clips are already
       resampled, so the pack is normalised to a single cache dict up front
       (:func:`unwrap_pack_clip`) and then loaded like any other cache.
    4. **Pre-resampled cache** — written by :meth:`cache_to_file`, containing
       NumPy arrays at the control rate.  Auto-detected by the presence of
       a ``control_dt`` key.

    Parameters
    ----------
    motion_file:
        Path to any of the four formats above.
    motion_index:
        Index of the clip to extract from a packaged ``.pt`` library or a
        motion pack.  **Required** for both, ignored for ``.motion`` and
        cache files.
    control_dt:
        Control period in seconds (default 0.02 s = 50 Hz).  Determines the
        resampling rate when loading raw motion data.  Ignored when loading
        from a cache file (the cache stores its own dt).
    """

    def __init__(
        self,
        motion_file: str,
        motion_index: int = 0,
        control_dt: float = 0.02,
    ):
        import torch

        self._torch = torch
        motion_file = str(motion_file)
        data = torch.load(motion_file, map_location="cpu", weights_only=False)

        # Normalise a motion pack down to the single clip that was asked for.
        # The result is cache-shaped, so the two branches below are unchanged.
        if is_motion_pack(data):
            data = unwrap_pack_clip(data, motion_index, control_dt)

        if _is_cache_file(data):
            self._load_cache(data)
        else:
            self._load_raw(data, motion_index, control_dt)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def total_frames(self) -> int:
        """Total number of frames available at the control rate."""
        return self._num_frames

    @property
    def num_bodies(self) -> int:
        return self._body_rot.shape[1]

    @property
    def num_dofs(self) -> int:
        return self._dof_pos.shape[1]

    @property
    def control_dt(self) -> float:
        return self._control_dt

    def get_state_at_frame(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Return the motion state at *frame_idx*.

        Frame index is clamped to ``[0, total_frames - 1]``.

        Returns
        -------
        dict with keys ``dof_pos``, ``dof_vel``, ``body_rot``, ``body_pos``,
        ``body_vel``, ``body_ang_vel``.  All arrays have no batch dimension.
        """
        idx = int(np.clip(frame_idx, 0, self._num_frames - 1))
        return {
            "dof_pos":      self._dof_pos[idx],
            "dof_vel":      self._dof_vel[idx],
            "body_rot":     self._body_rot[idx],
            "body_pos":     self._body_pos[idx],
            "body_vel":     self._body_vel[idx],
            "body_ang_vel": self._body_ang_vel[idx],
        }

    def get_future_references(
        self,
        frame_idx: int,
        step_indices: List[int],
    ) -> Dict[str, np.ndarray]:
        """Return stacked future motion states.

        Each entry in *step_indices* is a positive 1-indexed offset (e.g.
        ``step_indices=[1, 25]`` means ``frame_idx + 1`` and
        ``frame_idx + 25``).  Future frames beyond the last available frame
        are clamped to the last frame.

        Parameters
        ----------
        frame_idx:
            Current frame index (0-based).
        step_indices:
            List of positive integer offsets (1-indexed), matching the
            ``MimicControlConfig.future_steps`` list used during training.

        Returns
        -------
        dict with keys ``dof_pos``, ``dof_vel``, ``body_rot``, ``body_pos``,
        ``body_vel``, ``body_ang_vel``.  Arrays have shape
        ``[len(step_indices), ...]``.
        """
        future_states = [
            self.get_state_at_frame(frame_idx + s) for s in step_indices
        ]
        return {
            key: np.stack([s[key] for s in future_states], axis=0)
            for key in _STATE_KEYS
        }

    def cache_to_file(self, output_path: str) -> None:
        """Write a pre-resampled cache file at the current control rate.

        The cache contains NumPy arrays (not tensors) so it can be loaded
        without any PyTorch computation on subsequent runs.  The file is
        written with ``torch.save`` so ``torch.load`` is still needed to
        read it; the format is intentionally simple and could be converted
        to ``.npz`` if PyTorch must be removed entirely.

        Args:
            output_path: Destination path, e.g. ``walk.50fps.pt``.
        """
        import torch

        cache = {
            "dof_pos":      self._dof_pos,
            "dof_vel":      self._dof_vel,
            "body_rot":     self._body_rot,
            "body_pos":     self._body_pos,
            "body_vel":     self._body_vel,
            "body_ang_vel": self._body_ang_vel,
            "control_dt":   self._control_dt,
            "num_frames":   self._num_frames,
        }
        torch.save(cache, output_path)
        print(
            f"[MotionPlayer] Cached {self._num_frames} frames @ "
            f"{1.0 / self._control_dt:.0f} Hz → {output_path}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_cache(self, data: dict) -> None:
        """Load from a pre-resampled cache dict."""
        self._dof_pos      = np.asarray(data["dof_pos"],      dtype=np.float32)
        self._dof_vel      = np.asarray(data["dof_vel"],      dtype=np.float32)
        self._body_rot     = np.asarray(data["body_rot"],     dtype=np.float32)
        self._body_pos     = np.asarray(data["body_pos"],     dtype=np.float32)
        self._body_vel     = np.asarray(data["body_vel"],     dtype=np.float32)
        self._body_ang_vel = np.asarray(data["body_ang_vel"], dtype=np.float32)
        self._control_dt   = float(data["control_dt"])
        self._num_frames   = int(data["num_frames"])
        self._cached = True
        print(
            f"[MotionPlayer] Loaded cache: {self._num_frames} frames "
            f"@ {1.0 / self._control_dt:.0f} Hz"
        )

    def _load_raw(self, data: dict, motion_index: int, control_dt: float) -> None:
        """Load from a raw ProtoMotions motion file and resample.

        Supports two formats:
        - **Packaged library** (``.pt`` with ``length_starts``, ``gts``, ``grs``, …):
          multi-motion file; ``motion_index`` selects which motion to load.
        - **Single-motion file** (``.motion`` / ``.npy`` loaded via ``torch.load``):
          dict with ``fps``, ``dof_pos``, ``rigid_body_pos``, ``rigid_body_rot``,
          ``rigid_body_vel``, ``rigid_body_ang_vel`` keys (RobotState format).

        Uses the exact same interpolation as training:
        ``calc_frame_blend`` + ``interpolate_pos`` + ``interpolate_quat``.
        """
        import torch
        from protomotions.utils.motion_interpolation_utils import (
            calc_frame_blend,
            interpolate_pos,
            interpolate_quat,
        )

        self._control_dt = control_dt
        self._cached = False

        if "length_starts" in data:
            # ---- Multi-motion packaged library (.pt) ----
            length_starts     = data["length_starts"]
            motion_num_frames = data["motion_num_frames"]
            motion_dt_all     = data["motion_dt"]

            start  = int(length_starts[motion_index].item())
            nf     = int(motion_num_frames[motion_index].item())
            end    = start + nf
            src_dt = float(motion_dt_all[motion_index].item())

            gts  = data["gts"][start:end]
            grs  = data["grs"][start:end]
            gvs  = data["gvs"][start:end]
            gavs = data["gavs"][start:end]
            dps  = data["dps"][start:end]
            dvs  = data["dvs"][start:end]
            motion_length = src_dt * (nf - 1)

        elif "rigid_body_pos" in data:
            # ---- Single-motion file (.motion / .npy via torch.load) ----
            # RobotState dict: fps, dof_pos, dof_vel, rigid_body_pos/rot/vel/ang_vel
            fps    = float(data["fps"])
            src_dt = 1.0 / fps

            gts  = data["rigid_body_pos"]     # [nf, num_bodies, 3]
            grs  = data["rigid_body_rot"]      # [nf, num_bodies, 4]  xyzw (COMMON)
            gvs  = data["rigid_body_vel"]      # [nf, num_bodies, 3]
            gavs = data["rigid_body_ang_vel"]  # [nf, num_bodies, 3]
            dps  = data["dof_pos"]             # [nf, num_dofs]
            dvs  = data["dof_vel"]             # [nf, num_dofs]
            nf   = gts.shape[0]
            motion_length = src_dt * (nf - 1)
        else:
            raise ValueError(
                "Unrecognised raw motion format.  Expected one of:\n"
                "  - packaged library: keys 'length_starts', 'gts', 'grs', …\n"
                "  - single-motion:   keys 'rigid_body_pos', 'fps', 'dof_pos', …\n"
                "  - motion pack:     key 'clips' (a list of cache-shaped dicts)\n"
                "  - resampled cache: keys 'control_dt', 'body_rot', …\n"
                f"got keys: {sorted(data)[:12]}"
            )

        # ---- resample to control rate via training-identical interpolation ----
        num_ctrl_frames = max(1, int(round(motion_length / control_dt)) + 1)
        ctrl_times = torch.linspace(0.0, motion_length, num_ctrl_frames)

        # Expand to batch dim = 1 so calc_frame_blend works
        motion_len_t  = torch.tensor([motion_length])
        num_frames_t  = torch.tensor([nf])
        motion_dt_t   = torch.tensor([src_dt])

        # calc_frame_blend expects scalar tensors per query
        f0_list, f1_list, blend_list = [], [], []
        for t in ctrl_times:
            t_t = t.unsqueeze(0)
            f0, f1, bl = calc_frame_blend(t_t, motion_len_t, num_frames_t, motion_dt_t)
            f0_list.append(f0)
            f1_list.append(f1)
            blend_list.append(bl)

        f0    = torch.cat(f0_list)     # [num_ctrl_frames]
        f1    = torch.cat(f1_list)
        blend = torch.cat(blend_list)

        def _interp_pos(src):
            # src: [nf, ...], returns [num_ctrl_frames, ...]
            s0 = src[f0]
            s1 = src[f1]
            return interpolate_pos(s0, s1, blend)

        def _interp_quat(src):
            s0 = src[f0]
            s1 = src[f1]
            return interpolate_quat(s0, s1, blend)

        body_pos     = _interp_pos(gts)   # [T, num_bodies, 3]
        body_rot     = _interp_quat(grs)  # [T, num_bodies, 4]  xyzw
        body_vel     = _interp_pos(gvs)
        body_ang_vel = _interp_pos(gavs)
        dof_pos      = _interp_pos(dps)   # [T, num_dofs]
        dof_vel      = _interp_pos(dvs)

        self._dof_pos      = dof_pos.numpy().astype(np.float32)
        self._dof_vel      = dof_vel.numpy().astype(np.float32)
        self._body_rot     = body_rot.numpy().astype(np.float32)
        self._body_pos     = body_pos.numpy().astype(np.float32)
        self._body_vel     = body_vel.numpy().astype(np.float32)
        self._body_ang_vel = body_ang_vel.numpy().astype(np.float32)
        self._num_frames   = num_ctrl_frames

        print(
            f"[MotionPlayer] Loaded raw motion #{motion_index}: "
            f"{nf} source frames @ {1.0 / src_dt:.1f} Hz → "
            f"{num_ctrl_frames} resampled frames @ {1.0 / control_dt:.0f} Hz"
        )
