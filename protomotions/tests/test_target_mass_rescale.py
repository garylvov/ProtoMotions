# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for runtime target-mass reweighting (rescale_to_target_mass /
load_mix_target) added for the dataset-mix reweight task.

Covers:
  - rescale_to_target_mass produces exact target global fractions.
  - Rescaled weights, run through compute_rank_shard for every rank of an
    8-way shard, preserve the target per-class fractions per-rank AND when
    local class masses are summed back across ranks.
  - load_mix_target: no-sidecar path returns None (byte-identical behavior).
  - load_mix_target: invalid sidecars (bad sum, unknown/missing class names)
    raise loudly instead of silently falling back.
"""

from __future__ import annotations

import json

import pytest
import torch

from protomotions.components.motion_lib import (
    compute_rank_shard,
    load_mix_target,
    rescale_to_target_mass,
)


def _make_synthetic_pack(seed=0):
    """3 classes of uneven size/weight, boundaries contiguous as the real
    pack builder produces (base then synth classes in mix order)."""
    g = torch.Generator().manual_seed(seed)
    sizes = {"base": 400, "pauses": 250, "speed_warp": 150}
    boundaries = []
    start = 0
    weights = []
    for name, n in sizes.items():
        boundaries.append((name, start, start + n))
        # Nontrivial per-motion weight variance within each class.
        weights.append(0.5 + torch.rand(n, generator=g))
        start += n
    motion_weights = torch.cat(weights)
    return motion_weights, boundaries


def test_rescale_to_target_mass_hits_exact_target_fractions():
    motion_weights, boundaries = _make_synthetic_pack()
    target = {"base": 0.80, "pauses": 0.13, "speed_warp": 0.07}

    rescaled = rescale_to_target_mass(motion_weights, boundaries, target)

    assert rescaled.shape == motion_weights.shape
    assert rescaled.dtype == motion_weights.dtype
    total = float(rescaled.sum())
    for name, start, end in boundaries:
        frac = float(rescaled[start:end].sum()) / total
        # float32 round-trip (motion_weights dtype) limits precision to ~1e-7,
        # not the 1e-9 achievable purely in float64.
        assert frac == pytest.approx(target[name], abs=1e-6)


def test_rescale_preserves_total_mass():
    motion_weights, boundaries = _make_synthetic_pack()
    target = {"base": 0.5, "pauses": 0.3, "speed_warp": 0.2}
    rescaled = rescale_to_target_mass(motion_weights, boundaries, target)
    # Total mass is invariant by construction (target fractions sum to 1.0).
    assert float(rescaled.sum()) == pytest.approx(float(motion_weights.sum()), rel=1e-9)


def test_rescale_then_shard_preserves_target_per_rank_and_summed():
    """Rescale globally, then shard across 8 ranks: each rank's local
    class mass fraction (post compute_rank_shard renorm) must equal the
    target, AND summing local class masses back across ranks must equal
    the target class's global mass."""
    motion_weights, boundaries = _make_synthetic_pack()
    target = {"base": 0.80, "pauses": 0.13, "speed_warp": 0.07}
    rescaled = rescale_to_target_mass(motion_weights, boundaries, target)

    world_size = 8
    num_motions = motion_weights.shape[0]
    global_total = float(rescaled.sum())

    per_class_local_mass_sum = {name: 0.0 for name, _, _ in boundaries}

    for rank in range(world_size):
        local_indices, local_weights, report = compute_rank_shard(
            num_motions, rescaled, rank, world_size, boundaries
        )
        # report's global_mass_fraction is derived from `rescaled`, so it
        # should already equal target (this is what makes assert C in
        # _shard_launch_asserts transparently check against target when a
        # mix_target sidecar is active).
        for c in report["classes"]:
            assert c["global_mass_fraction"] == pytest.approx(
                target[c["name"]], abs=1e-6
            )

        # Per-rank: local mass renormalized to equal each class's global
        # (target) mass exactly.
        for name, start, end in boundaries:
            in_class = (local_indices >= start) & (local_indices < end)
            local_mass = float(local_weights[in_class].sum())
            expected_global_mass = target[name] * global_total
            assert local_mass == pytest.approx(expected_global_mass, rel=1e-6)
            per_class_local_mass_sum[name] += local_mass

    # Summed across all 8 ranks, each class's total local mass is
    # world_size * its global (target) mass (since compute_rank_shard sets
    # EACH rank's local mass equal to the full global class mass, by
    # design — see compute_rank_shard docstring: "each rank's per-class
    # sampling mass fractions exactly match the global pack's").
    for name, _, _ in boundaries:
        expected = world_size * target[name] * global_total
        assert per_class_local_mass_sum[name] == pytest.approx(expected, rel=1e-6)


def test_rescale_zero_mass_class_raises():
    motion_weights, boundaries = _make_synthetic_pack()
    motion_weights[boundaries[1][1] : boundaries[1][2]] = 0.0
    target = {"base": 0.80, "pauses": 0.13, "speed_warp": 0.07}
    with pytest.raises(AssertionError):
        rescale_to_target_mass(motion_weights, boundaries, target)


def test_load_mix_target_missing_sidecar_returns_none(tmp_path):
    _, boundaries = _make_synthetic_pack()
    sidecar = tmp_path / "pack.pt.mix_target.json"
    assert not sidecar.exists()
    assert load_mix_target(str(sidecar), boundaries) is None


def test_load_mix_target_valid_sidecar_round_trips(tmp_path):
    _, boundaries = _make_synthetic_pack()
    target = {"base": 0.80, "pauses": 0.13, "speed_warp": 0.07}
    sidecar = tmp_path / "pack.pt.mix_target.json"
    sidecar.write_text(json.dumps({"target_mass": target}))
    loaded = load_mix_target(str(sidecar), boundaries)
    assert loaded == pytest.approx(target)


def test_load_mix_target_bad_sum_raises(tmp_path):
    _, boundaries = _make_synthetic_pack()
    target = {"base": 0.80, "pauses": 0.13, "speed_warp": 0.10}  # sums to 1.03
    sidecar = tmp_path / "pack.pt.mix_target.json"
    sidecar.write_text(json.dumps({"target_mass": target}))
    with pytest.raises(ValueError, match="sum"):
        load_mix_target(str(sidecar), boundaries)


def test_load_mix_target_unknown_class_raises(tmp_path):
    _, boundaries = _make_synthetic_pack()
    target = {"base": 0.80, "pauses": 0.13, "unknown_class": 0.07}
    sidecar = tmp_path / "pack.pt.mix_target.json"
    sidecar.write_text(json.dumps({"target_mass": target}))
    with pytest.raises(ValueError, match="do not match"):
        load_mix_target(str(sidecar), boundaries)


def test_load_mix_target_missing_class_raises(tmp_path):
    _, boundaries = _make_synthetic_pack()
    # Missing 'speed_warp' entirely.
    target = {"base": 0.87, "pauses": 0.13}
    sidecar = tmp_path / "pack.pt.mix_target.json"
    sidecar.write_text(json.dumps({"target_mass": target}))
    with pytest.raises(ValueError, match="do not match"):
        load_mix_target(str(sidecar), boundaries)


def test_load_mix_target_missing_key_raises(tmp_path):
    _, boundaries = _make_synthetic_pack()
    sidecar = tmp_path / "pack.pt.mix_target.json"
    sidecar.write_text(json.dumps({"not_target_mass": {}}))
    with pytest.raises(ValueError, match="target_mass"):
        load_mix_target(str(sidecar), boundaries)
