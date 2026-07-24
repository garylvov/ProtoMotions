# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for per-rank motion-lib shard partition math.

Mirrors the invariants ``_shard_launch_asserts`` checks at launch (which use
torch.distributed collectives), but exercised as pure functions
(``compute_rank_shard`` / ``load_shard_class_boundaries``) so no process group
is needed:

  * shards are disjoint and their union is the complete motion set,
  * shard sizes are balanced to within 1 (interleave ``motions[rank::W]``),
  * with a ``.mix.json`` sidecar, per-class renormalized sampling-mass fractions
    are identical across ranks and equal to the global pack's.
"""

import json

import pytest
import torch

from protomotions.components.motion_lib import (
    compute_rank_shard,
    load_shard_class_boundaries,
)


def _all_shards(num_motions, weights, world_size, class_boundaries=None):
    return [
        compute_rank_shard(
            num_motions, weights, r, world_size,
            class_boundaries=class_boundaries,
        )
        for r in range(world_size)
    ]


@pytest.mark.parametrize("num_motions,world_size", [(20, 3), (17, 4), (8, 8), (100, 7)])
def test_shards_disjoint_and_union_complete(num_motions, world_size):
    weights = torch.rand(num_motions) + 0.1
    shards = _all_shards(num_motions, weights, world_size)

    seen = torch.cat([li for li, _, _ in shards]).tolist()
    assert sorted(seen) == list(range(num_motions))          # union complete
    assert len(seen) == len(set(seen))                       # disjoint


@pytest.mark.parametrize("num_motions,world_size", [(20, 3), (17, 4), (100, 7)])
def test_shard_sizes_balanced_within_one(num_motions, world_size):
    weights = torch.rand(num_motions) + 0.1
    sizes = [li.numel() for li, _, _ in _all_shards(num_motions, weights, world_size)]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == num_motions


def test_interleave_pattern_is_rank_stride_world_size():
    num_motions, world_size = 20, 3
    weights = torch.rand(num_motions) + 0.1
    for rank in range(world_size):
        local, _, _ = compute_rank_shard(num_motions, weights, rank, world_size)
        expected = torch.arange(rank, num_motions, world_size)
        assert torch.equal(local, expected)


def test_load_shard_class_boundaries_from_sidecar(tmp_path):
    sidecar = tmp_path / "pack.mix.json"
    sidecar.write_text(json.dumps(
        {"num_motions": 20, "mix": {"clean": {"kept": 12}, "synth": {"kept": 8}}}
    ))
    boundaries = load_shard_class_boundaries(str(sidecar), 20)
    assert boundaries == [("clean", 0, 12), ("synth", 12, 20)]


def test_sidecar_count_mismatch_is_ignored(tmp_path):
    sidecar = tmp_path / "pack.mix.json"
    # class counts sum to 15, pack claims 20 -> unusable, returns None.
    sidecar.write_text(json.dumps(
        {"num_motions": 20, "mix": {"clean": {"kept": 10}, "synth": {"kept": 5}}}
    ))
    assert load_shard_class_boundaries(str(sidecar), 20) is None


def test_missing_sidecar_returns_none():
    assert load_shard_class_boundaries("/no/such/pack.mix.json", 20) is None


def test_per_class_mass_renormalized_equal_across_ranks(tmp_path):
    num_motions, world_size = 20, 3
    weights = torch.rand(num_motions) + 0.1
    sidecar = tmp_path / "pack.mix.json"
    sidecar.write_text(json.dumps(
        {"num_motions": num_motions,
         "mix": {"clean": {"kept": 12}, "synth": {"kept": 8}}}
    ))
    boundaries = load_shard_class_boundaries(str(sidecar), num_motions)

    # Global per-class mass fractions.
    total = float(weights.sum())
    global_frac = {
        name: float(weights[start:end].sum()) / total
        for name, start, end in boundaries
    }

    for rank in range(world_size):
        local, local_w, report = compute_rank_shard(
            num_motions, weights, rank, world_size, class_boundaries=boundaries
        )
        assert report["renormalized"] is True
        local_total = float(local_w.sum())
        for name, start, end in boundaries:
            in_class = (local >= start) & (local < end)
            local_frac = float(local_w[in_class].sum()) / local_total
            # local per-class fraction == global fraction (the invariant
            # _shard_launch_asserts' assert C enforces across ranks).
            assert abs(local_frac - global_frac[name]) < 1e-6, (rank, name)


def test_report_records_global_mass_fractions(tmp_path):
    num_motions, world_size = 20, 4
    weights = torch.rand(num_motions) + 0.1
    sidecar = tmp_path / "pack.mix.json"
    sidecar.write_text(json.dumps(
        {"num_motions": num_motions,
         "mix": {"clean": {"kept": 12}, "synth": {"kept": 8}}}
    ))
    boundaries = load_shard_class_boundaries(str(sidecar), num_motions)
    total = float(weights.sum())

    _, _, report = compute_rank_shard(
        num_motions, weights, 0, world_size, class_boundaries=boundaries
    )
    reported = {c["name"]: c["global_mass_fraction"] for c in report["classes"]}
    for name, start, end in boundaries:
        # report computes fractions in float64; our expected sums float32, so
        # allow a float32-rounding-sized tolerance.
        expected = float(weights[start:end].sum()) / total
        assert abs(reported[name] - expected) < 1e-6, name


def test_without_boundaries_no_renorm_but_still_partitions():
    num_motions, world_size = 20, 3
    weights = torch.rand(num_motions) + 0.1
    seen = []
    for rank in range(world_size):
        local, local_w, report = compute_rank_shard(
            num_motions, weights, rank, world_size
        )
        assert report["renormalized"] is False
        assert report["classes"] == []
        # weights plain-sliced (no scaling) when no sidecar.
        assert torch.allclose(local_w, weights[local].float())
        seen += local.tolist()
    assert sorted(seen) == list(range(num_motions))


def test_cannot_shard_fewer_motions_than_ranks():
    with pytest.raises(AssertionError):
        compute_rank_shard(3, torch.rand(3) + 0.1, 0, 4)


def test_rank_out_of_range_rejected():
    with pytest.raises(AssertionError):
        compute_rank_shard(20, torch.rand(20) + 0.1, 4, 4)
