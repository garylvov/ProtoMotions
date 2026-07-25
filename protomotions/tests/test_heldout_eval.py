# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for held-out eval support (PM_HELDOUT_FILE).

Covers:
  * the pure global<->local mapping / artifact-loading helpers in motion_lib,
  * MotionManager sampler masking (never-sampled over 10k draws, renorm,
    resume/curriculum re-apply, byte-identical when unset),
  * MimicEvaluator held-out targeting (compact buffers, tracks only held-out
    clips, cap at EVAL_SUBSET_N, byte-identical when unset).

Never touches a GPU or loads a real motion pack.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from protomotions.components.motion_lib import (
    load_heldout_global_ids,
    map_heldout_to_local,
)
from protomotions.envs.motion_manager.config import MotionManagerConfig
from protomotions.envs.motion_manager.motion_manager import MotionManager
from protomotions.agents.evaluators.mimic_evaluator import MimicEvaluator


# ---------------------------------------------------------------- pure helpers


def test_load_heldout_global_ids_sorted_unique(tmp_path):
    p = tmp_path / "x.heldout.json"
    p.write_text(json.dumps({"ids": [7, 1, 7, 3]}))
    out = load_heldout_global_ids(str(p))
    assert out.tolist() == [1, 3, 7]


def test_load_heldout_global_ids_missing_or_malformed(tmp_path):
    assert load_heldout_global_ids(str(tmp_path / "nope.json")).numel() == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_heldout_global_ids(str(bad)).numel() == 0
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"ids": []}))
    assert load_heldout_global_ids(str(empty)).numel() == 0


def test_map_heldout_unsharded_is_identity_in_range():
    g = torch.tensor([2, 5, 9, 100])  # 100 is out of range -> dropped
    local = map_heldout_to_local(g, global_motion_ids=None, num_motions=10)
    assert local.tolist() == [2, 5, 9]


def test_map_heldout_sharded_translates_global_to_local():
    # rank owns global clips [1, 4, 7, 10] at local ids [0, 1, 2, 3]
    gids = torch.tensor([1, 4, 7, 10])
    # hold out globals {4, 10, 99}; 99 lives on another rank -> dropped here
    heldout = torch.tensor([4, 10, 99])
    local = map_heldout_to_local(heldout, global_motion_ids=gids, num_motions=4)
    assert local.tolist() == [1, 3]  # local ids of globals 4 and 10


# ---------------------------------------------------------------- manager mask


class _MotionLib:
    def __init__(self, n=100, sharded=False, global_ids=None):
        self.motion_weights = torch.full((n,), 0.1, dtype=torch.float)
        self.motion_lengths = torch.arange(1, n + 1, dtype=torch.float)
        self.motion_file = "pack.pt"
        self.sharded_across_ranks = sharded
        self.global_motion_ids = global_ids

    def num_motions(self):
        return len(self.motion_weights)


def _manager(motion_lib, num_envs=4):
    return MotionManager(
        MotionManagerConfig(init_start_prob=0.0),
        num_envs=num_envs,
        env_dt=0.1,
        device=torch.device("cpu"),
        motion_lib=motion_lib,
    )


def _write_heldout(tmp_path, ids):
    p = tmp_path / "pack.pt.heldout.json"
    p.write_text(json.dumps({"ids": list(ids)}))
    return str(p)


def test_manager_unset_is_byte_identical(monkeypatch):
    monkeypatch.delenv("PM_HELDOUT_FILE", raising=False)
    ml = _MotionLib(n=100)
    before = ml.motion_weights.clone()
    mgr = _manager(ml)
    assert mgr.heldout_motion_ids is None
    assert torch.equal(mgr.motion_weights, before)


def test_manager_masks_and_never_samples_heldout(monkeypatch, tmp_path):
    ids = [2, 5, 7]  # 3/100 -> 3% masked mass (< 10% safety bound)
    monkeypatch.setenv("PM_HELDOUT_FILE", _write_heldout(tmp_path, ids))
    ml = _MotionLib(n=100)
    mgr = _manager(ml)

    assert mgr.heldout_motion_ids.tolist() == ids
    assert torch.all(mgr.motion_weights[torch.tensor(ids)] == 0.0)
    # renormalized back to the pre-mask total mass (100 * 0.1 = 10.0)
    assert abs(float(mgr.motion_weights.sum()) - 10.0) < 1e-5

    draws = mgr.sample_n_motion_ids(10_000)
    drawn = set(draws.tolist())
    assert drawn.isdisjoint(set(ids))
    assert drawn.issubset(set(range(100)) - set(ids))


def test_manager_curriculum_update_cannot_resurrect_heldout(monkeypatch, tmp_path):
    ids = [2, 5, 7]
    monkeypatch.setenv("PM_HELDOUT_FILE", _write_heldout(tmp_path, ids))
    ml = _MotionLib(n=100)
    mgr = _manager(ml)
    # An eval-curriculum rewrite that tries to give held-out clips full weight
    # must be re-masked (update_sampling_weights re-applies the held-out mask).
    resurrect = torch.ones(100)
    mgr.update_sampling_weights(resurrect)
    assert torch.all(mgr.motion_weights[torch.tensor(ids)] == 0.0)


def test_manager_resume_load_state_reapplies_mask(monkeypatch, tmp_path):
    ids = [2, 5, 7]
    monkeypatch.setenv("PM_HELDOUT_FILE", _write_heldout(tmp_path, ids))
    ml = _MotionLib(n=100)
    mgr = _manager(ml)
    # Checkpoint predating the mask: nonzero weight for every clip.
    ckpt = {"motion_file_name": "pack.pt", "motion_weights": torch.ones(100)}
    mgr.load_state_dict(ckpt)
    assert torch.all(mgr.motion_weights[torch.tensor(ids)] == 0.0)


def test_manager_masks_sharded_via_global_ids(monkeypatch, tmp_path):
    # rank owns the 40 EVEN globals [0,2,..,78] as locals [0..39]; hold out
    # globals 4 & 10 (locals 2 & 5) -> 2/40 = 5% masked mass. Global 99 lives on
    # another rank and is dropped here.
    gids = torch.arange(0, 80, 2)
    monkeypatch.setenv("PM_HELDOUT_FILE", _write_heldout(tmp_path, [4, 10, 99]))
    ml = _MotionLib(n=40, sharded=True, global_ids=gids)
    mgr = _manager(ml, num_envs=2)
    assert mgr.heldout_motion_ids.tolist() == [2, 5]
    assert torch.all(mgr.motion_weights[torch.tensor([2, 5])] == 0.0)


def test_manager_refuses_oversized_mask(monkeypatch, tmp_path):
    # Holding out most of the pack -> masked mass >= 0.10 -> refuse (bad artifact)
    monkeypatch.setenv("PM_HELDOUT_FILE", _write_heldout(tmp_path, list(range(15))))
    ml = _MotionLib(n=20)
    with pytest.raises(AssertionError, match="masked sampling mass fraction"):
        _manager(ml)


# ---------------------------------------------------------------- evaluator


def _stub_evaluator(heldout_local, *, subset_n_env, monkeypatch, n_local=40):
    """Build a MimicEvaluator without its heavy __init__, wired with just the
    stubs the held-out selection code path touches."""
    ev = MimicEvaluator.__new__(MimicEvaluator)
    mm = SimpleNamespace(
        heldout_motion_ids=(
            None if heldout_local is None else torch.tensor(heldout_local)
        ),
        motion_ids=torch.zeros(64, dtype=torch.long),
        motion_times=torch.zeros(64, dtype=torch.float),
    )
    ml = SimpleNamespace(
        motion_lengths=torch.arange(1, n_local + 1, dtype=torch.float),
        get_motion_length=lambda mids: (
            torch.arange(1, n_local + 1, dtype=torch.float)
            if mids is None
            else torch.arange(1, n_local + 1, dtype=torch.float)[mids]
        ),
        num_motions=lambda: n_local,
    )
    env = SimpleNamespace(dt=1.0, save_state=lambda: "snap", motion_manager=mm)
    ev.agent = SimpleNamespace(env=env, motion_lib=ml)
    ev.fabric = SimpleNamespace(global_rank=0, device=torch.device("cpu"))
    ev.config = SimpleNamespace(max_eval_steps=100)
    ev._captured_buf_n = None
    ev._init_eval_component_buffers = lambda n: setattr(ev, "_captured_buf_n", n)
    ev._create_metrics = lambda n, f, m: {"num_motions": n, "frames": f}
    if subset_n_env is None:
        monkeypatch.delenv("EVAL_SUBSET_N", raising=False)
    else:
        monkeypatch.setenv("EVAL_SUBSET_N", str(subset_n_env))
    return ev, mm


def test_evaluator_unset_uses_subset_path(monkeypatch):
    monkeypatch.delenv("PM_HELDOUT_FILE", raising=False)
    ev, _ = _stub_evaluator(None, subset_n_env=8, monkeypatch=monkeypatch, n_local=40)
    metrics = ev.initialize_eval()
    assert ev._heldout_track_ids is None
    assert metrics["num_motions"] == 8  # min(EVAL_SUBSET_N, n_local)
    # _episode_track_ids is identity when not held-out
    mids = torch.arange(8)
    assert torch.equal(ev._episode_track_ids(mids), mids)


def test_evaluator_selects_only_heldout_clips(monkeypatch, tmp_path):
    monkeypatch.setenv("PM_HELDOUT_FILE", str(tmp_path / "any.json"))
    heldout = [3, 8, 15, 22, 31]
    ev, mm = _stub_evaluator(
        heldout, subset_n_env=None, monkeypatch=monkeypatch, n_local=40
    )
    metrics = ev.initialize_eval()
    k = len(heldout)
    assert metrics["num_motions"] == k
    assert ev._captured_buf_n == k
    assert ev._heldout_track_ids.tolist() == heldout

    # one compact batch [0, K); tracked clips == exactly the held-out ids
    batches = ev._build_eval_batches()
    assert len(batches) == 1
    env_ids, motion_ids = batches[0]
    assert motion_ids.tolist() == list(range(k))
    track_ids = ev._episode_track_ids(motion_ids)
    assert track_ids.tolist() == heldout

    # _on_episode_start writes the ACTUAL held-out clips into the manager, and
    # never a trained clip.
    ev._episode_ctx = SimpleNamespace(track_ids=track_ids)
    ev._on_episode_start(env_ids)
    assert set(mm.motion_ids[env_ids].tolist()) == set(heldout)
    assert set(mm.motion_ids[env_ids].tolist()).issubset(set(range(40)))


def test_evaluator_caps_heldout_at_subset_n(monkeypatch, tmp_path):
    monkeypatch.setenv("PM_HELDOUT_FILE", str(tmp_path / "any.json"))
    heldout = [3, 8, 15, 22, 31]
    ev, _ = _stub_evaluator(
        heldout, subset_n_env=3, monkeypatch=monkeypatch, n_local=40
    )
    metrics = ev.initialize_eval()
    assert metrics["num_motions"] == 3  # capped at EVAL_SUBSET_N
    assert ev._heldout_track_ids.tolist() == heldout[:3]
