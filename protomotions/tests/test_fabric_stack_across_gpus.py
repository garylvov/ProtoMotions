# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only construction tests for the PM_STACK_ACROSS_GPUS fabric strategy.

TRUE cross-GPU rank-stacking (imprint #116/#117): one DDP process group spanning
N physical GPUs with S ranks pinned to each GPU. These tests exercise only the
pure strategy construction in ``_default_ddp_strategy`` / ``FabricConfig`` — no
CUDA runtime is touched (``torch.device("cuda", g)`` objects construct fine on a
CPU-only box), so nothing here is skipped for lack of a GPU.
"""

import pytest

from protomotions.utils.fabric_config import _default_ddp_strategy, FabricConfig


# Every stacking env var this module toggles. Cleared before each test so the
# host environment (a real launcher may export these) can never leak in.
_STACK_ENVS = (
    "PM_STACK_ACROSS_GPUS",
    "PM_STACK_RANKS_ON_GPU0",
    "PM_STACK_NRANKS",
    "PM_NGPU",
    "CUDA_VISIBLE_DEVICES",
)


@pytest.fixture(autouse=True)
def _clear_stack_env(monkeypatch):
    for key in _STACK_ENVS:
        monkeypatch.delenv(key, raising=False)


def _device_pairs(strategy):
    return [(d.type, d.index) for d in strategy._parallel_devices]


def test_across_gpus_pins_s_copies_of_each_of_n_devices(monkeypatch):
    monkeypatch.setenv("PM_STACK_ACROSS_GPUS", "1")
    monkeypatch.setenv("PM_STACK_NRANKS", "2")
    monkeypatch.setenv("PM_NGPU", "3")

    strategy = _default_ddp_strategy()

    # parallel_devices = [cuda:0]*S + [cuda:1]*S + ... in GPU-major order.
    assert _device_pairs(strategy) == [
        ("cuda", 0), ("cuda", 0),
        ("cuda", 1), ("cuda", 1),
        ("cuda", 2), ("cuda", 2),
    ]
    # world_size == N * S.
    assert len(strategy._parallel_devices) == 6
    # gloo is mandatory when S > 1 (NCCL rejects co-located ranks).
    assert strategy._process_group_backend == "gloo"


def test_across_gpus_world_size_is_n_times_s(monkeypatch):
    monkeypatch.setenv("PM_STACK_ACROSS_GPUS", "1")
    monkeypatch.setenv("PM_STACK_NRANKS", "4")
    monkeypatch.setenv("PM_NGPU", "8")

    strategy = _default_ddp_strategy()

    assert len(strategy._parallel_devices) == 32
    # Ordering: each GPU index appears in a contiguous run of length S.
    for gpu in range(8):
        run = strategy._parallel_devices[gpu * 4:(gpu + 1) * 4]
        assert [d.index for d in run] == [gpu, gpu, gpu, gpu]


def test_across_gpus_n_derived_from_cuda_visible_devices(monkeypatch):
    # PM_NGPU absent -> N = count of CUDA_VISIBLE_DEVICES entries.
    monkeypatch.setenv("PM_STACK_ACROSS_GPUS", "1")
    monkeypatch.setenv("PM_STACK_NRANKS", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,3,5")

    strategy = _default_ddp_strategy()

    # 3 visible GPUs (indices are the LOCAL cuda ordinals 0..N-1, not the CVD
    # values, because CUDA_VISIBLE_DEVICES masks the process's view).
    assert [d.index for d in strategy._parallel_devices] == [0, 0, 1, 1, 2, 2]
    assert strategy._process_group_backend == "gloo"


def test_s_equals_one_falls_through_to_default_nccl(monkeypatch):
    monkeypatch.setenv("PM_STACK_ACROSS_GPUS", "1")
    monkeypatch.setenv("PM_STACK_NRANKS", "1")
    monkeypatch.setenv("PM_NGPU", "4")

    strategy = _default_ddp_strategy()

    # S == 1 is plain one-rank-per-GPU cross-GPU DDP: no parallel_devices
    # pinning, default (NCCL) backend.
    assert strategy._parallel_devices is None
    assert strategy._process_group_backend is None


def test_default_strategy_when_mode_off():
    strategy = _default_ddp_strategy()
    assert strategy._parallel_devices is None
    assert strategy._process_group_backend is None


def test_fabric_config_coerces_devices_to_auto_when_stacked(monkeypatch):
    # With S > 1, `devices` (= args.ngpu = N*S) would fail CUDAAccelerator's
    # parse_devices against only N visible GPUs, so __post_init__ coerces it to
    # "auto" while the strategy's parallel_devices keeps world_size = N*S.
    monkeypatch.setenv("PM_STACK_ACROSS_GPUS", "1")
    monkeypatch.setenv("PM_STACK_NRANKS", "2")
    monkeypatch.setenv("PM_NGPU", "3")

    config = FabricConfig(devices=6, strategy=None)
    assert config.devices == "auto"


def test_fabric_config_keeps_devices_when_s_equals_one(monkeypatch):
    monkeypatch.setenv("PM_STACK_ACROSS_GPUS", "1")
    monkeypatch.setenv("PM_STACK_NRANKS", "1")
    monkeypatch.setenv("PM_NGPU", "4")

    # S == 1: devices == visible GPU count, parses cleanly -> not coerced.
    config = FabricConfig(devices=4, strategy=None)
    assert config.devices == 4


def test_fabric_config_default_devices_untouched_when_mode_off():
    config = FabricConfig(devices=2, strategy=None)
    assert config.devices == 2
