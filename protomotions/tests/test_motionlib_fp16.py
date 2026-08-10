# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the PM_MOTIONLIB_FP16 pose-tensor quantization path.

PM_MOTIONLIB_FP16=1 stores the big static per-frame pose tensors
(gts/grs/gvs/gavs/dps/dvs) as fp16 to roughly halve their VRAM footprint;
everything timing/indexing/cumulative (motion_dt, length_starts,
motion_num_frames, motion_lengths, motion_weights, contacts) is untouched;
quaternions (grs) are renormalized post-cast; and the read path
(get_motion_state_exact_frame) upcasts back to fp32. All exercised on CPU with a
tiny synthetic pack — no GPU required.
"""

import numpy as np
import pytest
import torch

from protomotions.components.motion_lib import (
    MotionLib,
    MotionLibConfig,
    _FP16_MANTISSA_BITS,
    _FP16_QUANTIZE_FIELDS,
)

_FP16_MANTISSA_ULP_SCALE = 2.0 ** -_FP16_MANTISSA_BITS

# Fields quantized to fp16; contacts/weights/indexing fields must stay put.
_UNTOUCHED_FIELDS = ("contacts", "motion_weights", "motion_num_frames",
                     "motion_dt", "motion_lengths", "length_starts")


def _synth_clip(num_frames, num_bodies=4, num_dofs=3, seed=0):
    rng = np.random.default_rng(seed)
    grs = rng.standard_normal((num_frames, num_bodies, 4)).astype(np.float32)
    grs /= np.linalg.norm(grs, axis=-1, keepdims=True)
    return {
        "gts": rng.standard_normal((num_frames, num_bodies, 3)).astype(np.float32),
        "grs": grs,
        "gvs": rng.standard_normal((num_frames, num_bodies, 3)).astype(np.float32),
        "gavs": rng.standard_normal((num_frames, num_bodies, 3)).astype(np.float32),
        "dps": rng.standard_normal((num_frames, num_dofs)).astype(np.float32),
        "dvs": rng.standard_normal((num_frames, num_dofs)).astype(np.float32),
        "contacts": (rng.random((num_frames, num_bodies)) > 0.5),
    }


_FRAME_FIELDS = ("gts", "grs", "gvs", "gavs", "dps", "dvs", "contacts")


def _write_pack(path, clips, dt=1.0 / 30.0):
    """Write a minimal packaged MotionLib .pt (mirrors pack_io.PackWriter)."""
    chunks = {k: [c[k] for c in clips] for k in _FRAME_FIELDS}
    save = {}
    for k in _FRAME_FIELDS:
        arr = np.ascontiguousarray(np.concatenate(chunks[k], axis=0))
        t = torch.from_numpy(arr)
        save[k] = t if k == "contacts" else t.float()
    nf = torch.tensor([c["gts"].shape[0] for c in clips], dtype=torch.long)
    save["motion_num_frames"] = nf
    shifted = nf.roll(1).clone()
    shifted[0] = 0
    save["length_starts"] = shifted.cumsum(0)
    save["motion_dt"] = torch.full((len(clips),), dt, dtype=torch.float32)
    save["motion_lengths"] = (nf.float() - 1) * save["motion_dt"]
    save["motion_weights"] = torch.ones(len(clips), dtype=torch.float32)
    save["motion_files"] = tuple(f"amass/clip{i}" for i in range(len(clips)))
    torch.save(save, path)
    return save


@pytest.fixture
def mini_pack(tmp_path):
    clips = [_synth_clip(5 + i, seed=i) for i in range(3)]
    path = tmp_path / "mini.pt"
    fp32 = _write_pack(str(path), clips)
    return str(path), fp32


def test_gate_on_pose_tensors_become_fp16(monkeypatch, mini_pack):
    path, _ = mini_pack
    monkeypatch.setenv("PM_MOTIONLIB_FP16", "1")

    ml = MotionLib(MotionLibConfig(motion_file=path), device="cpu")

    for field in _FP16_QUANTIZE_FIELDS:
        assert getattr(ml, field).dtype == torch.float16, field


def test_gate_on_leaves_indexing_and_contacts_untouched(monkeypatch, mini_pack):
    path, fp32 = mini_pack
    monkeypatch.setenv("PM_MOTIONLIB_FP16", "1")

    ml = MotionLib(MotionLibConfig(motion_file=path), device="cpu")

    assert ml.contacts.dtype == torch.bool
    assert ml.motion_weights.dtype == torch.float32
    assert ml.motion_num_frames.dtype == torch.long
    assert ml.motion_dt.dtype == torch.float32
    # values of the untouched metadata match what we wrote.
    assert torch.equal(ml.motion_num_frames, fp32["motion_num_frames"])
    assert torch.allclose(ml.motion_dt, fp32["motion_dt"])


def test_gate_on_values_allclose_within_fp16_eps(monkeypatch, mini_pack):
    path, fp32 = mini_pack
    monkeypatch.setenv("PM_MOTIONLIB_FP16", "1")

    ml = MotionLib(MotionLibConfig(motion_file=path), device="cpu")

    for field in ("gts", "gvs", "gavs", "dps", "dvs"):
        orig = fp32[field]
        got = getattr(ml, field).float()
        # fp16 has ~1e-3 relative precision; use a matching tolerance.
        assert torch.allclose(got, orig, rtol=2e-3, atol=1e-3), field


def test_gate_on_quaternions_renormalized_after_cast(monkeypatch, mini_pack):
    path, _ = mini_pack
    monkeypatch.setenv("PM_MOTIONLIB_FP16", "1")

    ml = MotionLib(MotionLibConfig(motion_file=path), device="cpu")

    norms = ml.grs.float().norm(dim=-1)
    # fp16 rounding can nudge ||q|| off 1.0; renorm keeps it within fp16 eps.
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_gate_on_read_path_upcasts_to_fp32(monkeypatch, mini_pack):
    path, _ = mini_pack
    monkeypatch.setenv("PM_MOTIONLIB_FP16", "1")

    ml = MotionLib(MotionLibConfig(motion_file=path), device="cpu")
    state = ml.get_motion_state_exact_frame(
        torch.tensor([0, 1]), torch.tensor([0, 0])
    )

    # Never keep fp16 past sampling.
    assert state.rigid_body_pos.dtype == torch.float32
    assert state.rigid_body_rot.dtype == torch.float32
    assert state.dof_pos.dtype == torch.float32


def test_gate_off_keeps_fp32(monkeypatch, mini_pack):
    path, fp32 = mini_pack
    monkeypatch.delenv("PM_MOTIONLIB_FP16", raising=False)

    ml = MotionLib(MotionLibConfig(motion_file=path), device="cpu")

    for field in _FP16_QUANTIZE_FIELDS:
        assert getattr(ml, field).dtype == torch.float32, field
    # bit-exact with what we wrote (no quantization at all).
    assert torch.equal(ml.gts, fp32["gts"])
    assert torch.equal(ml.grs, fp32["grs"])


# ---------------------------------------------------------------------------
# fp16 x ABSOLUTE world magnitude safety net (GTS_CONSISTENCY_AUDIT 2026-08-06)
#
# gts is cast at absolute magnitude, BEFORE the per-env re-origining at reset,
# so fp16's relative ulp (2^-10) scales the error with the coordinate: ~1 mm at
# 1 m but ~1 m at 1 km. 640 clips (0.89%) of the live corpus sit beyond 10 m.
# ---------------------------------------------------------------------------


@pytest.fixture
def far_origin_pack(tmp_path):
    """Same mini pack, translated 1.3 km in +x (the LocoMuJoCo walk_chunk case)."""
    clips = [_synth_clip(5 + i, seed=i) for i in range(3)]
    for c in clips:
        c["gts"][:, :, 0] += 1300.0
    path = tmp_path / "far.pt"
    fp32 = _write_pack(str(path), clips)
    return str(path), fp32


def test_ulp_table_matches_the_documented_derivation():
    """fp16 grid spacing at |x| is in [|x|*2^-11, |x|*2^-10]; that bound is the
    whole basis for the 10 m threshold and for the message's ulp figure."""
    for magnitude in (1.0, 10.0, 128.0, 1024.0, 1300.0):
        x = torch.tensor([magnitude], dtype=torch.float32)
        nxt = torch.nextafter(x.half(), torch.tensor([float("inf")]).half())
        ulp = float(nxt.float() - x.half().float())
        assert magnitude * _FP16_MANTISSA_ULP_SCALE / 2.0 <= ulp <= \
            magnitude * _FP16_MANTISSA_ULP_SCALE * 1.001, magnitude
    # exact powers of two hit the upper bound the code reports.
    for magnitude in (1.0, 1024.0):
        x = torch.tensor([magnitude], dtype=torch.float32)
        nxt = torch.nextafter(x.half(), torch.tensor([float("inf")]).half())
        assert float(nxt.float() - x.half().float()) == pytest.approx(
            magnitude * _FP16_MANTISSA_ULP_SCALE)
    # 1 km -> a ~1 m grid: the whole 29-body skeleton collapses onto one cell.
    assert 1300.0 * _FP16_MANTISSA_ULP_SCALE > 1.0
    # 10 m -> ~1e-2 m upper bound, ~4e-3 m half-ulp: the measured knee.
    assert 10.0 * _FP16_MANTISSA_ULP_SCALE < 1.1e-2


def test_near_origin_pack_passes_the_magnitude_check(monkeypatch, mini_pack,
                                                     capsys):
    path, _ = mini_pack
    monkeypatch.setenv("PM_MOTIONLIB_FP16", "1")
    monkeypatch.delenv("PM_MOTIONLIB_FP16_STRICT", raising=False)

    MotionLib(MotionLibConfig(motion_file=path), device="cpu")

    out = capsys.readouterr().out
    assert "world-magnitude check OK" in out
    assert "UNSAFE WORLD MAGNITUDE" not in out


def test_far_origin_pack_warns_loudly_but_still_loads(monkeypatch,
                                                      far_origin_pack, capsys):
    """Default = WARN: v58's live corpus has far-origin clips and must load."""
    path, _ = far_origin_pack
    monkeypatch.setenv("PM_MOTIONLIB_FP16", "1")
    monkeypatch.delenv("PM_MOTIONLIB_FP16_STRICT", raising=False)

    ml = MotionLib(MotionLibConfig(motion_file=path), device="cpu")

    out = capsys.readouterr().out
    assert "UNSAFE WORLD MAGNITUDE" in out
    assert "--reorigin-xy" in out
    assert ml.gts.dtype == torch.float16  # still quantized; warning only


def test_far_origin_pack_hard_errors_under_strict(monkeypatch,
                                                  far_origin_pack):
    path, _ = far_origin_pack
    monkeypatch.setenv("PM_MOTIONLIB_FP16", "1")
    monkeypatch.setenv("PM_MOTIONLIB_FP16_STRICT", "1")

    with pytest.raises(ValueError, match="UNSAFE WORLD MAGNITUDE"):
        MotionLib(MotionLibConfig(motion_file=path), device="cpu")


def test_check_is_silent_and_value_preserving_when_fp16_is_off(
        monkeypatch, far_origin_pack, capsys):
    """Rule 10: unset PM_MOTIONLIB_FP16 -> byte-identical, no new output."""
    path, fp32 = far_origin_pack
    monkeypatch.delenv("PM_MOTIONLIB_FP16", raising=False)
    monkeypatch.setenv("PM_MOTIONLIB_FP16_STRICT", "1")  # must not fire

    ml = MotionLib(MotionLibConfig(motion_file=path), device="cpu")

    out = capsys.readouterr().out
    assert "world-magnitude check" not in out
    assert torch.equal(ml.gts, fp32["gts"])


def test_threshold_is_configurable(monkeypatch, mini_pack, capsys):
    path, _ = mini_pack
    monkeypatch.setenv("PM_MOTIONLIB_FP16", "1")
    monkeypatch.setenv("PM_MOTIONLIB_FP16_MAX_COORD_M", "0.001")

    MotionLib(MotionLibConfig(motion_file=path), device="cpu")

    assert "UNSAFE WORLD MAGNITUDE" in capsys.readouterr().out
