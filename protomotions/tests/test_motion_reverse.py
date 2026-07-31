# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for online time-reversal augmentation (motion_reverse.py).

Run standalone (CPU):
    PYTHONPATH=. python protomotions/tests/test_motion_reverse.py

Covers (all through the REAL MotionLib.get_motion_state code path, via a
synthetic in-memory motion pack):
  A. double-reverse == identity (time remap + velocity negation are both
     involutions; fetch-level: reversed fetch at L - t reproduces the forward
     fetch at t with velocities negated).
  B. reversed velocities == -flip(original) across the whole dense timeline.
  C. reversed walking clip: root displacement over playback is NEGATED relative
     to the (unchanged) heading -- walking forward becomes walking backward.
  D. prob=0 / flags-off byte-identity: reverse_flags=None and all-False flags
     both bit-match the plain fetch; MotionManager PM_REVERSE_PROB env gate
     (unset -> off, set -> coins drawn, bad value -> assert).
  E. mirror+reverse composition: fetch(mirror&reverse) == mirror(fetch(reverse))
     == reverse-velocities(fetch(mirror)) -- the two transforms commute and the
     coins are independent.
  F. future-window composition: playback time t + k*dt on a reversed clip
     fetches original time L - t - k*dt (future targets walk the original clip
     backwards); past-the-end future lookups hold the reversed clip's final
     frame (== original first frame).
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from protomotions.components.motion_lib import MotionLib
from protomotions.components.motion_reverse import (
    reverse_motion_times,
    reverse_robot_state_velocities,
)

torch.manual_seed(0)
ATOL = 1e-6

FIELDS = (
    "rigid_body_pos",
    "rigid_body_rot",
    "rigid_body_vel",
    "rigid_body_ang_vel",
    "dof_pos",
    "dof_vel",
    "rigid_body_contacts",
)
VEL_FIELDS = ("rigid_body_vel", "rigid_body_ang_vel", "dof_vel")


def make_fake_lib(num_bodies=8, num_dofs=12, frames=(33, 17), fps=32.0):
    # Default frames/fps chosen so frame times are exact in binary fp
    # (num_frames-1 and fps are powers of two): exact-frame fetches then hit
    # blend==0 identically on the forward and reversed timelines, making the
    # flip() comparisons in test_B/test_F bitwise-clean.
    """Minimal in-memory MotionLib exercising the real get_motion_state path."""
    lib = MotionLib.__new__(MotionLib)
    total = sum(frames)
    lib.device = torch.device("cpu")
    lib.motion_num_frames = torch.tensor(frames, dtype=torch.long)
    lib.motion_dt = torch.full((len(frames),), 1.0 / fps)
    lib.motion_lengths = (lib.motion_num_frames - 1).float() * lib.motion_dt
    starts = [0]
    for f in frames[:-1]:
        starts.append(starts[-1] + f)
    lib.length_starts = torch.tensor(starts, dtype=torch.long)
    lib.gts = torch.randn(total, num_bodies, 3)
    q = torch.randn(total, num_bodies, 4)
    lib.grs = q / q.norm(dim=-1, keepdim=True)
    lib.gvs = torch.randn(total, num_bodies, 3)
    lib.gavs = torch.randn(total, num_bodies, 3)
    lib.dps = torch.randn(total, num_dofs)
    lib.dvs = torch.randn(total, num_dofs)
    lib.lrs = None
    lib.contacts = torch.rand(total, num_bodies)  # float (smoothed) contacts
    return lib


def _cmp(sa, sb, negate_vels=False, atol=ATOL, label=""):
    worst = 0.0
    for f in FIELDS:
        a, b = getattr(sa, f, None), getattr(sb, f, None)
        if a is None or b is None:
            assert a is None and b is None, f"{label}: field {f} presence mismatch"
            continue
        if negate_vels and f in VEL_FIELDS:
            b = -b
        e = (a - b).abs().max().item()
        worst = max(worst, e)
        assert e < atol, f"{label}: field {f} mismatch (err {e:.2e})"
    return worst


def frame_times(lib, mid):
    n = int(lib.motion_num_frames[mid])
    return torch.arange(n, dtype=torch.float32) * float(lib.motion_dt[mid])


def test_A_double_reverse(lib):
    # primitive involutions
    mid = torch.zeros(16, dtype=torch.long)
    t = torch.rand(16) * lib.motion_lengths[mid]
    flags = torch.ones(16, dtype=torch.bool)
    L = lib.motion_lengths[mid]
    t2 = reverse_motion_times(reverse_motion_times(t, L, flags), L, flags)
    assert (t2 - t).abs().max().item() < ATOL, "time remap not an involution"
    s = lib.get_motion_state(mid, t.clone())
    s2 = lib.get_motion_state(mid, t.clone())
    reverse_robot_state_velocities(s2, flags)
    reverse_robot_state_velocities(s2, flags)
    _cmp(s, s2, label="[A] velocity negation involution")
    # fetch-level: reversed clip evaluated at L - t == forward clip at t with
    # velocities negated (i.e. reversing the reversed timeline is the original).
    s_fwd = lib.get_motion_state(mid, t.clone())
    s_rev = lib.get_motion_state(mid, (L - t).clone(), reverse_flags=flags)
    _cmp(s_fwd, s_rev, negate_vels=True, atol=1e-4,
         label="[A] rev(L-t) vs fwd(t)")
    print("[A] PASS: double-reverse == identity (primitives + fetch level)")


def test_B_velocity_flip(lib):
    for mid_i in range(len(lib.motion_num_frames)):
        ts = frame_times(lib, mid_i)
        n = ts.numel()
        mid = torch.full((n,), mid_i, dtype=torch.long)
        flags = torch.ones(n, dtype=torch.bool)
        fwd = lib.get_motion_state(mid, ts.clone())
        rev = lib.get_motion_state(mid, ts.clone(), reverse_flags=flags)
        for f in FIELDS:
            a = getattr(fwd, f, None)
            if a is None:
                continue
            b = getattr(rev, f)
            exp = a.flip(0)
            if f in VEL_FIELDS:
                exp = -exp
            e = (b - exp).abs().max().item()
            assert e < 1e-5, f"[B] motion {mid_i} field {f}: err {e:.2e}"
    print("[B] PASS: reversed timeline == flip(frames); velocities == -flip(original)")


def test_C_walking_displacement(lib):
    """Synthetic constant-velocity +x walk: reversal negates root displacement
    relative to the unchanged heading, and the served root velocity is -v."""
    n_frames, fps, v = 33, 32.0, 1.3
    wlib = make_fake_lib(num_bodies=4, num_dofs=6, frames=(n_frames,), fps=fps)
    ts_all = torch.arange(n_frames, dtype=torch.float32) / fps
    wlib.gts[:] = 0.0
    wlib.gts[:, :, 0] = (ts_all * v).unsqueeze(-1)  # all bodies march +x
    wlib.grs[:] = torch.tensor([0.0, 0.0, 0.0, 1.0])  # identity heading (XYZW)
    wlib.gvs[:] = 0.0
    wlib.gvs[:, :, 0] = v
    L = float(wlib.motion_lengths[0])
    mid = torch.zeros(2, dtype=torch.long)
    flags = torch.ones(2, dtype=torch.bool)
    ends = torch.tensor([0.0, L])
    fwd = wlib.get_motion_state(mid, ends.clone())
    rev = wlib.get_motion_state(mid, ends.clone(), reverse_flags=flags)
    disp_fwd = fwd.rigid_body_pos[1, 0] - fwd.rigid_body_pos[0, 0]
    disp_rev = rev.rigid_body_pos[1, 0] - rev.rigid_body_pos[0, 0]
    e_disp = (disp_rev + disp_fwd).abs().max().item()
    # heading untouched by reversal
    e_rot = (rev.rigid_body_rot - fwd.rigid_body_rot.flip(0)).abs().max().item()
    # served velocity points backward relative to that heading
    e_vel = (rev.rigid_body_vel[:, 0, 0] + v).abs().max().item()
    print(f"[C] disp_fwd {disp_fwd.tolist()} disp_rev {disp_rev.tolist()} "
          f"(negation err {e_disp:.2e}); heading err {e_rot:.2e}; "
          f"root-vel==-v err {e_vel:.2e}")
    assert e_disp < 1e-5 and e_rot < 1e-6 and e_vel < 1e-6
    print("[C] PASS: walking forward becomes walking backward "
          "(displacement negated vs unchanged heading)")


def test_D_off_is_byte_identical(lib):
    mid = torch.zeros(32, dtype=torch.long)
    t = torch.rand(32) * lib.motion_lengths[mid]
    base = lib.get_motion_state(mid, t.clone())
    none_flags = lib.get_motion_state(mid, t.clone())  # reverse_flags default None
    false_flags = lib.get_motion_state(
        mid, t.clone(), reverse_flags=torch.zeros(32, dtype=torch.bool)
    )
    for f in FIELDS:
        a = getattr(base, f, None)
        if a is None:
            continue
        assert torch.equal(a, getattr(none_flags, f)), f"[D] None-flags differ: {f}"
        assert torch.equal(a, getattr(false_flags, f)), f"[D] False-flags differ: {f}"

    # MotionManager env gate: unset -> off; set -> live-read + coins drawn.
    from protomotions.envs.motion_manager.config import MotionManagerConfig
    from protomotions.envs.motion_manager.motion_manager import MotionManager

    lib.motion_weights = torch.ones(len(lib.motion_num_frames))
    lib.motion_file = "fake.pt"
    cfg = MotionManagerConfig(init_start_prob=0.0)
    saved = {
        k: os.environ.pop(k, None)
        for k in ("PM_REVERSE_PROB", "PM_MIRROR_PROB", "PM_HELDOUT_FILE")
    }
    try:
        mm = MotionManager(cfg, num_envs=64, env_dt=1.0 / 30.0,
                           device=torch.device("cpu"), motion_lib=lib)
        assert mm.reverse_prob == 0.0
        mm.sample_motions(torch.arange(64))
        assert not mm.reverse_flags.any(), "[D] flags drawn with prob unset"

        os.environ["PM_REVERSE_PROB"] = "1.0"
        mm = MotionManager(cfg, num_envs=64, env_dt=1.0 / 30.0,
                           device=torch.device("cpu"), motion_lib=lib)
        assert mm.reverse_prob == 1.0
        mm.sample_motions(torch.arange(64))
        assert mm.reverse_flags.all(), "[D] prob=1.0 must set every flag"

        os.environ["PM_REVERSE_PROB"] = "1.5"
        try:
            MotionManager(cfg, num_envs=4, env_dt=1.0 / 30.0,
                          device=torch.device("cpu"), motion_lib=lib)
            raise RuntimeError("[D] out-of-range prob must assert")
        except AssertionError:
            pass
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("[D] PASS: unset/False byte-identical; PM_REVERSE_PROB live-read; "
          "range-checked")


def test_E_mirror_compose():
    from protomotions.robot_configs.h1_2 import H1_2RobotConfig
    from protomotions.components.motion_mirror import (
        build_mirror_maps,
        mirror_robot_state,
    )

    ki = H1_2RobotConfig().kinematic_info
    maps = build_mirror_maps(ki.body_names, ki.dof_names, ki.hinge_axes_map,
                             w_last=True)
    lib = make_fake_lib(num_bodies=ki.num_bodies, num_dofs=ki.num_dofs,
                        frames=(20,), fps=30.0)
    lib.set_mirror_maps(maps)
    n = 24
    mid = torch.zeros(n, dtype=torch.long)
    t = torch.rand(n) * lib.motion_lengths[mid]
    mflags = torch.zeros(n, dtype=torch.bool); mflags[::2] = True
    rflags = torch.zeros(n, dtype=torch.bool); rflags[::3] = True

    both = lib.get_motion_state(mid, t.clone(), mirror_flags=mflags,
                                reverse_flags=rflags)
    # mirror( fetch(reverse) )
    a = lib.get_motion_state(mid, t.clone(), reverse_flags=rflags)
    mirror_robot_state(a, mflags, maps)
    _cmp(both, a, label="[E] mirror-after-reverse")
    # reverse-velocities( fetch(mirror) ): time remap applied for reversed rows
    # before the mirror-only fetch, then negate velocities.
    t_remap = reverse_motion_times(t.clone(), lib.motion_lengths[mid], rflags)
    b = lib.get_motion_state(mid, t_remap, mirror_flags=mflags)
    reverse_robot_state_velocities(b, rflags)
    _cmp(both, b, label="[E] reverse-after-mirror")
    print("[E] PASS: mirror(reverse) == reverse(mirror) == fetch(both flags), "
          "independent per-row coins")


def test_F_future_window(lib):
    """Future-target composition: playback t + k*dt on a reversed clip fetches
    original time L - t - k*dt; past-the-end lookups hold original frame 0."""
    mid_i = 0
    dt = float(lib.motion_dt[mid_i])
    L = float(lib.motion_lengths[mid_i])
    t0 = 5 * dt
    ks = torch.arange(0, 32, dtype=torch.float32)  # some k push past clip end
    fut = t0 + ks * dt
    n = fut.numel()
    mid = torch.full((n,), mid_i, dtype=torch.long)
    flags = torch.ones(n, dtype=torch.bool)
    rev = lib.get_motion_state(mid, fut.clone(), reverse_flags=flags)
    # expected: original clip at clamp(L - t0 - k*dt, 0), velocities negated
    exp_t = (L - fut).clamp(min=0.0)
    exp = lib.get_motion_state(mid, exp_t)
    _cmp(rev, exp, negate_vels=True, atol=1e-5, label="[F] future window")
    # explicit hold check: all past-the-end rows equal original frame 0 pose
    over = fut > L
    assert over.any()
    frame0 = lib.get_motion_state(mid[:1], torch.zeros(1))
    e = (rev.rigid_body_pos[over] - frame0.rigid_body_pos).abs().max().item()
    assert e < 1e-5, f"[F] past-end hold wrong (err {e:.2e})"
    print("[F] PASS: future lookups walk the original clip backwards; "
          "past-the-end holds the reversed clip's final frame")


if __name__ == "__main__":
    lib = make_fake_lib()
    test_A_double_reverse(lib)
    test_B_velocity_flip(lib)
    test_C_walking_displacement(lib)
    test_D_off_is_byte_identical(lib)
    test_E_mirror_compose()
    test_F_future_window(lib)
    print("\nALL TIME-REVERSAL CORRECTNESS TESTS PASSED")
