"""Offscreen MuJoCo render of a MASKED-MIMIC student ONNX (unified_pipeline)
on selected clips -> mp4. Teleop-style sparse conditioning (sparse3: head_aux
pos-only + both wrist_yaw pos+rot), mean-latent deterministic (baked into the
export). Reuses batch_mj_eval helpers (load_mujoco_model / set_initial_pose)
so no raw body-index math is re-derived here (campaign rule 8: MuJoCo bodies
are +1 offset from body_names; helpers already handle it).

Contract source: deployment/export_masked_mimic_onnx.py YAML `_runtime`.
Raw context tensors fed each control step (obs kernels are baked in the graph):
  current/noisy rigid body pos/rot/vel/ang_vel (world, xyzw), ground heights,
  masked_mimic ref_pos/ref_rot [1,S,nb,*] + masks + time_offsets,
  historical ring buffer (H past states, most-recent-first, per
  env.py reset_from_single_state semantics: index 0 = t-dt) + past raw actions.

Conditioning-time scheme mirrors MaskedMimicControl: absolute target_times,
shift+append when current time passes target_times[0]; fixed gap (--gap,
default 0.5 s, inside training's clamp [0.02, 2.0]).

Usage:
  python render_mm_eval.py --onnx unified_pipeline.onnx --yaml unified_pipeline.yaml \
      --motion teleop_val.pt --clips "1:squat,2:floorpick" --outdir ep3000/ \
      [--cond sparse3|dense] [--gap 0.5] [--res 640x480] [--max-seconds 20]
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import argparse, json, re, sys
from collections import deque
import numpy as np
import yaml as _yaml
import imageio.v2 as imageio

H2H = "/oscar/scratch/glvov/robojudo_eval/canonical_h2h"
sys.path.insert(0, H2H)
import mujoco
import onnxruntime as ort
import batch_mj_eval as B  # inserts GOLDEN on sys.path; load_mujoco_model, set_initial_pose
from deployment.motion_utils import MotionPlayer
from deployment.state_utils import mujoco_wxyz_to_xyzw


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


# sparse3 teleop conditioning (mm_canonical_v1_eval_loop.sh + masked_mimic_stiffv2.py):
# head_aux + wrists, constraint 1 (pos+rot); MM_POS_ONLY_BODIES remaps head_aux
# (non-wrist) to translation-only.
SPARSE3 = {"head_aux": (True, False),
           "left_wrist_yaw_link": (True, True),
           "right_wrist_yaw_link": (True, True)}


def build_body_mask(trackable, S, cond):
    """[S * num_cond * 2] float32; per step, per trackable body, (pos, rot)."""
    m = np.zeros((S, len(trackable), 2), dtype=np.float32)
    for bi, name in enumerate(trackable):
        if cond == "dense":
            m[:, bi, :] = 1.0
        elif name in SPARSE3:
            p, r = SPARSE3[name]
            m[:, bi, 0] = float(p)
            m[:, bi, 1] = float(r)
    return m.reshape(1, -1)


def read_full_state(model, data, nb):
    body_rot = mujoco_wxyz_to_xyzw(data.xquat[1:1 + nb].copy())
    body_rot[0] = mujoco_wxyz_to_xyzw(data.qpos[3:7].copy())  # root canonical
    body_pos = data.xpos[1:1 + nb].copy()
    vel = np.zeros((nb, 3)); ang = np.zeros((nb, 3))
    res = np.zeros(6)
    for i in range(nb):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, i + 1, res, 0)
        ang[i] = res[0:3]; vel[i] = res[3:6]
    return (body_pos.astype(np.float32), body_rot.astype(np.float32),
            vel.astype(np.float32), ang.astype(np.float32))


def render_clip(session, in_names, model, data, renderer, cam, player, meta, args):
    nb = meta["nb"]; ndofs = meta["ndofs"]; S = meta["S"]; Hs = meta["H"]
    decim = meta["decimation"]; control_dt = meta["control_dt"]
    body_mask = meta["body_mask"]

    B.set_initial_pose(model, data, player)
    mujoco.mj_forward(model, data)

    clip_len = player.total_frames * control_dt
    # absolute conditioning target times (training-faithful shift+append)
    tt = [min(args.gap * (i + 1), clip_len) for i in range(S)]

    pos0, rot0, vel0, ang0 = read_full_state(model, data, nb)
    hp = deque([pos0] * Hs, maxlen=Hs); hr = deque([rot0] * Hs, maxlen=Hs)
    hv = deque([vel0] * Hs, maxlen=Hs); ha = deque([ang0] * Hs, maxlen=Hs)
    hact = deque([np.zeros(ndofs, np.float32)] * Hs, maxlen=Hs)
    zeros_gh = np.zeros((1, Hs), np.float32)

    frames = []
    ema_prev = None            # EMA filter state, reset every episode (this fn = one episode)
    applied_stream = []        # applied PD target stream (post-filter), for jerk report
    root_stream = []           # root body world xyz per control step (pause-drift report)
    track_err_sum = 0.0; track_err_n = 0
    status = "full"
    N = min(player.total_frames, int(args.max_seconds / control_dt))
    for fi in range(N):
        t = fi * control_dt
        while t >= tt[0] - 1e-9 and tt[0] < clip_len - 1e-6:
            nxt = min(max(tt[-1] + args.gap, t + control_dt), clip_len)
            tt = tt[1:] + [nxt]

        pos, rot, vel, angv = read_full_state(model, data, nb)

        ref_pos = np.zeros((1, S, nb, 3), np.float32)
        ref_rot = np.zeros((1, S, nb, 4), np.float32)
        for s, ts in enumerate(tt):
            st = player.get_state_at_frame(int(round(ts / control_dt)))
            ref_pos[0, s] = st["body_pos"]; ref_rot[0, s] = st["body_rot"]
        time_off = np.array([tt], np.float32) - t

        # historical: most-recent-first (deque appends newest right -> reverse)
        H_pos = np.stack(list(hp)[::-1])[None]; H_rot = np.stack(list(hr)[::-1])[None]
        H_vel = np.stack(list(hv)[::-1])[None]; H_ang = np.stack(list(ha)[::-1])[None]
        H_act = np.stack(list(hact)[::-1])[None]

        feed = {
            "current_rigid_body_pos": pos[None], "current_rigid_body_rot": rot[None],
            "noisy_rigid_body_pos": pos[None], "noisy_rigid_body_rot": rot[None],
            "noisy_rigid_body_vel": vel[None], "noisy_rigid_body_ang_vel": angv[None],
            "noisy_ground_heights": np.zeros(1, np.float32),
            "historical_rigid_body_pos": H_pos, "historical_rigid_body_rot": H_rot,
            "historical_rigid_body_vel": H_vel, "historical_rigid_body_ang_vel": H_ang,
            "historical_actions": H_act, "historical_ground_heights": zeros_gh,
            "masked_mimic_ref_pos": ref_pos, "masked_mimic_ref_rot": ref_rot,
            "masked_mimic_target_bodies_masks": body_mask,
            "masked_mimic_target_poses_masks": np.ones((1, S), np.float32),
            "masked_mimic_time_offsets": time_off,
        }
        feed = {k: v.astype(np.float32) for k, v in feed.items() if k in in_names}
        missing = set(in_names) - set(feed)
        if missing:
            raise RuntimeError(f"unfed onnx inputs: {missing}")

        actions, pd = session.run(["actions", "joint_pos_targets"], feed)
        pd = pd.squeeze()
        # Optional action EMA (eval-side only). joint_pos_targets are affine in the
        # policy actions, and EMA weights sum to 1, so filtering the PD targets is
        # exactly filtered_action = a*new + (1-a)*prev applied to the actions
        # (matches deployment/test_tracker_mujoco.py action_ema_alpha semantics).
        # historical_actions still receives the RAW policy actions (export contract).
        if args.action_ema is not None:
            a = args.action_ema
            if ema_prev is None:
                ema_prev = pd.copy()
            pd = a * pd + (1.0 - a) * ema_prev
            ema_prev = pd.copy()
        applied_stream.append(pd.astype(np.float64).copy())
        data.ctrl[:] = pd
        for _ in range(decim):
            mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            print("[render] qpos non-finite, stopping clip", flush=True)
            status = "nonfinite"
            break
        root_stream.append(data.xpos[1].copy())
        ref = player.get_state_at_frame(fi)
        if "dof_pos" in ref:
            track_err_sum += float(np.mean(np.abs(data.qpos[7:] - ref["dof_pos"])))
            track_err_n += 1

        hp.append(pos); hr.append(rot); hv.append(vel); ha.append(angv)
        hact.append(actions.squeeze().astype(np.float32).copy())

        cam.lookat[:] = data.xpos[1]
        cam.lookat[2] = max(0.4, float(data.xpos[1][2]))
        renderer.update_scene(data, cam)
        frames.append(renderer.render().copy())
        if float(data.xpos[1][2]) < 0.25:
            print(f"[render] fall at t={t:.2f}s, stopping clip", flush=True)
            status = "fall"
            break

    # ---- jerk report on the FILTERED (applied) action stream ----
    stats = {"status": status, "steps": len(applied_stream), "planned_steps": N,
             "mean_abs_delta_deg": float("nan"), "norm_jerk": float("nan"),
             "mean_track_err_deg": (np.degrees(track_err_sum / track_err_n)
                                    if track_err_n else float("nan"))}
    if len(applied_stream) >= 3:
        A = np.stack(applied_stream)              # [T, ndofs] radians
        d1 = np.abs(np.diff(A, axis=0))           # |a_t - a_{t-1}|
        d2 = np.abs(np.diff(A, n=2, axis=0))      # |a_t - 2a_{t-1} + a_{t-2}|
        stats["mean_abs_delta_deg"] = float(np.degrees(d1.mean()))
        # normalized jerk proxy: second-difference magnitude relative to
        # first-difference magnitude (dimensionless; 0 = perfectly smooth ramp)
        stats["norm_jerk"] = float(d2.mean() / (d1.mean() + 1e-12))
    # ---- per-pause-window stability (drift/creep) ----
    # windows are (t0, t1) seconds in MOTION time; the rollout runs at control_dt
    # so frame index = round(t / control_dt) -- same mapping used for the refs above.
    wins = meta.get("pause_windows", {}).get(meta["clip_idx"], [])
    if wins and root_stream:
        R = np.stack(root_stream)                       # [T,3] world root xyz
        stats["pause_windows"] = []
        for (t0, t1) in wins:
            i0 = max(0, int(round(t0 / control_dt))); i1 = min(len(R), int(round(t1 / control_dt)))
            if i1 - i0 < 5:
                continue
            seg = R[i0:i1]
            xy = seg[:, :2]
            w = {"t0": round(float(t0), 2), "t1": round(float(t1), 2),
                 "covered": bool(i1 - i0 == int(round((t1 - t0) / control_dt))),
                 # net displacement of the root over the pause = creep
                 "xy_drift_m": float(np.linalg.norm(xy[-1] - xy[0])),
                 # worst excursion from where the pause started = wobble amplitude
                 "xy_max_excursion_m": float(np.max(np.linalg.norm(xy - xy[0], axis=1))),
                 "z_start_m": float(seg[0, 2]), "z_min_m": float(seg[:, 2].min()),
                 "z_drop_m": float(seg[0, 2] - seg[:, 2].min())}
            if i1 - i0 >= 3:
                da = np.abs(np.diff(np.stack(applied_stream[i0:i1]), axis=0))
                w["mean_abs_delta_deg"] = float(np.degrees(da.mean()))
            stats["pause_windows"].append(w)
    return frames, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True); ap.add_argument("--yaml", required=True)
    ap.add_argument("--motion", required=True); ap.add_argument("--clips", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cond", default="sparse3", choices=["sparse3", "dense"])
    ap.add_argument("--gap", type=float, default=0.5)
    ap.add_argument("--res", default="640x480")
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--action-ema", type=float, default=None,
                    help="EMA alpha on applied actions: filt = a*new + (1-a)*prev. "
                         "Omit for current (unfiltered) behavior.")
    ap.add_argument("--pause-windows", default=None,
                    help='JSON {"<clip_idx>": [[t0,t1], ...]} of pause/hold segments '
                         "(seconds, motion time). Per-window root drift/creep is reported.")
    ap.add_argument("--stats-json", default=None,
                    help="Write the full per-clip stats dict to this JSON path.")
    ap.add_argument("--no-video", action="store_true",
                    help="Run rollouts + jerk report only, skip mp4 writing.")
    args = ap.parse_args()
    if args.action_ema is not None and not (0.0 < args.action_ema <= 1.0):
        ap.error("--action-ema must be in (0, 1]")
    W, Ht = (int(x) for x in args.res.split("x"))
    os.makedirs(args.outdir, exist_ok=True)

    y = _yaml.safe_load(open(args.yaml))
    robot = y["robot"]; timing = y["timing"]; ctrl = y["control"]; mm = y["masked_mimic"]
    meta = dict(
        nb=int(robot["num_bodies"]), ndofs=int(robot["num_dofs"]),
        S=int(mm["num_masked_future_steps"]), H=int(mm["num_history_steps"]),
        decimation=int(timing["decimation"]), control_dt=float(timing["control_dt"]),
        body_mask=build_body_mask(mm["trackable_bodies_subset"],
                                  int(mm["num_masked_future_steps"]), args.cond),
        pause_windows=({int(k): v for k, v in json.loads(open(args.pause_windows).read()
                                                         if os.path.exists(args.pause_windows)
                                                         else args.pause_windows).items()}
                       if args.pause_windows else {}),
        clip_idx=-1,
    )
    fps = int(round(1.0 / meta["control_dt"]))  # render every control step, real-time

    so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.inter_op_num_threads = 1
    session = ort.InferenceSession(args.onnx, sess_options=so, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in session.get_inputs()]

    model, data = B.load_mujoco_model(robot["mjcf_path"], list(ctrl["stiffness"]),
                                      list(ctrl["damping"]), float(timing["physics_dt"]))
    renderer = mujoco.Renderer(model, height=Ht, width=W)
    cam = mujoco.MjvCamera()
    cam.azimuth = 120.0; cam.elevation = -15.0; cam.distance = 3.2

    import torch
    names = torch.load(args.motion, map_location="cpu", weights_only=False).get("motion_names", [])

    written = []
    all_stats = {}
    for spec in args.clips.split(","):
        idx_s, label = spec.split(":"); idx = int(idx_s)
        meta["clip_idx"] = idx
        short = sanitize(str(names[idx]).split("/")[-1].replace(".motion", "")) if idx < len(names) else f"clip{idx}"
        player = MotionPlayer(args.motion, motion_index=idx, control_dt=meta["control_dt"])
        try:
            frames, stats = render_clip(session, in_names, model, data, renderer, cam, player, meta, args)
        except Exception as e:
            print(f"[render] FAIL clip {idx} ({label}): {e}", flush=True)
            import traceback; traceback.print_exc(); continue
        ema_tag = f"ema={args.action_ema}" if args.action_ema is not None else "ema=off"
        print(f"[jerk] clip={idx}:{label} {ema_tag} status={stats['status']} "
              f"steps={stats['steps']}/{stats['planned_steps']} "
              f"mean|d_action|={stats['mean_abs_delta_deg']:.4f}deg "
              f"norm_jerk={stats['norm_jerk']:.4f} "
              f"mean_track_err={stats['mean_track_err_deg']:.4f}deg", flush=True)
        for w in stats.get("pause_windows", []):
            print(f"[pause] clip={idx}:{label} {ema_tag} win={w['t0']}-{w['t1']}s "
                  f"covered={w['covered']} xy_drift={w['xy_drift_m']*100:.2f}cm "
                  f"xy_max_exc={w['xy_max_excursion_m']*100:.2f}cm "
                  f"z_start={w['z_start_m']:.3f}m z_drop={w['z_drop_m']*100:.2f}cm "
                  f"mean|d_action|={w.get('mean_abs_delta_deg', float('nan')):.4f}deg", flush=True)
        all_stats[f"{idx}:{label}"] = stats
        if not frames:
            print(f"[render] no frames clip {idx} ({label})", flush=True); continue
        if args.no_video:
            continue
        fn = os.path.join(args.outdir, f"{label}__{args.cond}__{short}.mp4")
        imageio.mimwrite(fn, frames, fps=fps, codec="libx264", macro_block_size=16,
                         output_params=["-pix_fmt", "yuv420p"])
        written.append(fn)
        print(f"[render] wrote {fn} ({len(frames)} frames @ {fps}fps)", flush=True)
    renderer.close()
    if args.stats_json:
        with open(args.stats_json, "w") as fh:
            json.dump(all_stats, fh, indent=1)
        print(f"[render] stats -> {args.stats_json}", flush=True)
    print("[render] DONE files:", json.dumps(written))


if __name__ == "__main__":
    main()
