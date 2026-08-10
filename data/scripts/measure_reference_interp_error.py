"""Measure reference-interpolation error at the wrists.

Two independent estimators, no assumed ground truth:

(1) LEADING-ORDER (exact, no extrapolation). For lerp between samples spaced h,
    err(u) = x(t0+uh) - [(1-u)x0 + u x1] = -1/2 u(1-u) h^2 x''(t0) + O(h^3)
    and h^2 x'' is EXACTLY the second difference D2[k] = p[k+1]-2p[k]+p[k-1] of the
    stored samples. So the error is computable straight from stored data:
        err_rms = rms_u(1/2 u(1-u)) * |D2|
    with u drawn from the REAL 50Hz-vs-30fps blend set {0,.6,.2,.8,.4}.

(2) COARSEN-AND-CHECK (ground-truthed). Decimate the stored 30fps samples by
    k=2,3,4, reconstruct at every stored 30fps time with lerp / Catmull-Rom,
    compare against the true stored frames. Fit E(h)=A h^p, extrapolate to h=1/30.
    Also yields the cubic-vs-linear improvement ratio at real spacing.

Wrist bodies: 21 = left_wrist_yaw_link, 28 = right_wrist_yaw_link (root=0).
"""
import argparse
import json
import math

import numpy as np
import torch

WRISTS = [21, 28]
ROOT = 0
CTRL_DT = 0.02  # 50 Hz control loop


def blend_set(motion_dt, ctrl_dt=CTRL_DT, n=2000):
    """Actual blend factors the control loop produces against this frame grid."""
    t = np.arange(n) * ctrl_dt
    idx0 = np.floor(t / motion_dt)
    return np.clip((t - idx0 * motion_dt) / motion_dt, 0.0, 1.0)


def lerp_coeff_rms(u):
    """rms over u of the leading-order lerp error coefficient 1/2 u(1-u)."""
    return float(np.sqrt(np.mean((0.5 * u * (1.0 - u)) ** 2)))


def catmull_rom(p, u):
    """p: (..., 4, 3) samples at knots -1,0,1,2 ; u in [0,1] between knots 0 and 1."""
    p0, p1, p2, p3 = p[..., 0, :], p[..., 1, :], p[..., 2, :], p[..., 3, :]
    u = u[..., None]
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * u
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * u3
    )


def coarsen_experiment(pos, k, mode):
    """pos: (T,3) stored 30fps wrist track. Decimate by k, rebuild at stored times.

    Returns per-query-point error norms (only for query points that are NOT
    coarse knots, i.e. genuinely interpolated).
    """
    T = pos.shape[0]
    knots = np.arange(0, T, k)
    if knots.shape[0] < 4:
        return None
    cp = pos[knots]  # (K,3) coarse samples
    K = cp.shape[0]
    # query at every stored time that lies strictly inside the coarse span
    q = np.arange(0, knots[-1] + 1)
    seg = np.minimum(q // k, K - 2)
    u = (q - seg * k) / float(k)
    interior = u > 1e-9
    seg, u, q = seg[interior], u[interior], q[interior]
    if seg.size == 0:
        return None
    if mode == "lerp":
        rec = (1.0 - u)[:, None] * cp[seg] + u[:, None] * cp[seg + 1]
    else:  # catmull-rom, clamped at ends
        i0 = np.clip(seg - 1, 0, K - 1)
        i1 = seg
        i2 = np.clip(seg + 1, 0, K - 1)
        i3 = np.clip(seg + 2, 0, K - 1)
        stack = np.stack([cp[i0], cp[i1], cp[i2], cp[i3]], axis=-2)
        rec = catmull_rom(stack, u)
    return np.linalg.norm(rec - pos[q], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--n-clips", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root-relative", action="store_true",
                    help="measure wrist position relative to root (anchor-relative stat)")
    args = ap.parse_args()

    d = torch.load(args.pack, weights_only=True, mmap=True, map_location="cpu")
    motion_dt = d["motion_dt"].float().numpy()
    nframes = d["motion_num_frames"].numpy()
    starts = d["length_starts"].numpy()
    weights = d["motion_weights"].float().numpy()
    gts = d["gts"]
    grs = d["grs"]

    # ---- quaternion convention sanity (task item 3) --------------------------
    probe = grs[starts[0] : starts[0] + 200].float().numpy()
    qnorm = np.linalg.norm(probe, axis=-1)
    comp_mean_abs = np.abs(probe).reshape(-1, 4).mean(axis=0)
    quat_report = {
        "norm_mean": float(qnorm.mean()),
        "norm_min": float(qnorm.min()),
        "norm_max": float(qnorm.max()),
        "mean_abs_per_component": [float(x) for x in comp_mean_abs],
        "likely_w_index": int(np.argmax(comp_mean_abs)),
    }

    # ---- pick clips: only the 30fps population (dt != ctrl_dt) ---------------
    thirty = np.where(np.abs(motion_dt - CTRL_DT) > 1e-6)[0]
    thirty = thirty[nframes[thirty] >= 12]
    rng = np.random.default_rng(args.seed)
    w = weights[thirty].astype(np.float64)
    w = w / w.sum()
    n = min(args.n_clips, thirty.size)
    pick = rng.choice(thirty, size=n, replace=False, p=w)

    # blend coefficient for the REAL 50Hz control grid on a 30fps frame grid
    u_real = blend_set(float(motion_dt[pick[0]]))
    coeff = lerp_coeff_rms(u_real)
    coeff_max = 0.5 * 0.25  # u=0.5 worst case

    d2_all = []          # |second difference| per wrist sample, metres
    speed_all = []       # per-sample wrist speed, m/s
    per_k = {k: {"lerp": [], "cr": []} for k in (2, 3, 4)}
    travel_all = []

    for mid in pick:
        s, nf = int(starts[mid]), int(nframes[mid])
        dt = float(motion_dt[mid])
        blk = gts[s : s + nf].float().numpy()  # (nf, B, 3)
        for wb in WRISTS:
            p = blk[:, wb, :].astype(np.float64)
            if args.root_relative:
                p = p - blk[:, ROOT, :].astype(np.float64)
            if p.shape[0] < 12:
                continue
            d2 = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=-1)
            d2_all.append(d2)
            v = np.linalg.norm(np.diff(p, axis=0), axis=-1) / dt
            speed_all.append(v[:-1])
            travel_all.append(np.linalg.norm(np.diff(p, axis=0), axis=-1))
            for k in (2, 3, 4):
                for mode, key in (("lerp", "lerp"), ("cr", "cr")):
                    e = coarsen_experiment(p, k, "lerp" if mode == "lerp" else "cr")
                    if e is not None and e.size:
                        per_k[k][key].append(e)

    d2 = np.concatenate(d2_all)
    speed = np.concatenate(speed_all)
    travel = np.concatenate(travel_all)

    # ---- estimator (1): leading order, at the true 30fps spacing --------------
    err_lo = coeff * d2  # metres, rms-over-blend error per sample location
    est1 = {
        "n_samples": int(d2.size),
        "blend_rms_coeff": coeff,
        "rms_mm": float(np.sqrt(np.mean(err_lo ** 2)) * 1000),
        "mean_mm": float(err_lo.mean() * 1000),
        "p50_mm": float(np.percentile(err_lo, 50) * 1000),
        "p95_mm": float(np.percentile(err_lo, 95) * 1000),
        "p99_mm": float(np.percentile(err_lo, 99) * 1000),
        "max_mm": float(err_lo.max() * 1000),
        "worstblend_rms_mm": float(np.sqrt(np.mean((coeff_max * d2) ** 2)) * 1000),
    }

    # speed dependence
    qs = [0, 50, 80, 95, 99, 100]
    bins = []
    edges = np.percentile(speed, qs)
    for i in range(len(qs) - 1):
        m = (speed >= edges[i]) & (speed <= edges[i + 1])
        if m.sum() > 10:
            bins.append({
                "speed_pctile": f"{qs[i]}-{qs[i+1]}",
                "speed_range_mps": [float(edges[i]), float(edges[i + 1])],
                "n": int(m.sum()),
                "interp_err_rms_mm": float(np.sqrt(np.mean(err_lo[m] ** 2)) * 1000),
            })

    # ---- estimator (2): coarsen-and-check + power-law extrapolation -----------
    est2 = {"per_k": {}}
    hs, els, ecs = [], [], []
    base_dt = 1.0 / 30.0
    for k in (2, 3, 4):
        el = np.concatenate(per_k[k]["lerp"])
        ec = np.concatenate(per_k[k]["cr"])
        rl = float(np.sqrt(np.mean(el ** 2)) * 1000)
        rc = float(np.sqrt(np.mean(ec ** 2)) * 1000)
        est2["per_k"][k] = {
            "h_s": k * base_dt,
            "lerp_rms_mm": rl, "lerp_p95_mm": float(np.percentile(el, 95) * 1000),
            "cr_rms_mm": rc, "cr_p95_mm": float(np.percentile(ec, 95) * 1000),
            "cr_over_lerp": rc / rl if rl else float("nan"),
        }
        hs.append(k * base_dt); els.append(rl); ecs.append(rc)
    hs = np.array(hs)
    pl = np.polyfit(np.log(hs), np.log(els), 1)
    pc = np.polyfit(np.log(hs), np.log(ecs), 1)
    est2["fit"] = {
        "lerp_order_p": float(pl[0]),
        "cubic_order_p": float(pc[0]),
        "lerp_rms_mm_at_30fps": float(math.exp(np.polyval(pl, math.log(base_dt)))),
        "cubic_rms_mm_at_30fps": float(math.exp(np.polyval(pc, math.log(base_dt)))),
    }

    out = {
        "pack": args.pack,
        "root_relative": args.root_relative,
        "n_clips": int(n),
        "ctrl_dt": CTRL_DT,
        "quaternion_check": quat_report,
        "per_frame_wrist_travel_mm": {
            "mean": float(travel.mean() * 1000),
            "p95": float(np.percentile(travel, 95) * 1000),
        },
        "wrist_speed_mps": {
            "mean": float(speed.mean()),
            "p95": float(np.percentile(speed, 95)),
            "max": float(speed.max()),
        },
        "estimator1_leading_order": est1,
        "estimator1_by_speed": bins,
        "estimator2_coarsen": est2,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
