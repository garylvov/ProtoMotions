"""Direct, extrapolation-free measurement of what a cubic upgrade would CHANGE.

At the REAL control instants (50 Hz) against the REAL stored frame grid (30 fps),
evaluate the current lerp and a Catmull-Rom cubic through the same stored samples.
|CR - lerp| upper-bounds the mm that switching to cubic could move the reference.

Also reports overshoot risk: Catmull-Rom rings on the piecewise-constant segments
that the synthetic `pauses` / `frozen_bottom` classes deliberately contain.
"""
import argparse
import json

import numpy as np
import torch

WRISTS = [21, 28]
ROOT = 0
CTRL_DT = 0.02


def cr(p0, p1, p2, p3, u):
    u = u[:, None]
    u2, u3 = u * u, u * u * u
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--n-clips", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root-relative", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = torch.load(args.pack, weights_only=True, mmap=True, map_location="cpu")
    motion_dt = d["motion_dt"].float().numpy()
    nframes = d["motion_num_frames"].numpy()
    starts = d["length_starts"].numpy()
    weights = d["motion_weights"].float().numpy()
    gts = d["gts"]

    thirty = np.where(np.abs(motion_dt - CTRL_DT) > 1e-6)[0]
    thirty = thirty[nframes[thirty] >= 12]
    rng = np.random.default_rng(args.seed)
    w = weights[thirty].astype(np.float64); w /= w.sum()
    pick = rng.choice(thirty, size=min(args.n_clips, thirty.size), replace=False, p=w)

    diffs, blends, speeds, static_flag = [], [], [], []
    for mid in pick:
        s, nf = int(starts[mid]), int(nframes[mid])
        dt = float(motion_dt[mid])
        L = (nf - 1) * dt
        blk = gts[s : s + nf].float().numpy()
        # real control instants, exactly as calc_frame_blend produces them
        t = np.arange(0, int(L / CTRL_DT) + 1) * CTRL_DT
        t = np.clip(t, 0.0, L)
        phase = np.clip(t / L, 0.0, 1.0)
        i0 = np.minimum((phase * (nf - 1)).astype(np.int64), nf - 1)
        i1 = np.minimum(i0 + 1, nf - 1)
        b = np.clip((t - i0 * dt) / dt, 0.0, 1.0)
        for wb in WRISTS:
            p = blk[:, wb, :].astype(np.float64)
            if args.root_relative:
                p = p - blk[:, ROOT, :].astype(np.float64)
            lin = (1 - b)[:, None] * p[i0] + b[:, None] * p[i1]
            im1 = np.clip(i0 - 1, 0, nf - 1)
            ip2 = np.clip(i1 + 1, 0, nf - 1)
            cub = cr(p[im1], p[i0], p[i1], p[ip2], b)
            diffs.append(np.linalg.norm(cub - lin, axis=-1))
            blends.append(b)
            sp = np.linalg.norm(p[i1] - p[i0], axis=-1) / dt
            speeds.append(sp)
            # "static" = local frame travel under 1 mm (pause-like segment)
            static_flag.append(np.linalg.norm(p[i1] - p[i0], axis=-1) < 1e-3)

    diff = np.concatenate(diffs)
    b = np.concatenate(blends)
    sp = np.concatenate(speeds)
    st = np.concatenate(static_flag)

    interior = b > 1e-9  # blend==0 rows are exact frame hits, zero by construction
    di = diff[interior]

    out = {
        "n_clips": int(pick.size),
        "root_relative": args.root_relative,
        "n_control_instants": int(diff.size),
        "frac_exact_frame_hits": float((~interior).mean()),
        "cubic_minus_lerp_mm": {
            "rms_all": float(np.sqrt(np.mean(diff ** 2)) * 1000),
            "rms_interior": float(np.sqrt(np.mean(di ** 2)) * 1000),
            "mean_interior": float(di.mean() * 1000),
            "p50": float(np.percentile(di, 50) * 1000),
            "p95": float(np.percentile(di, 95) * 1000),
            "p99": float(np.percentile(di, 99) * 1000),
            "max": float(di.max() * 1000),
        },
        "overshoot_on_static_segments_mm": {
            "n": int((st & interior).sum()),
            "frac_of_instants": float((st & interior).mean()),
            "rms": float(np.sqrt(np.mean(diff[st & interior] ** 2)) * 1000)
            if (st & interior).any() else None,
            "p99": float(np.percentile(diff[st & interior], 99) * 1000)
            if (st & interior).any() else None,
            "max": float(diff[st & interior].max() * 1000)
            if (st & interior).any() else None,
        },
    }
    bins = []
    qs = [0, 50, 80, 95, 99, 100]
    e = np.percentile(sp[interior], qs)
    spi = sp[interior]
    for i in range(len(qs) - 1):
        m = (spi >= e[i]) & (spi <= e[i + 1])
        if m.sum() > 10:
            bins.append({"speed_pctile": f"{qs[i]}-{qs[i+1]}",
                         "speed_range_mps": [float(e[i]), float(e[i + 1])],
                         "n": int(m.sum()),
                         "cubic_minus_lerp_rms_mm": float(np.sqrt(np.mean(di[m] ** 2)) * 1000)})
    out["by_speed"] = bins
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
