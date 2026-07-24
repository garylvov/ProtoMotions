# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline mirror of individual motion clips, for the render + ONNX gates.

This reuses the SAME validated transform as the online path
(``components/motion_mirror.mirror_robot_state``) but writes the result to disk,
so the existing kinematic-replay renderer and the MuJoCo/ONNX eval harness can
consume mirrored clips without any online plumbing. It is NOT used in training
(training mirrors online); it exists only to produce side-by-side render clips
and the ~10 mirrored probe clips for the pre-enable gates.

Usage (CPU):
    PYTHONPATH=. python -m protomotions.tools.mirror_clips \
        --robot h1_2 --in clip.pt --out clip_mirrored.pt

A clip .pt is a dict of per-frame tensors in COMMON ordering (rigid_body_pos
[T,B,3], rigid_body_rot [T,B,4] XYZW, rigid_body_vel, rigid_body_ang_vel,
dof_pos [T,D], dof_vel, optional rigid_body_contacts, fps). Every frame is
mirrored (flags all-True).
"""

import argparse
import torch

from protomotions.components.motion_mirror import build_mirror_maps, mirror_robot_state
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    StateConversion,
)


def _robot_kinematic_info(robot: str):
    if robot == "h1_2":
        from protomotions.robot_configs.h1_2 import H1_2RobotConfig
        return H1_2RobotConfig().kinematic_info
    raise ValueError(f"unsupported robot for mirror_clips: {robot}")


def mirror_clip_dict(clip: dict, ki) -> dict:
    """Mirror one clip dict in place-ish; returns a mirrored dict."""
    maps = build_mirror_maps(ki.body_names, ki.dof_names, ki.hinge_axes_map,
                             w_last=True)
    fps = clip.get("fps", None)
    st = RobotState.from_dict(clip, state_conversion=StateConversion.COMMON)
    T = st.rigid_body_pos.shape[0]
    mirror_robot_state(st, torch.ones(T, dtype=torch.bool), maps)
    out = st.to_dict()
    if fps is not None and "fps" not in out:
        out["fps"] = fps
    # preserve any non-RobotState metadata keys unchanged
    for k, v in clip.items():
        if k not in out and not isinstance(v, torch.Tensor):
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="h1_2")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()
    ki = _robot_kinematic_info(args.robot)
    clip = torch.load(args.inp, weights_only=False, map_location="cpu")
    out = mirror_clip_dict(clip, ki)
    torch.save(out, args.out)
    print(f"[mirror_clips] wrote mirrored clip -> {args.out} "
          f"({out['rigid_body_pos'].shape[0]} frames)")


if __name__ == "__main__":
    main()
