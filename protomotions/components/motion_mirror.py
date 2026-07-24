# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Online reference-motion mirror augmentation (sagittal plane).

Motivation
----------
Human demonstration corpora are right-hand dominant, which biases the WBC teacher:
MuJoCo eval on the canonical teacher measured a left/right wrist-tracking gap
(left 0.139 vs right 0.107). Offline corpus mirroring (append a mirrored twin of
every clip) doubles the motion-pack VRAM footprint (27 -> 54 GiB/GPU at STACK=3,
unsharded), which is prohibitive. This module instead reflects the *sampled*
reference state on the fly, per env-episode (Bernoulli(PM_MIRROR_PROB)), so the
policy sees both handednesses without storing a second copy of the corpus.

Geometry
--------
We reflect across the robot sagittal plane, i.e. the world x-z plane (mirror
matrix ``M = diag(1, -1, 1)`` -- a *negation of the y axis*). Because the mimic
observations/rewards are computed in a heading-local (root-relative) frame, a
rigid reflection of the whole world-frame reference pose about any vertical plane
yields a valid mirrored motion; we use the world y=0 plane and let the existing
spawn-offset machinery re-center it.

Transforms (all in COMMON RobotState ordering; ``rigid_body_rot`` is XYZW):

* positions (``rigid_body_pos``)          : plain vector  -> negate y  (idx 1)
* linear velocities (``rigid_body_vel``)  : plain vector  -> negate y  (idx 1)
* angular velocities (``rigid_body_ang_vel``): PSEUDOVECTOR under a reflection
      (det M = -1): ``w' = det(M) * M w`` -> negate x, z (idx 0, 2); keep y.
* rotations (``rigid_body_rot``, XYZW)    : reflect the rotation ``R -> M R M``.
      For M = diag(1,-1,1) this is the quaternion op ``(w,x,y,z)->(w,-x,y,-z)``
      i.e. negate the x and z slots, keep the y and w slots. Convention-robust:
      it always negates the x-slot and z-slot regardless of w-first vs w-last,
      because y is the mirror-plane normal. (Derivation + numerical compose test
      ``R(q') == M R(q) M`` live in the unit tests.)
* left<->right permutation: after the per-component sign flips, swap the
      ``left_*`` and ``right_*`` bodies / DOFs. Central bodies (pelvis, torso,
      head) map to themselves. Contacts permute with the *same* body permutation
      (this is what swaps the L/R foot-contact columns).
* hinge DOFs: a hinge angle about a body-local axis ``a`` mirrors to the twin
      joint's angle ``sign * theta`` where ``sign = +1`` for a pitch axis (y) and
      ``sign = -1`` for a roll (x) or yaw (z) axis -- derived from
      ``M rot(a, theta) M = rot(diag(-1,1,-1) a, theta)``. Signs are taken from
      ``kinematic_info.hinge_axes_map`` (not name heuristics), then the twin
      permutation is applied.

Root / anchor: the root body (index 0) is central and handled by the uniform
per-body ops above -- its yaw (heading) negates via the quaternion reflection and
its lateral position/velocity negate, exactly as required.

Default OFF: with ``PM_MIRROR_PROB=0`` the per-env mask is all-False and every
mirror entry point early-returns, so training is byte-identical to no-mirror.
"""

from typing import Dict, List, Optional

import torch


def _twin_name(name: str) -> str:
    """Return the left<->right twin of a body/joint name (self if central)."""
    if name.startswith("left_"):
        return "right_" + name[len("left_"):]
    if name.startswith("right_"):
        return "left_" + name[len("right_"):]
    # Some assets embed side as an infix (e.g. "L_", "_left_"); handle the common
    # infix form defensively, otherwise treat as central (self-mirrored).
    for a, b in (("_left_", "_right_"), ("_right_", "_left_")):
        if a in name:
            return name.replace(a, b)
    return name


class MirrorMaps:
    """Precomputed sagittal-mirror index/sign maps for one robot.

    All tensors are built ONCE and moved to the compute device so the per-step
    hot path is pure gather + elementwise-multiply (no host work, no syncs).

    Attributes:
        body_perm: LongTensor [num_bodies]; twin body index per body.
        dof_perm:  LongTensor [num_dofs];  twin DOF index per DOF.
        dof_sign:  FloatTensor [num_dofs]; per-DOF hinge-angle sign under mirror
            (symmetric under the twin swap, so applying it pre- or post-permute
            is equivalent).
        body_id / dof_id: identity aranges (the "unmirrored" gather rows).
        w_last:    bool; quaternion layout of ``rigid_body_rot`` (True == XYZW).
    """

    def __init__(
        self,
        body_perm: torch.Tensor,
        dof_perm: torch.Tensor,
        dof_sign: torch.Tensor,
        w_last: bool = True,
    ):
        self.body_perm = body_perm
        self.dof_perm = dof_perm
        self.dof_sign = dof_sign
        self.body_id = torch.arange(body_perm.numel(), dtype=torch.long)
        self.dof_id = torch.arange(dof_perm.numel(), dtype=torch.long)
        self._dof_ones = torch.ones_like(dof_sign)  # unmirrored-row DOF sign
        self.w_last = w_last
        # Quaternion slots to negate: x and z. y (plane normal) and w are kept.
        # xyzw layout -> (x=0, z=2); wxyz layout -> (x=1, z=3).
        self._quat_neg = (0, 2) if w_last else (1, 3)

    def to(self, device) -> "MirrorMaps":
        self.body_perm = self.body_perm.to(device)
        self.dof_perm = self.dof_perm.to(device)
        self.dof_sign = self.dof_sign.to(device)
        self.body_id = self.body_id.to(device)
        self.dof_id = self.dof_id.to(device)
        self._dof_ones = self._dof_ones.to(device)
        return self


def build_mirror_maps(
    body_names: List[str],
    dof_names: List[str],
    hinge_axes_map: Optional[Dict[int, torch.Tensor]] = None,
    w_last: bool = True,
    device=None,
) -> MirrorMaps:
    """Build sagittal-mirror maps from robot kinematic naming + hinge axes.

    Args:
        body_names: COMMON-ordering body names (kinematic_info.body_names).
        dof_names: COMMON-ordering DOF names (kinematic_info.dof_names).
        hinge_axes_map: body_idx -> hinge axes [n_dof_of_body, 3]; used to derive
            the per-DOF mirror sign rigorously (falls back to a roll/yaw name
            heuristic when None).
        w_last: quaternion layout of rigid_body_rot (True == XYZW == COMMON).
        device: optional device for the returned tensors.

    Verifies that every body/DOF maps to a valid twin (present in the list),
    raising if the robot is not left/right symmetric.
    """
    name_to_bidx = {n: i for i, n in enumerate(body_names)}
    body_perm = []
    for i, n in enumerate(body_names):
        tw = _twin_name(n)
        if tw not in name_to_bidx:
            raise ValueError(
                f"mirror: body '{n}' twin '{tw}' not found in body_names; "
                f"robot is not left/right symmetric."
            )
        body_perm.append(name_to_bidx[tw])

    # Map each DOF's body to derive its hinge axis. Convention: '<x>_joint' body
    # is '<x>_link' (matches H1-2 / G1 MJCF naming).
    name_to_didx = {n: j for j, n in enumerate(dof_names)}
    dof_perm, dof_sign = [], []
    for j, n in enumerate(dof_names):
        tw = _twin_name(n)
        if tw not in name_to_didx:
            raise ValueError(
                f"mirror: dof '{n}' twin '{tw}' not found in dof_names; "
                f"robot is not left/right symmetric."
            )
        dof_perm.append(name_to_didx[tw])

        sign = None
        if hinge_axes_map is not None:
            link = n[:-len("_joint")] + "_link" if n.endswith("_joint") else None
            bidx = name_to_bidx.get(link)
            if bidx is not None and bidx in hinge_axes_map:
                axes = hinge_axes_map[bidx]
                axis = axes[0].abs()  # single-DOF hinge bodies
                # pitch (y dominant) keeps sign; roll (x) / yaw (z) flip.
                sign = 1.0 if int(torch.argmax(axis)) == 1 else -1.0
        if sign is None:
            # Name heuristic fallback: roll/yaw flip, pitch/knee/elbow keep;
            # a bare 'torso_joint' (yaw) also flips.
            lname = n.lower()
            flip = ("roll" in lname) or ("yaw" in lname) or ("torso" in lname)
            sign = -1.0 if flip else 1.0
        dof_sign.append(sign)

    maps = MirrorMaps(
        body_perm=torch.tensor(body_perm, dtype=torch.long),
        dof_perm=torch.tensor(dof_perm, dtype=torch.long),
        dof_sign=torch.tensor(dof_sign, dtype=torch.float32),
        w_last=w_last,
    )
    if device is not None:
        maps = maps.to(device)
    return maps


def mirror_robot_state(state, flags: torch.Tensor, maps: MirrorMaps):
    """Sagittal-mirror the rows of a COMMON-ordering RobotState, per-env.

    HOT PATH -- called every step for the future-frame ref window. Fully
    vectorized and sync-free:

    * ``flags`` is a boolean tensor [B] aligned 1:1 with the batch dim of
      ``state`` (== the ``motion_ids`` rows given to ``get_motion_state``). It
      selects which rows are mirrored; unmirrored rows pass through unchanged.
    * per-component sign flips are done IN PLACE with a per-row +/-1 multiply
      (``fsign`` == -1 on mirrored rows, +1 elsewhere), so unmirrored rows are
      untouched and no boolean split / recombine is needed;
    * the left<->right permutation is a SINGLE batched ``gather`` per field,
      using a per-env index map that is the twin permutation on mirrored rows
      and the identity arange on unmirrored rows.

    No ``.any()`` / ``.item()`` calls -> no host-device sync. Callers must only
    invoke this when ``mirror_prob > 0`` (python-level bypass keeps
    ``PM_MIRROR_PROB=0`` byte-identical with zero added work).
    """
    if flags is None:
        return state
    dev = None
    for _f in ("rigid_body_pos", "dof_pos"):
        _v = getattr(state, _f, None)
        if _v is not None:
            dev = _v.device
            break
    if dev is None:
        return state
    flags = flags.to(dev)
    fb = flags.view(-1, 1)  # [B,1] bool for column broadcasts
    # Per-row +/-1 sign: -1 on mirrored rows, +1 elsewhere. [B,1] float. The
    # in-place sign multiply touches only the affected component columns and
    # leaves unmirrored rows untouched (sign +1), so no boolean split is needed.
    fsign = torch.where(fb, torch.as_tensor(-1.0, device=dev),
                        torch.as_tensor(1.0, device=dev))
    nq0, nq1 = maps._quat_neg

    # Per-env gather maps: twin perm on mirrored rows, identity arange elsewhere.
    # This lets ONE gather (per field) cover both mirrored and unmirrored envs --
    # no boolean split/recombine, no host sync. Cheaper than a whole-batch mirror
    # + torch.where (which would add an extra full read/write pass per field).
    bmap = torch.where(fb, maps.body_perm.view(1, -1), maps.body_id.view(1, -1))  # [B,Bn]
    dmap = torch.where(fb, maps.dof_perm.view(1, -1), maps.dof_id.view(1, -1))    # [B,D]

    def _gather_bodies(v):  # v: [B, Bn, C]
        idx = bmap.unsqueeze(-1).expand(-1, -1, v.shape[-1])
        return v.gather(1, idx)

    # ---- positions / linear velocities: plain vectors -> negate y (idx 1) ----
    for field in ("rigid_body_pos", "rigid_body_vel", "rigid_body_contact_forces"):
        v = getattr(state, field, None)
        if v is None:
            continue
        v[..., 1] *= fsign  # in-place y negate on mirrored rows only
        setattr(state, field, _gather_bodies(v))

    # ---- angular velocities: pseudovector -> negate x,z; keep y --------------
    v = getattr(state, "rigid_body_ang_vel", None)
    if v is not None:
        v[..., 0] *= fsign
        v[..., 2] *= fsign
        setattr(state, "rigid_body_ang_vel", _gather_bodies(v))

    # ---- rotations (world + cached local): reflect quat (negate x,z slots) ---
    for field in ("rigid_body_rot", "local_rigid_body_rot"):
        v = getattr(state, field, None)
        if v is None:
            continue
        v[..., nq0] *= fsign
        v[..., nq1] *= fsign
        setattr(state, field, _gather_bodies(v))

    # ---- contacts: body permutation only (swaps L/R foot columns) -----------
    v = getattr(state, "rigid_body_contacts", None)
    if v is not None:
        setattr(state, "rigid_body_contacts", v.gather(1, bmap))

    # ---- hinge DOFs: per-row sign multiply then twin permute ----------------
    # dof_sign is symmetric under the twin swap, so source-side sign then gather
    # equals dest-side sign; per-row dsign is +1 on unmirrored rows.
    dsign = torch.where(fb, maps.dof_sign.view(1, -1),
                        maps._dof_ones.view(1, -1))  # [B,D]
    for field in ("dof_pos", "dof_vel"):
        v = getattr(state, field, None)
        if v is None:
            continue
        v = v * dsign
        setattr(state, field, v.gather(1, dmap))

    return state
