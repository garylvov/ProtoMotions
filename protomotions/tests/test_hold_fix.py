# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit smokes for the HOLD-FIX hold-conditioning interventions.

Covers (CPU-only, synthetic reference data):
- RefStillnessTracker: moving refs stay unmasked, still refs mask after the
  window, frozen-clock refs (zero delta, arbitrary stored velocities) mask,
  motion-resample discontinuities reset the streak, explicit freeze mask ORs,
  per-axis epsilon sensitivity (root lin / root ang / dof).
- compute_hold_balance_bonus: pays ~1 for quiet upright double-support stance
  inside a still window; pays 0 outside stills, on lifted feet, and ~0 when
  tilted or drifting; None mask -> zeros.
- setup_hold_fix_components: injects hold_fix_fall_penalty (weight = -gate,
  threshold mirrored from the fall termination) and hold_balance; injects
  nothing when gates are off.
- load_hold_fix_settings env-var parsing (off by default).
"""

import math
import os

import torch

from protomotions.envs.base_env.hold_fix import (
    RefStillnessTracker,
    compute_hold_balance_bonus,
    load_hold_fix_settings,
    resolve_foot_body_ids,
    setup_hold_fix_components,
)

DT = 0.02  # 50 Hz
NUM_ENVS = 3
NUM_DOFS = 5
DEVICE = torch.device("cpu")

IDENTITY_QUAT = torch.tensor([0.0, 0.0, 0.0, 1.0])


def _tracker(window=5, **kw):
    return RefStillnessTracker(
        num_envs=NUM_ENVS, device=DEVICE, window=window, **kw
    )


def _ref(root_xyz=(0.0, 0.0, 0.9), yaw=0.0, dof_fill=0.0):
    root_pos = torch.tensor(root_xyz).repeat(NUM_ENVS, 1)
    root_rot = torch.tensor(
        [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]
    ).repeat(NUM_ENVS, 1)
    dof_pos = torch.full((NUM_ENVS, NUM_DOFS), dof_fill)
    return root_pos, root_rot, dof_pos


def test_tracker_still_reference_masks_after_window():
    tr = _tracker(window=5)
    p, r, d = _ref()
    masks = [tr.update(p, r, d, DT) for _ in range(8)]
    # First update primes the buffers; streak reaches 5 at update 6.
    assert not masks[0].any()
    assert not masks[4].any()  # streak 4 < window
    assert masks[5].all()  # streak 5 >= window
    assert masks[7].all()


def test_tracker_moving_reference_never_masks():
    tr = _tracker(window=2)
    for i in range(10):
        # Root advances 2 cm/step = 1.0 m/s >> 0.05 m/s epsilon.
        p, r, d = _ref(root_xyz=(0.02 * i, 0.0, 0.9))
        mask = tr.update(p, r, d, DT)
    assert not mask.any()


def test_tracker_dof_motion_blocks_mask():
    # Jog-in-place-style reference: root still, dofs swinging.
    tr = _tracker(window=2)
    for i in range(10):
        p, r, d = _ref(dof_fill=0.1 * (i % 2))  # 0.1 rad/step = 5 rad/s
        mask = tr.update(p, r, d, DT)
    assert not mask.any()


def test_tracker_root_spin_blocks_mask():
    tr = _tracker(window=2)
    for i in range(10):
        p, r, d = _ref(yaw=0.02 * i)  # 1 rad/s >> 0.30 rad/s epsilon
        mask = tr.update(p, r, d, DT)
    assert not mask.any()


def test_tracker_frozen_clock_masks_despite_stored_velocities():
    # A frozen reference clock replays the SAME frame: deltas are zero even
    # if the frame was mid-jog. The tracker sees only deltas, so it masks.
    tr = _tracker(window=3)
    p, r, d = _ref(root_xyz=(1.23, 4.56, 0.9), yaw=0.7, dof_fill=0.4)
    for _ in range(4):
        mask = tr.update(p, r, d, DT)
    assert mask.all()


def test_tracker_resample_discontinuity_resets_streak():
    tr = _tracker(window=3)
    p, r, d = _ref()
    for _ in range(5):
        mask = tr.update(p, r, d, DT)
    assert mask.all()
    # Motion resample: reference jumps discontinuously -> streak resets.
    p2, r2, d2 = _ref(root_xyz=(5.0, 5.0, 0.9), dof_fill=1.0)
    mask = tr.update(p2, r2, d2, DT)
    assert not mask.any()
    # Still again from the new pose: needs a fresh window.
    mask = tr.update(p2, r2, d2, DT)
    assert not mask.any()
    mask = tr.update(p2, r2, d2, DT)
    mask = tr.update(p2, r2, d2, DT)
    assert mask.all()


def test_tracker_explicit_freeze_mask_ors_per_env():
    tr = _tracker(window=2)
    freeze = torch.tensor([True, False, False])
    for i in range(5):
        # Reference is MOVING for all envs; env 0 flagged frozen explicitly.
        p, r, d = _ref(root_xyz=(0.02 * i, 0.0, 0.9))
        mask = tr.update(p, r, d, DT, freeze_mask=freeze)
    assert bool(mask[0])
    assert not mask[1] and not mask[2]


def test_tracker_subthreshold_drift_masks():
    # 0.0005 m/step = 0.025 m/s < 0.05 m/s epsilon: counts as still.
    tr = _tracker(window=3)
    for i in range(6):
        p, r, d = _ref(root_xyz=(0.0005 * i, 0.0, 0.9))
        mask = tr.update(p, r, d, DT)
    assert mask.all()


# =============================================================================
# Balance bonus kernel
# =============================================================================

LEFT_FOOT = torch.tensor([2])
RIGHT_FOOT = torch.tensor([3])
NUM_BODIES = 4


def _balance_inputs(
    still=True, planted_left=True, planted_right=True, tilt_deg=0.0, speed=0.0
):
    still_mask = torch.full((NUM_ENVS,), bool(still), dtype=torch.bool)
    tilt = math.radians(tilt_deg)
    # Roll about x by tilt: q = (sin(t/2), 0, 0, cos(t/2)) xyzw.
    root_rot = torch.tensor(
        [math.sin(tilt / 2), 0.0, 0.0, math.cos(tilt / 2)]
    ).repeat(NUM_ENVS, 1)
    root_vel = torch.zeros(NUM_ENVS, 3)
    root_vel[:, 0] = speed
    contacts = torch.zeros(NUM_ENVS, NUM_BODIES, dtype=torch.bool)
    contacts[:, LEFT_FOOT] = planted_left
    contacts[:, RIGHT_FOOT] = planted_right
    return still_mask, root_rot, root_vel, contacts


def _bonus(**kw):
    still_mask, root_rot, root_vel, contacts = _balance_inputs(**kw)
    return compute_hold_balance_bonus(
        still_mask, root_rot, root_vel, contacts, LEFT_FOOT, RIGHT_FOOT
    )


def test_balance_bonus_quiet_upright_stance_pays_full():
    b = _bonus()
    assert torch.allclose(b, torch.ones(NUM_ENVS), atol=1e-5)


def test_balance_bonus_zero_outside_still_window():
    assert (_bonus(still=False) == 0).all()


def test_balance_bonus_zero_when_foot_lifted():
    # Single support (stepping-in-place) earns nothing.
    assert (_bonus(planted_left=False) == 0).all()
    assert (_bonus(planted_right=False) == 0).all()


def test_balance_bonus_decays_with_tilt_and_speed():
    upright = _bonus()[0]
    tilted = _bonus(tilt_deg=30.0)[0]
    fallen = _bonus(tilt_deg=80.0)[0]
    assert tilted < upright and fallen < tilted
    assert fallen < 0.05
    slow = _bonus(speed=0.1)[0]
    fast = _bonus(speed=1.0)[0]
    assert 0.9 < slow < 1.0  # small corrective motion barely taxed
    assert fast < 0.05  # drifting/jogging pays ~nothing


def test_balance_bonus_none_mask_returns_zeros():
    _, root_rot, root_vel, contacts = _balance_inputs()
    b = compute_hold_balance_bonus(
        None, root_rot, root_vel, contacts, LEFT_FOOT, RIGHT_FOOT
    )
    assert (b == 0).all()


# =============================================================================
# Settings parsing + component injection
# =============================================================================

BODY_NAMES = ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link"]
COMMON_NAMING = {
    "all_left_foot_bodies": ["left_ankle_roll_link"],
    "all_right_foot_bodies": ["right_ankle_roll_link"],
}


def _with_env(env: dict):
    """Context manager: temporarily set env vars."""
    class _Ctx:
        def __enter__(self):
            self.saved = {k: os.environ.get(k) for k in env}
            os.environ.update({k: str(v) for k, v in env.items()})

        def __exit__(self, *a):
            for k, v in self.saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    return _Ctx()


def test_settings_default_all_off():
    with _with_env({}):
        for k in ("HOLD_ACTION_GRACE", "FALL_TERM_PENALTY", "HOLD_BALANCE_BONUS"):
            os.environ.pop(k, None)
        s = load_hold_fix_settings()
    assert not s.any_enabled and not s.needs_still_mask


def test_settings_parse_gates():
    with _with_env(
        {"HOLD_ACTION_GRACE": "1", "FALL_TERM_PENALTY": "5.0",
         "HOLD_BALANCE_BONUS": "0.2", "HOLD_STILL_WINDOW": "7"}
    ):
        s = load_hold_fix_settings()
    assert s.hold_action_grace
    assert s.fall_term_penalty == 5.0
    assert s.hold_balance_bonus == 0.2
    assert s.still_window == 7
    assert s.any_enabled and s.needs_still_mask


def test_setup_injects_nothing_when_off():
    from protomotions.envs.base_env.hold_fix import HoldFixSettings

    rewards, terms = {}, {}
    setup_hold_fix_components(
        HoldFixSettings(), rewards, terms, BODY_NAMES, COMMON_NAMING, DEVICE
    )
    assert rewards == {}


def test_setup_injects_fall_penalty_mirroring_term_threshold():
    from protomotions.envs.base_env.hold_fix import HoldFixSettings
    from protomotions.envs.component_factories import anchor_height_error_term_factory

    rewards = {}
    terms = {"fall": anchor_height_error_term_factory(threshold=0.31)}
    setup_hold_fix_components(
        HoldFixSettings(fall_term_penalty=5.0),
        rewards, terms, BODY_NAMES, COMMON_NAMING, DEVICE,
    )
    comp = rewards["hold_fix_fall_penalty"]
    params = comp.get_params()
    assert params["weight"] == -5.0
    assert params["height_threshold"] == 0.31


def test_setup_injects_balance_bonus_with_resolved_feet():
    from protomotions.envs.base_env.hold_fix import HoldFixSettings

    rewards = {}
    setup_hold_fix_components(
        HoldFixSettings(hold_balance_bonus=0.2),
        rewards, {}, BODY_NAMES, COMMON_NAMING, DEVICE,
    )
    comp = rewards["hold_balance"]
    params = comp.get_params()
    assert params["weight"] == 0.2
    assert params["left_foot_body_ids"].tolist() == [2]
    assert params["right_foot_body_ids"].tolist() == [3]


def test_resolve_foot_ids_fallback_substring():
    ids = resolve_foot_body_ids(
        ["pelvis", "L_Ankle", "R_Ankle"], None, DEVICE
    )
    assert ids["left"].tolist() == [1]
    assert ids["right"].tolist() == [2]


# =============================================================================
# Env-side per-step flow (_apply_hold_fix on a mock env)
# =============================================================================


def test_env_apply_hold_fix_updates_mask_and_ors_grace():
    from types import SimpleNamespace

    from protomotions.envs.base_env.env import BaseEnv
    from protomotions.envs.base_env.hold_fix import HoldFixSettings

    num_bodies = 4
    ref_state = SimpleNamespace(
        rigid_body_pos=torch.zeros(NUM_ENVS, num_bodies, 3),
        rigid_body_rot=IDENTITY_QUAT.repeat(NUM_ENVS, num_bodies, 1),
        dof_pos=torch.zeros(NUM_ENVS, NUM_DOFS),
    )
    # env 2 is under an explicit reference freeze
    freeze = torch.tensor([0.0, 0.0, 1.5])

    env = SimpleNamespace(
        _hold_fix=HoldFixSettings(hold_action_grace=True),
        _hold_still_tracker=_tracker(window=2),
        _hold_still_mask=torch.zeros(NUM_ENVS, dtype=torch.bool),
        motion_manager=SimpleNamespace(_freeze_time_left=freeze),
        dt=DT,
        extras={},
    )

    def run_step(sim_grace):
        ctx = SimpleNamespace(
            mimic=SimpleNamespace(ref_state=ref_state),
            perturbation_grace_mask=sim_grace,
        )
        BaseEnv._apply_hold_fix(env, ctx)
        return ctx

    # step 1 primes the tracker (all-still ref, but window=2 not yet reached)
    ctx = run_step(sim_grace=None)
    assert not env._hold_still_mask.any()
    # steps 2-3: still streak reaches the window for all envs
    ctx = run_step(sim_grace=None)
    ctx = run_step(sim_grace=None)
    assert env._hold_still_mask.all()
    # grace mask was None -> becomes the still mask
    assert ctx.perturbation_grace_mask.all()
    assert (env.extras["hold_fix/ref_still_mask"] == 1.0).all()
    assert (env.extras["hold_fix/grace_mask"] == 1.0).all()

    # now make the reference MOVE: mask drops, but sim grace is preserved by OR
    ref_state.rigid_body_pos = ref_state.rigid_body_pos + 1.0
    sim_grace = torch.tensor([True, False, False])
    ctx = run_step(sim_grace=sim_grace)
    # env 2 stays still via the explicit freeze... but its delta also jumped
    # (ref tensor moved for all envs), so only the sim grace survives.
    assert not env._hold_still_mask[0] and not env._hold_still_mask[1]
    assert bool(ctx.perturbation_grace_mask[0])  # sim grace OR-ed through
    assert not bool(ctx.perturbation_grace_mask[1])


def test_env_apply_hold_fix_noop_when_disabled():
    from types import SimpleNamespace

    from protomotions.envs.base_env.env import BaseEnv

    env = SimpleNamespace(_hold_still_tracker=None, extras={})
    ctx = SimpleNamespace(perturbation_grace_mask=None)
    BaseEnv._apply_hold_fix(env, ctx)  # must not raise or touch anything
    assert ctx.perturbation_grace_mask is None
    assert env.extras == {}


# =============================================================================
# WRIST-DIR direction-agreement reward
# =============================================================================


def _wd_tracker(K=5, disp_eps=0.02, vmax=0.0, instantaneous=False, speed_eps=0.1):
    from protomotions.envs.base_env.hold_fix import WristDirTracker

    return WristDirTracker(
        num_envs=NUM_ENVS, num_wrists=2, device=DEVICE,
        window_steps=K, window_s=K * DT, disp_eps=disp_eps,
        vel_weight_vmax=vmax, instantaneous=instantaneous, speed_eps=speed_eps,
    )


def _wd_step(tr, t, pol_dir=(1.0, 0.0, 0.0), ref_dir=(1.0, 0.0, 0.0),
             speed=0.5, ref_speed=0.5, progress=None):
    """One tracker step with wrists moving along straight lines."""
    pol = torch.tensor(pol_dir) * speed * DT * t
    ref = torch.tensor(ref_dir) * ref_speed * DT * t
    wrist_pos = pol.repeat(NUM_ENVS, 2, 1)
    ref_pos = ref.repeat(NUM_ENVS, 2, 1)
    wrist_vel = (torch.tensor(pol_dir) * speed).repeat(NUM_ENVS, 2, 1)
    ref_vel = (torch.tensor(ref_dir) * ref_speed).repeat(NUM_ENVS, 2, 1)
    if progress is None:
        progress = torch.full((NUM_ENVS,), t, dtype=torch.long)
    return tr.update(wrist_pos, ref_pos, wrist_vel, ref_vel, progress)


def test_wrist_dir_parallel_motion_pays_full_after_warmup():
    tr = _wd_tracker(K=5)
    for t in range(1, 5):
        r = _wd_step(tr, t)
        assert (r == 0).all(), f"cold window must pay 0 (t={t})"
    for t in range(5, 9):
        r = _wd_step(tr, t)
    assert torch.allclose(r, torch.ones(NUM_ENVS), atol=1e-5)
    assert torch.allclose(tr.dir_cos, torch.ones(NUM_ENVS), atol=1e-5)
    assert (tr.active_frac == 1.0).all()


def test_wrist_dir_antiparallel_zero_reward_negative_cos():
    tr = _wd_tracker(K=3)
    for t in range(1, 8):
        r = _wd_step(tr, t, pol_dir=(-1.0, 0.0, 0.0), ref_dir=(1.0, 0.0, 0.0))
    assert (r == 0).all()  # relu floors antiparallel at 0
    assert (tr.dir_cos < -0.99).all()  # raw cos extra still watches the axis


def test_wrist_dir_orthogonal_zero_reward():
    tr = _wd_tracker(K=3)
    for t in range(1, 8):
        r = _wd_step(tr, t, pol_dir=(0.0, 1.0, 0.0), ref_dir=(1.0, 0.0, 0.0))
    assert (r.abs() < 1e-5).all()


def test_wrist_dir_still_reference_inert():
    # Hold: reference wrists do not travel -> inactive -> zero reward, even
    # though the POLICY wrists move. Composes with the still-mask by
    # construction (gating is on reference displacement only).
    tr = _wd_tracker(K=3, disp_eps=0.02)
    for t in range(1, 10):
        r = _wd_step(tr, t, ref_speed=0.0)
    assert (r == 0).all()
    assert (tr.active_frac == 0).all()
    # sub-threshold creep (< disp_eps over window) also inert:
    tr = _wd_tracker(K=3, disp_eps=0.02)
    for t in range(1, 10):
        r = _wd_step(tr, t, ref_speed=0.02)  # 0.02*3*0.02 = 1.2mm < 20mm
    assert (r == 0).all()


def test_wrist_dir_reset_clears_window_no_cross_episode_ghost():
    tr = _wd_tracker(K=3)
    for t in range(1, 8):
        r = _wd_step(tr, t)
    assert (r > 0.99).all()
    # Episode reset: progress drops. Even with a HUGE apparent displacement
    # (teleport to spawn), the env must go cold for K steps.
    r = _wd_step(tr, 100, progress=torch.ones(NUM_ENVS, dtype=torch.long))
    assert (r == 0).all(), "reset step must not pay from stale buffer"
    for t in range(2, 4):
        r = _wd_step(tr, 100 + t, progress=torch.full((NUM_ENVS,), t, dtype=torch.long))
        assert (r == 0).all(), f"cold rewarm step {t} must pay 0"
    for t in range(4, 8):
        r = _wd_step(tr, 100 + t, progress=torch.full((NUM_ENVS,), t, dtype=torch.long))
    assert (r > 0.99).all()  # warm again after K fresh frames


def test_wrist_dir_velocity_proportional_weighting():
    # vmax=1.0: ref at 0.5 m/s -> weight 0.5; ref at 2.0 m/s -> capped 1.0.
    tr_slow = _wd_tracker(K=3, vmax=1.0)
    tr_fast = _wd_tracker(K=3, vmax=1.0)
    for t in range(1, 8):
        r_slow = _wd_step(tr_slow, t, speed=0.5, ref_speed=0.5)
        r_fast = _wd_step(tr_fast, t, speed=2.0, ref_speed=2.0)
    assert torch.allclose(r_slow, torch.full((NUM_ENVS,), 0.5), atol=1e-4)
    assert torch.allclose(r_fast, torch.ones(NUM_ENVS), atol=1e-4)


def test_wrist_dir_instantaneous_subflag():
    tr = _wd_tracker(instantaneous=True, speed_eps=0.1)
    r = _wd_step(tr, 1)  # no warmup needed for the velocity variant
    assert torch.allclose(r, torch.ones(NUM_ENVS), atol=1e-5)
    r = _wd_step(tr, 2, ref_speed=0.05)  # below speed_eps -> inert
    assert (r == 0).all()


def test_wrist_dir_batch_shapes():
    tr = _wd_tracker(K=4)
    for t in range(1, 7):
        r = _wd_step(tr, t)
    assert r.shape == (NUM_ENVS,)
    assert tr.dir_cos.shape == (NUM_ENVS,)
    assert tr.active_frac.shape == (NUM_ENVS,)
    assert tr._buf_pol.shape == (NUM_ENVS, 4, 2, 3)


def test_wrist_dir_setup_injects_component():
    from protomotions.envs.base_env.hold_fix import (
        HoldFixSettings, setup_hold_fix_components,
    )

    rewards = {}
    naming = {
        "all_left_hand_bodies": ["left_ankle_roll_link"],  # stand-in names
        "all_right_hand_bodies": ["right_ankle_roll_link"],
    }
    setup_hold_fix_components(
        HoldFixSettings(wrist_dir_weight=0.03),
        rewards, {}, BODY_NAMES, naming, DEVICE,
    )
    comp = rewards["wrist_dir"]
    params = comp.get_params()
    assert params["weight"] == 0.03
    paths = " ".join(str(v) for v in comp.dynamic_vars.values())
    assert "wrist_dir_reward" in paths


def test_wrist_dir_settings_parse():
    with _with_env({"WRIST_DIR_REWARD": "0.03", "WRIST_DIR_WINDOW_S": "0.4",
                    "WRIST_DIR_DISP_EPS": "0.03", "WRIST_DIR_INSTANT": "1",
                    "WRIST_DIR_VMAX": "1.5"}):
        s = load_hold_fix_settings()
    assert s.wrist_dir_weight == 0.03 and s.any_enabled
    assert s.wrist_dir_window_s == 0.4
    assert s.wrist_dir_disp_eps == 0.03
    assert s.wrist_dir_instant
    assert s.wrist_dir_vmax == 1.5
    # off by default
    with _with_env({}):
        os.environ.pop("WRIST_DIR_REWARD", None)
        s = load_hold_fix_settings()
    assert s.wrist_dir_weight == 0.0


def test_wrist_dir_subsample_ring():
    # K=6, M=2 -> 3 slots; lookback quantizes to ~slots*M steps. Parallel
    # motion must pay full after warmup; reset must still go cold.
    from protomotions.envs.base_env.hold_fix import WristDirTracker

    tr = WristDirTracker(
        num_envs=NUM_ENVS, num_wrists=2, device=DEVICE,
        window_steps=6, window_s=6 * DT, disp_eps=0.02, subsample=2,
    )
    assert tr.slots == 3 and tr.M == 2
    assert tr._buf_pol.shape == (NUM_ENVS, 3, 2, 3)  # bytes per env
    r = None
    for t in range(1, 12):
        r = _wd_step(tr, t)
    assert torch.allclose(r, torch.ones(NUM_ENVS), atol=1e-5)
    # reset -> cold despite stale slots
    r = _wd_step(tr, 100, progress=torch.ones(NUM_ENVS, dtype=torch.long))
    assert (r == 0).all()
    # still-ref inertness holds under subsampling
    tr2 = WristDirTracker(
        num_envs=NUM_ENVS, num_wrists=2, device=DEVICE,
        window_steps=6, window_s=6 * DT, disp_eps=0.02, subsample=3,
    )
    for t in range(1, 15):
        r = _wd_step(tr2, t, ref_speed=0.0)
    assert (r == 0).all()


# =============================================================================
# ROOT-GAIN displacement-gain reward
# =============================================================================


def _rg_tracker(K=5, disp_eps=0.03, subsample=1):
    from protomotions.envs.base_env.hold_fix import WristDirTracker

    return WristDirTracker(
        num_envs=NUM_ENVS, num_wrists=1, device=DEVICE,
        window_steps=K, window_s=K * DT, disp_eps=disp_eps,
        subsample=subsample, shaping="gain_proj",
    )


def _rg_step(tr, t, pol_speed=1.0, ref_speed=1.0, pol_dir=(1.0, 0.0, 0.0),
             ref_dir=(1.0, 0.0, 0.0), progress=None):
    pol = (torch.tensor(pol_dir) * pol_speed * DT * t).repeat(NUM_ENVS, 1, 1)
    ref = (torch.tensor(ref_dir) * ref_speed * DT * t).repeat(NUM_ENVS, 1, 1)
    if progress is None:
        progress = torch.full((NUM_ENVS,), t, dtype=torch.long)
    return tr.update(pol, ref, torch.zeros(NUM_ENVS, 1, 3),
                     torch.zeros(NUM_ENVS, 1, 3), progress)


def test_root_gain_matched_progress_pays_full():
    tr = _rg_tracker(K=3)
    for t in range(1, 8):
        r = _rg_step(tr, t, pol_speed=1.0, ref_speed=1.0)
    assert torch.allclose(r, torch.ones(NUM_ENVS), atol=1e-5)
    assert torch.allclose(tr.gain, torch.ones(NUM_ENVS), atol=1e-5)


def test_root_gain_undershoot_pays_proportionally():
    # The 0.464 pathology: policy covers 46.4% of ref displacement -> 0.464.
    tr = _rg_tracker(K=3)
    for t in range(1, 8):
        r = _rg_step(tr, t, pol_speed=0.464, ref_speed=1.0)
    assert torch.allclose(r, torch.full((NUM_ENVS,), 0.464), atol=1e-3)


def test_root_gain_overshoot_clamped_no_bonus():
    tr = _rg_tracker(K=3)
    for t in range(1, 8):
        r = _rg_step(tr, t, pol_speed=2.0, ref_speed=1.0)
    assert torch.allclose(r, torch.ones(NUM_ENVS), atol=1e-5)  # clamped at 1
    assert torch.allclose(tr.gain, torch.full((NUM_ENVS,), 2.0), atol=1e-3)
    # raw gain extra still shows the overshoot (watchable, unrewarded)


def test_root_gain_backward_and_lateral():
    tr = _rg_tracker(K=3)
    for t in range(1, 8):
        r = _rg_step(tr, t, pol_dir=(-1.0, 0.0, 0.0))  # backward vs ref fwd
    assert (r == 0).all()
    tr = _rg_tracker(K=3)
    for t in range(1, 8):
        r = _rg_step(tr, t, pol_dir=(0.0, 1.0, 0.0))  # pure lateral
    assert (r.abs() < 1e-5).all()  # projection: lateral motion earns nothing


def test_root_gain_still_reference_inert():
    tr = _rg_tracker(K=3, disp_eps=0.03)
    for t in range(1, 10):
        r = _rg_step(tr, t, ref_speed=0.0)  # hold: ref root does not travel
    assert (r == 0).all()
    assert (tr.active_frac == 0).all()
    # sub-eps creep also inert: 0.05 m/s * 3 steps * 0.02 s = 3 mm < 30 mm
    tr = _rg_tracker(K=3, disp_eps=0.03)
    for t in range(1, 10):
        r = _rg_step(tr, t, ref_speed=0.05)
    assert (r == 0).all()


def test_root_gain_reset_boundary_goes_cold():
    tr = _rg_tracker(K=3)
    for t in range(1, 8):
        r = _rg_step(tr, t)
    assert (r > 0.99).all()
    # reset: progress drops; huge apparent displacement must not pay
    r = _rg_step(tr, 200, progress=torch.ones(NUM_ENVS, dtype=torch.long))
    assert (r == 0).all()
    for t in range(2, 4):
        r = _rg_step(tr, 200 + t,
                     progress=torch.full((NUM_ENVS,), t, dtype=torch.long))
        assert (r == 0).all()
    for t in range(4, 8):
        r = _rg_step(tr, 200 + t,
                     progress=torch.full((NUM_ENVS,), t, dtype=torch.long))
    assert (r > 0.99).all()


def test_root_gain_setup_injects_component():
    from protomotions.envs.base_env.hold_fix import (
        HoldFixSettings, setup_hold_fix_components,
    )

    rewards = {}
    setup_hold_fix_components(
        HoldFixSettings(root_gain_weight=0.03),
        rewards, {}, BODY_NAMES, COMMON_NAMING, DEVICE,
    )
    comp = rewards["root_gain"]
    assert comp.get_params()["weight"] == 0.03
    paths = " ".join(str(v) for v in comp.dynamic_vars.values())
    assert "root_gain_reward" in paths


def test_root_gain_settings_parse():
    with _with_env({"ROOT_GAIN_REWARD": "0.05", "ROOT_GAIN_WINDOW_S": "0.4",
                    "ROOT_GAIN_DISP_EPS": "0.05", "ROOT_GAIN_SUBSAMPLE": "2"}):
        s = load_hold_fix_settings()
    assert s.root_gain_weight == 0.05 and s.any_enabled
    assert s.root_gain_window_s == 0.4
    assert s.root_gain_disp_eps == 0.05
    assert s.root_gain_subsample == 2
    with _with_env({}):
        os.environ.pop("ROOT_GAIN_REWARD", None)
        s = load_hold_fix_settings()
    assert s.root_gain_weight == 0.0
