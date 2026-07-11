# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DELAY-DR: reset-semantics audit tests + partial-exposure sampling tests.

Gary's warning: "be really careful, especially for resets." These tests PROVE
the existing DelayDomainRandomizationConfig machinery's reset semantics on the
real BaseEnv._apply_action_delay / _sample_delays code (unbound, mock env):

- Delay-of-1 and delay-of-2 EXACTNESS: action commanded at t is applied at
  t+d, verified step-by-step against ground truth.
- Masked partial-reset correctness: env A resets, env B's delayed stream is
  byte-identical to the no-reset control (no global flush, no cross-talk).
- Pre-fill semantics: the first post-reset step applies the env's OWN current
  action (effective delay clamps to steps-since-reset -> ramps 0..d); NEVER a
  pre-reset action, NEVER zeros/garbage.
- Partial-exposure sampling (ACTION_DELAY_DR): frac split statistics, values
  in {1..steps_max}, per-reset resampling, clean envs exactly 0.

Audit conclusion encoded here (2026-07-10): the machinery is SOUND — the
per-env clamp eff_delay = min(d_i, progress_buf_i) obviates buffer flushing
entirely; progress_buf zeroing (env.py reset path) and _sample_delays are both
masked to the reset env_ids. The laneb v1 non-convergence was NOT a stale
reset leak (consistent with the worklog diagnosis: global full-range delay
from epoch 0 + combined formulation).
"""

from types import SimpleNamespace

import torch

from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.base_env.hold_fix import (
    ActionDelaySettings,
    load_action_delay_settings,
)

DEVICE = torch.device("cpu")
ACT_DIM = 1


def _mock_env(num_envs, delays, max_delay):
    """Mock env exposing exactly the state _apply_action_delay touches."""
    return SimpleNamespace(
        num_envs=num_envs,
        device=DEVICE,
        progress_buf=torch.zeros(num_envs, dtype=torch.long),
        _has_action_delay=True,
        _action_delay=torch.tensor(delays, dtype=torch.long),
        _action_delay_buf=torch.zeros(num_envs, max_delay + 1, ACT_DIM),
        _action_delay_step=0,
    )


def _action(t, num_envs):
    """Distinct per-env, per-step action: env i at step t -> 1000*i + t."""
    return torch.tensor(
        [[1000.0 * i + t] for i in range(num_envs)]
    )


def _step(env, t):
    """One control step: apply delay (progress pre-bump), then bump progress
    (post_physics), exactly the real step() ordering."""
    applied = BaseEnv._apply_action_delay(env, _action(t, env.num_envs))
    env.progress_buf += 1
    return applied


def test_delay_exactness_d1_and_d2():
    env = _mock_env(2, delays=[1, 2], max_delay=2)
    # env0 d=1, env1 d=2. Ground truth: applied(t) = a(max(0, t-d)) while
    # ramping (eff = min(d, t)), i.e. a(t-d) once t >= d.
    for t in range(8):
        applied = _step(env, t)
        exp0 = 1000.0 * 0 + max(0, t - 1)
        exp1 = 1000.0 * 1 + max(0, t - 2)
        assert applied[0, 0] == exp0, (t, applied[0, 0].item(), exp0)
        assert applied[1, 0] == exp1, (t, applied[1, 0].item(), exp1)


def test_prefill_first_post_reset_step_is_own_current_action():
    env = _mock_env(2, delays=[2, 2], max_delay=2)
    for t in range(5):
        _step(env, t)
    # Reset env0 ONLY (masked, like env.py:1597): progress -> 0.
    env.progress_buf[0] = 0
    applied = _step(env, 5)
    # env0 first post-reset step: eff=0 -> its OWN action a(5). Under a
    # stale-leak bug it would get a(3) (pre-reset, 2 back); under a zero
    # pre-fill it would get 0. Both must not happen.
    assert applied[0, 0] == 5.0, applied[0, 0].item()
    # Ramp: next step eff=1 -> a(5); then eff=2 -> a(5); then a(6)...
    applied = _step(env, 6)
    assert applied[0, 0] == 5.0
    applied = _step(env, 7)
    assert applied[0, 0] == 5.0
    applied = _step(env, 8)
    assert applied[0, 0] == 6.0


def test_masked_partial_reset_does_not_touch_other_envs():
    # env1's delayed stream must be IDENTICAL whether or not env0 resets.
    env_a = _mock_env(2, delays=[2, 2], max_delay=2)
    env_b = _mock_env(2, delays=[2, 2], max_delay=2)  # control: no reset
    stream_a, stream_b = [], []
    for t in range(4):
        stream_a.append(_step(env_a, t)[1, 0].item())
        stream_b.append(_step(env_b, t)[1, 0].item())
    env_a.progress_buf[0] = 0  # masked reset of env0 only
    for t in range(4, 10):
        stream_a.append(_step(env_a, t)[1, 0].item())
        stream_b.append(_step(env_b, t)[1, 0].item())
    assert stream_a == stream_b, (stream_a, stream_b)


def test_delay_exactness_across_reset_boundary_d1():
    env = _mock_env(1, delays=[1], max_delay=2)
    for t in range(4):
        _step(env, t)
    env.progress_buf[0] = 0  # reset
    assert _step(env, 4)[0, 0] == 4.0  # eff=0: own current action
    assert _step(env, 5)[0, 0] == 4.0  # eff=1: previous (post-reset) action
    assert _step(env, 6)[0, 0] == 5.0  # steady d=1


# =============================================================================
# Partial-exposure sampling (ACTION_DELAY_DR)
# =============================================================================


class _Cfg:
    action_delay_steps = (0, 2)

    @staticmethod
    def effective_max_action_delay(epoch):
        return 2


def _sample(n, frac, steps_max, seed=0):
    torch.manual_seed(seed)
    env = SimpleNamespace(
        _delay_cfg=_Cfg(),
        _has_action_delay=True,
        _has_obs_delay=False,
        _current_epoch=0,
        device=DEVICE,
        _action_delay=torch.zeros(n, dtype=torch.long),
        _obs_delay=torch.zeros(n, dtype=torch.long),
        _delay_partial=ActionDelaySettings(
            enabled=True, frac=frac, steps_max=steps_max
        ),
    )
    BaseEnv._sample_delays(env, torch.arange(n))
    return env


def test_partial_exposure_frac_split_and_values():
    n = 20000
    env = _sample(n, frac=0.25, steps_max=2)
    d = env._action_delay
    exposed_frac = (d > 0).float().mean().item()
    assert abs(exposed_frac - 0.25) < 0.02, exposed_frac
    assert set(d.unique().tolist()) <= {0, 1, 2}
    # exposed envs uniform over {1,2}
    ones = (d == 1).sum().item()
    twos = (d == 2).sum().item()
    assert abs(ones - twos) / max(ones + twos, 1) < 0.1


def test_partial_exposure_resampled_at_reset():
    env = _sample(4096, frac=0.5, steps_max=2, seed=1)
    before = env._action_delay.clone()
    BaseEnv._sample_delays(env, torch.arange(4096))
    after = env._action_delay
    assert not torch.equal(before, after)  # assignment churns across resets
    assert abs((after > 0).float().mean().item() - 0.5) < 0.03


def test_partial_exposure_masked_resample_only_touches_reset_envs():
    env = _sample(100, frac=1.0, steps_max=2, seed=2)
    before = env._action_delay.clone()
    reset_ids = torch.arange(0, 10)
    torch.manual_seed(99)
    BaseEnv._sample_delays(env, reset_ids)
    after = env._action_delay
    assert torch.equal(before[10:], after[10:])  # non-reset envs untouched


def test_partial_exposure_disabled_falls_back_to_stock():
    n = 20000
    torch.manual_seed(3)
    env = SimpleNamespace(
        _delay_cfg=_Cfg(),
        _has_action_delay=True,
        _has_obs_delay=False,
        _current_epoch=0,
        device=DEVICE,
        _action_delay=torch.zeros(n, dtype=torch.long),
        _obs_delay=torch.zeros(n, dtype=torch.long),
        _delay_partial=ActionDelaySettings(enabled=False),
    )
    BaseEnv._sample_delays(env, torch.arange(n))
    d = env._action_delay
    # stock: uniform {0,1,2} -> ~1/3 each
    for v in (0, 1, 2):
        assert abs((d == v).float().mean().item() - 1.0 / 3.0) < 0.02


def test_settings_env_parsing():
    import os

    for k in ("ACTION_DELAY_DR", "ACTION_DELAY_FRAC", "ACTION_DELAY_STEPS_MAX"):
        os.environ.pop(k, None)
    s = load_action_delay_settings()
    assert not s.enabled  # off by default
    os.environ.update({"ACTION_DELAY_DR": "1", "ACTION_DELAY_FRAC": "0.4",
                       "ACTION_DELAY_STEPS_MAX": "3"})
    try:
        s = load_action_delay_settings()
        assert s.enabled and s.frac == 0.4 and s.steps_max == 3
    finally:
        for k in ("ACTION_DELAY_DR", "ACTION_DELAY_FRAC",
                  "ACTION_DELAY_STEPS_MAX"):
            os.environ.pop(k, None)
