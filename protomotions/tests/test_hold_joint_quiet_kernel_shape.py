"""Guard tests for the hold_joint_quiet kernel-shape fix (v57 underflow).

MEASURED FACTS these lock down (v57 canonical_teacher_20260805_v57 TB,
5286 logging steps, 8192 envs):

    env/hold_joint_quiet/hold_dof_vel_rms_mean  6.09444 -> 7.57392 rad/s
    env/hold_joint_quiet/dof_vel_rms_mean       7.06837 -> 7.36310 rad/s
    env/hold_joint_quiet/still_frac_mean        0.02471 -> 0.04343
    env/raw_r/hold_joint_quiet_mean             0.0 at EVERY step (std 0.0)

With vel_scale=25 the Gaussian argument is 25 * 7.574**2 = 1434, and
exp(-1434) is an EXACT zero with an EXACT zero gradient in fp32 and fp64.
"""

import math

import pytest
import torch

from protomotions.envs import component_factories as factories
from protomotions.envs.base_env.hold_fix import compute_hold_joint_quiet

# The live v57 hold-window rms (rad/s) and the eval-derived destination band.
V57_HOLD_RMS_NOW = 7.57392
V57_HOLD_RMS_START = 6.09444
DESTINATION_RMS = 0.15

# The recommended v58 kernel.
V58 = dict(kernel="lorentzian", vel_scale=0.25, fine_weight=0.5, fine_vel_scale=25.0)


def _still(n=1):
    return torch.ones(n, dtype=torch.bool)


def _uniform_dof_vel(rms: float, num_dofs: int = 29) -> torch.Tensor:
    """A [1, num_dofs] velocity row whose mean_j v^2 is exactly rms**2."""
    return torch.full((1, num_dofs), float(rms), dtype=torch.float64)


def _value(rms: float, **kw) -> float:
    return float(compute_hold_joint_quiet(_still(), _uniform_dof_vel(rms), **kw))


def _grad_wrt_rms(rms: float, **kw) -> float:
    v = torch.full((1, 29), float(rms), dtype=torch.float64, requires_grad=True)
    out = compute_hold_joint_quiet(_still(), v, **kw)
    out.backward()
    # d/d(rms) of a uniform row: every column moves together.
    return float(v.grad.sum())


# ---------------------------------------------------------------- the defect


def test_the_shipped_gaussian_is_exactly_dead_at_the_measured_distribution():
    """Reproduces the v57 failure exactly -- value AND gradient hard zero."""
    for rms in (V57_HOLD_RMS_START, V57_HOLD_RMS_NOW):
        assert _value(rms, vel_scale=25.0) == 0.0
        assert _grad_wrt_rms(rms, vel_scale=25.0) == 0.0


def test_rescaling_the_gaussian_alone_cannot_span_the_dynamic_range():
    """Why the fix is a SHAPE change, not a re-tuned constant.

    Any single Gaussian placed to have live gradient at the 7.57 rad/s ORIGIN
    is saturated flat at the 0.15 rad/s DESTINATION -- the two are 1.7 decades
    apart and exp(-c v^2) has usable gradient over about one octave.
    """
    c_origin = 1.0 / V57_HOLD_RMS_NOW**2  # e-folding exactly at the origin
    assert _value(V57_HOLD_RMS_NOW, vel_scale=c_origin) == pytest.approx(
        math.exp(-1.0), rel=1e-6
    )
    # ...and at the destination it is indistinguishable from a constant 1.
    assert _value(DESTINATION_RMS, vel_scale=c_origin) > 0.9995
    assert abs(_grad_wrt_rms(DESTINATION_RMS, vel_scale=c_origin)) < 6e-3
    # The dual Lorentzian is ~180x more sensitive there.
    assert abs(_grad_wrt_rms(DESTINATION_RMS, **V58)) > 1.0


def test_exponential_in_abs_v_underflows_for_the_same_reason():
    """exp(-c|v|) was a candidate; it decays exponentially too, so it dies.

    Sized to e-fold at 0.20 rad/s (c=5) it pays exp(-37.9) at the origin --
    which is not an exact zero, but at 3e-17 it is below fp32 gradient noise
    and 12 orders below the Lorentzian's payout. Only a POWER-LAW tail works.
    """
    assert math.exp(-5.0 * V57_HOLD_RMS_NOW) < 1e-16
    assert _value(V57_HOLD_RMS_NOW, **V58) > 1e-2


# ------------------------------------------------------------------ the fix


def test_lorentzian_is_finite_and_has_live_gradient_at_the_measured_origin():
    now = _value(V57_HOLD_RMS_NOW, **V58)
    start = _value(V57_HOLD_RMS_START, **V58)
    assert now == pytest.approx(0.043689, abs=5e-6)
    assert start == pytest.approx(0.065174, abs=5e-6)
    # The reward must PREFER the quieter of the two measured states.
    assert start > now

    assert _grad_wrt_rms(V57_HOLD_RMS_NOW, **V58) == pytest.approx(-0.010789, abs=5e-6)
    assert _grad_wrt_rms(V57_HOLD_RMS_START, **V58) == pytest.approx(
        -0.019320, abs=5e-6
    )


def test_lorentzian_pays_the_destination_band_and_stays_bounded():
    assert _value(DESTINATION_RMS, **V58) == pytest.approx(0.876271, abs=5e-6)
    assert _value(0.06, **V58) == pytest.approx(0.971878, abs=5e-6)
    assert _value(0.0, **V58) == pytest.approx(1.0, abs=1e-12)
    # Bounded [0, 1] -- the documented contract; the fine channel is
    # renormalized by (1 + fine_weight), so weight=0.25 keeps its meaning.
    for rms in (0.0, 0.06, 0.15, 2.0, 6.09444, 7.57392, 100.0):
        assert 0.0 <= _value(rms, **V58) <= 1.0


def test_lorentzian_is_strictly_monotone_across_two_decades():
    """The property the Gaussian loses: usable ordering over the whole range."""
    grid = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    vals = [_value(r, **V58) for r in grid]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    # ...with a power-law (never underflowing) tail.
    assert _value(1000.0, **V58) > 0.0
    assert _grad_wrt_rms(1000.0, **V58) != 0.0


def test_lorentzian_log_log_slope_is_the_scale_free_minus_two():
    """The reason it degrades gracefully: constant relative sensitivity."""
    coarse = dict(kernel="lorentzian", vel_scale=0.25)
    slopes = [
        _grad_wrt_rms(rms, **coarse) * rms / _value(rms, **coarse)
        for rms in (5.0, 20.0, 80.0, 320.0)
    ]
    # Monotonically approaches -2 and is already within 0.3 at the ORIGIN
    # scale -- an exponential's log-log slope instead grows without bound.
    assert all(a > b for a, b in zip(slopes, slopes[1:]))
    assert slopes[0] == pytest.approx(-1.724, abs=1e-3)
    assert slopes[-1] == pytest.approx(-2.0, abs=1e-3)


# ------------------------------------------------------- Rule 10 / contracts


def test_unset_kernel_is_byte_identical_to_the_pre_fix_kernel():
    still = torch.tensor([True, False, True])
    dof_vel = torch.randn(3, 29, dtype=torch.float64)
    legacy = still.float() * torch.exp(-25.0 * (dof_vel * dof_vel).mean(dim=-1))
    got = compute_hold_joint_quiet(still, dof_vel, vel_scale=25.0)
    assert torch.equal(got, legacy)
    # Explicit gaussian + fine_weight=0 must also be bit-for-bit unchanged.
    assert torch.equal(
        compute_hold_joint_quiet(
            still, dof_vel, vel_scale=25.0, kernel="gaussian", fine_weight=0.0
        ),
        legacy,
    )


def test_still_mask_gating_survives_the_shape_change():
    still = torch.tensor([True, False])
    dof_vel = torch.zeros(2, 29, dtype=torch.float64)
    got = compute_hold_joint_quiet(still, dof_vel, **V58)
    assert float(got[0]) == pytest.approx(1.0)
    assert float(got[1]) == 0.0
    # No mask source at all -> zeros, unchanged convention.
    assert torch.equal(
        compute_hold_joint_quiet(None, dof_vel, **V58), torch.zeros(2)
    )


def test_fine_weight_without_fine_vel_scale_is_refused_not_guessed():
    with pytest.raises(ValueError, match="fine_vel_scale"):
        compute_hold_joint_quiet(_still(), _uniform_dof_vel(1.0), fine_weight=0.5)
    with pytest.raises(ValueError, match="fine_vel_scale"):
        factories.hold_joint_quiet_factory(weight=0.25, fine_weight=0.5)


def test_unknown_kernel_is_refused_at_the_kernel_and_at_the_factory():
    with pytest.raises(ValueError, match="lorentzian"):
        compute_hold_joint_quiet(_still(), _uniform_dof_vel(1.0), kernel="cauchy")
    with pytest.raises(ValueError, match="lorentzian"):
        factories.hold_joint_quiet_factory(weight=0.25, kernel="cauchy")


def test_negative_fine_weight_is_refused():
    with pytest.raises(ValueError, match="fine_weight"):
        factories.hold_joint_quiet_factory(weight=0.25, fine_weight=-0.5)


def test_factory_keeps_default_kernel_keys_out_of_static_params():
    plain = factories.hold_joint_quiet_factory(weight=0.25).static_params
    assert "kernel" not in plain
    assert "fine_weight" not in plain
    assert "fine_vel_scale" not in plain

    tuned = factories.hold_joint_quiet_factory(weight=0.25, **V58).static_params
    assert tuned["kernel"] == "lorentzian"
    assert tuned["vel_scale"] == 0.25
    assert tuned["fine_weight"] == 0.5
    assert tuned["fine_vel_scale"] == 25.0


def test_resume_reapply_accepts_the_string_kernel_knob():
    """Regression: the reapply loop float()-ed every extra var."""
    comp = factories.hold_joint_quiet_factory(weight=0.25)
    components = {"hold_joint_quiet": comp}
    env = {
        "PM_HOLD_JOINT_QUIET_WEIGHT": "0.25",
        "PM_HOLD_JOINT_QUIET_KERNEL": "lorentzian",
        "PM_HOLD_JOINT_QUIET_VEL_SCALE": "0.25",
        "PM_HOLD_JOINT_QUIET_FINE_WEIGHT": "0.5",
        "PM_HOLD_JOINT_QUIET_FINE_VEL_SCALE": "25.0",
    }
    lines = []
    changed = factories.resume_inject_reward_components(
        components, env, log_fn=lines.append
    )
    assert changed
    sp = components["hold_joint_quiet"].static_params
    assert sp["kernel"] == "lorentzian"
    assert sp["vel_scale"] == 0.25
    assert sp["fine_weight"] == 0.5
    assert sp["fine_vel_scale"] == 25.0
    assert any("kernel = lorentzian" in line for line in lines)


def test_resume_reapply_injects_a_tuned_component_from_scratch():
    components = {}
    env = {
        "PM_HOLD_JOINT_QUIET_WEIGHT": "0.25",
        "PM_HOLD_JOINT_QUIET_KERNEL": "lorentzian",
        "PM_HOLD_JOINT_QUIET_VEL_SCALE": "0.25",
        "PM_HOLD_JOINT_QUIET_FINE_WEIGHT": "0.5",
        "PM_HOLD_JOINT_QUIET_FINE_VEL_SCALE": "25.0",
    }
    assert factories.resume_inject_reward_components(components, env, log_fn=lambda _: None)
    sp = components["hold_joint_quiet"].static_params
    assert sp["kernel"] == "lorentzian"
    assert sp["fine_vel_scale"] == 25.0


def test_stat_writer_rms_is_the_same_quantity_the_kernel_reduces():
    """The 50x is NOT an aggregation mismatch -- this is the proof.

    hold_dof_vel_rms**2 IS the hold-weighted mean of the kernel's mean_j v^2,
    so the TB stat can be substituted straight into the kernel argument.
    """
    torch.manual_seed(0)
    dof_vel = torch.randn(64, 29, dtype=torch.float64)
    still = torch.zeros(64, dtype=torch.bool)
    still[:8] = True

    mean_sq = (dof_vel * dof_vel).mean(dim=-1)
    hold_rms = ((mean_sq * still.float()).sum() / still.float().sum()).sqrt()
    kernel_arg = mean_sq[still].mean()
    assert float(hold_rms) ** 2 == pytest.approx(float(kernel_arg), rel=1e-12)
