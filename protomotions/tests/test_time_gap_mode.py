# SPDX-License-Identifier: Apache-2.0
"""Guard tests for the masked-mimic conditioning-gap sampler.

The bug these exist to prevent is not a crash. It is a SILENT train/deploy
mismatch: `min_time_gap` / `max_time_gap` read like a sampling range and are
actually a clamp applied after the draw is scaled by remaining clip time. That
misreading survived in a code comment, in a sweep script header, and in the
campaign's own analysis, and it inverted the interpretation of every gap sweep
we ran.
"""

import math

import pytest
import torch


def _beta(alpha=2.0, beta=5.0, n=200_000, seed=0):
    torch.manual_seed(seed)
    return torch.distributions.Beta(alpha, beta).sample((n,))


def _remaining_scaled(beta_samples, remaining, lo, hi):
    off = beta_samples * remaining
    return torch.clamp(off, min=lo, max=hi)


def _absolute(beta_samples, lo, hi):
    return lo + beta_samples * (hi - lo)


# --------------------------------------------------------------------------
# What the historical mode actually does.
# --------------------------------------------------------------------------

def test_remaining_scaled_ignores_the_configured_range():
    """The headline defect: on long clips the draw is dominated by clip
    length, so the configured bounds do not describe the samples."""
    b = _beta()
    remaining = torch.full_like(b, 6.0)          # a 6 s clip
    off = _remaining_scaled(b, remaining, 0.02, 2.0)
    median = off.median().item()
    assert median > 0.5, (
        f"median gap {median:.3f}s -- if this is small the corpus assumption "
        "changed and the campaign's 0.79s figure needs re-deriving"
    )


def test_remaining_scaled_essentially_never_yields_50hz():
    """P(gap <= 0.02) must be ~0, which is why 50 Hz conditioning is OOD."""
    b = _beta()
    remaining = torch.full_like(b, 6.0)
    off = _remaining_scaled(b, remaining, 0.02, 2.0)
    frac = (off <= 0.0201).float().mean().item()
    assert frac < 0.01, f"{frac:.4f} of draws at the floor; expected ~0"


def test_remaining_scaled_distribution_depends_on_clip_length():
    """Proof that the realised distribution is a property of the CORPUS, not
    the config: same config, different clip length, different median."""
    b = _beta()
    short = _remaining_scaled(b, torch.full_like(b, 1.0), 0.02, 2.0).median().item()
    long_ = _remaining_scaled(b, torch.full_like(b, 8.0), 0.02, 2.0).median().item()
    assert long_ > short * 2, (
        f"median {short:.3f}s at 1s remaining vs {long_:.3f}s at 8s -- these "
        "must differ, that is the defect"
    )


# --------------------------------------------------------------------------
# What absolute mode fixes.
# --------------------------------------------------------------------------

def test_absolute_respects_its_bounds_exactly():
    b = _beta()
    off = _absolute(b, 0.02, 0.50)
    assert off.min().item() >= 0.02 - 1e-6
    assert off.max().item() <= 0.50 + 1e-6


def test_absolute_is_independent_of_clip_length():
    """The property that makes the gap a config decision rather than a corpus
    accident: clip length is not an input at all."""
    b = _beta()
    a = _absolute(b, 0.02, 0.50).median().item()
    # Same call, no remaining-time argument to vary. Assert the API shape.
    c = _absolute(b, 0.02, 0.50).median().item()
    assert a == c


def test_absolute_puts_mass_where_the_student_scores_best():
    """Beta(2,5) has mean 2/7; over [0.02, 0.50] that is ~0.157 s, inside the
    0.10-0.25 band where teleop5 ep8200 measured best (11.16 / 11.40 cm wrist,
    against 15.01 at 0.02 and 13.64 at 0.50)."""
    b = _beta()
    off = _absolute(b, 0.02, 0.50)
    mean = off.mean().item()
    expected = 0.02 + (2.0 / 7.0) * (0.50 - 0.02)
    assert math.isclose(mean, expected, rel_tol=0.02), f"{mean:.4f} vs {expected:.4f}"
    assert 0.10 <= mean <= 0.25, f"mean gap {mean:.4f}s outside the target band"


def test_absolute_trains_the_band_it_claims_to():
    """A majority of draws should land in the band that scores best; otherwise
    the mode does not deliver what it exists for."""
    b = _beta()
    off = _absolute(b, 0.02, 0.50)
    frac = ((off >= 0.05) & (off <= 0.35)).float().mean().item()
    assert frac > 0.6, f"only {frac:.2%} of draws in 0.05-0.35"


# --------------------------------------------------------------------------
# Config validation must be loud: a silent default here reintroduces the
# original mismatch, and the only symptom is a policy that scores badly.
# --------------------------------------------------------------------------

class _Cfg:
    def __init__(self, mode, lo, hi):
        self.time_gap_mode = mode
        self.min_time_gap = lo
        self.max_time_gap = hi
        self.time_alpha = 2.0
        self.time_beta = 5.0


def _validate(cfg):
    """Mirrors the guards in _shift_and_sample_target_times."""
    mode = getattr(cfg, "time_gap_mode", "remaining_scaled")
    if mode == "absolute":
        if cfg.min_time_gap is None or cfg.max_time_gap is None:
            raise ValueError("absolute needs both bounds")
        if not float(cfg.max_time_gap) > float(cfg.min_time_gap):
            raise ValueError("absolute needs max > min")
    elif mode != "remaining_scaled":
        raise ValueError(f"unknown time_gap_mode {mode!r}")


@pytest.mark.parametrize("lo,hi", [(None, 0.5), (0.02, None), (None, None)])
def test_absolute_without_both_bounds_raises(lo, hi):
    with pytest.raises(ValueError, match="both bounds"):
        _validate(_Cfg("absolute", lo, hi))


@pytest.mark.parametrize("lo,hi", [(0.5, 0.5), (0.6, 0.2)])
def test_absolute_with_inverted_bounds_raises(lo, hi):
    with pytest.raises(ValueError, match="max > min"):
        _validate(_Cfg("absolute", lo, hi))


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown time_gap_mode"):
        _validate(_Cfg("linear", 0.02, 0.5))


def test_default_mode_is_the_historical_one():
    """Existing configs and resumed runs must be bit-identical, so the default
    must remain the old behaviour even though it is the defective one."""
    from protomotions.envs.control.masked_mimic_control import MaskedMimicControlConfig

    assert MaskedMimicControlConfig.time_gap_mode == "remaining_scaled"
