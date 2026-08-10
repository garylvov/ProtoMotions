"""Guard tests for the v66 per-DOF limits_dof_pos margin. Lands with the change."""
import torch, pytest
from protomotions.envs.rewards.regularization import compute_soft_pos_limit_rew as f

LO = torch.tensor([-0.897, -0.120]); UP = torch.tensor([0.524, 2.190])
AT_LIMIT   = torch.tensor([[-0.897, 2.190]])
NEAR_LIMIT = torch.tensor([[-0.890, 2.185]])
MID        = torch.tensor([[0.0, 0.0]])


def test_scalar_zero_is_byte_identical_to_default():
    """RESUME RULE: frozen configs lacking the key must be unchanged."""
    assert torch.equal(f(MID, LO, UP), f(MID, LO, UP, soft_margin_frac=0.0))


def test_per_dof_equals_scalar_when_uniform():
    """The new code path must reproduce the old one exactly when uniform."""
    for x in (AT_LIMIT, NEAR_LIMIT, MID):
        assert torch.allclose(
            f(x, LO, UP, soft_margin_frac=0.05, proximity_scale=0.1),
            f(x, LO, UP, soft_margin_frac=[0.05, 0.05], proximity_scale=0.1),
        )


def test_narrow_band_does_NOT_remove_the_at_limit_charge():
    """THE finding that set v66's value to 0.0 rather than a token 0.005 band.

    prox = ((margin - dist)/margin).clamp(0,1) = 1.0 whenever dist == 0, for ANY
    band width. The deep_hinge_crouch reference SITS AT the ankle limit on 10.5%
    of (frame,ankle) pairs, so a token band would have left the contradiction it
    was chosen to resolve.
    """
    wide = f(AT_LIMIT, LO, UP, soft_margin_frac=0.05, proximity_scale=0.1)
    thin = f(AT_LIMIT, LO, UP, soft_margin_frac=0.005, proximity_scale=0.1)
    assert torch.allclose(wide, thin), "at-limit charge must be band-independent"
    assert torch.allclose(wide, torch.tensor([0.2]))          # 2 joints x prox_scale
    # ...but the APPROACH ramp does shrink, which is what a token band buys.
    assert f(NEAR_LIMIT, LO, UP, 0.005, 0.1) < f(NEAR_LIMIT, LO, UP, 0.05, 0.1)


def test_zero_margin_removes_at_limit_charge_but_keeps_violation_charge():
    """v66's actual setting. Proximity off; genuine violations still charged."""
    assert torch.allclose(
        f(AT_LIMIT, LO, UP, soft_margin_frac=[0.0, 0.0], proximity_scale=0.1),
        torch.tensor([0.0]),
    )
    over = torch.tensor([[-1.097, 2.190]])          # 0.2 rad past the lower limit
    assert torch.allclose(
        f(over, LO, UP, soft_margin_frac=[0.0, 0.0], proximity_scale=0.1),
        torch.tensor([0.2]),
    ), "base out_of_limits term must still guard real violations"


def test_only_the_four_intended_joints_are_freed():
    """v66 frees ankle_pitch+knee only; the other 23 DOFs keep their band."""
    lo = torch.zeros(27) - 1.0; up = torch.zeros(27) + 1.0
    m = [0.05] * 27
    for i in (3, 4, 9, 10):
        m[i] = 0.0
    at = (torch.zeros(1, 27) - 1.0)                  # every joint at its lower limit
    per = f(at, lo, up, soft_margin_frac=m, proximity_scale=0.1)
    uni = f(at, lo, up, soft_margin_frac=0.05, proximity_scale=0.1)
    assert torch.allclose(uni, torch.tensor([2.7]))          # 27 x 0.1
    assert torch.allclose(per, torch.tensor([2.3]))          # 23 x 0.1
