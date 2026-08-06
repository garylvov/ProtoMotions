"""Guard tests for the generic silently-dead reward-term canary.

The class of defect these lock down is the v57 ``hold_joint_quiet`` failure:
a REGISTERED, WEIGHTED reward term whose kernel underflowed to an exact 0.0
(and an exact-0 gradient) for 5267 consecutive logging steps with no warning,
because the only existing guard checked that its still-mask WRITER existed --
a structural precondition that passed while the numerics died downstream.
"""

import torch

from protomotions.agents.utils.dead_term_canary import (
    DEFAULT_EPOCHS,
    DeadTermCanary,
)


def _log(**terms):
    """Build a log dict in the agent's exact key shape.

    terms: name -> (raw_mean, raw_std, scaled_mean_or_None). A None scaled mean
    models a DORMANT (weight-0) component, for which the framework emits no
    ``scaled_r`` key at all.
    """
    out = {}
    for name, (raw_mean, raw_std, scaled) in terms.items():
        out[f"env/raw_r/{name}_mean"] = torch.tensor(raw_mean)
        out[f"env/raw_r/{name}_std"] = torch.tensor(raw_std)
        if scaled is not None:
            out[f"env/scaled_r/{name}_mean"] = torch.tensor(scaled)
    return out


def test_canary_fires_on_the_exact_v57_hold_joint_quiet_signature():
    """The real failure: raw mean AND std exactly 0, term weighted, forever."""
    canary = DeadTermCanary(epochs=5)
    v57 = _log(hold_joint_quiet=(0.0, 0.0, 0.0))

    for epoch in range(1, 5):
        extras, warnings = canary.update(v57)
        assert extras == {}, f"warned too early at epoch {epoch}: {extras}"
        assert warnings == []

    extras, warnings = canary.update(v57)
    assert extras == {"canary/dead_term_epochs/hold_joint_quiet": 5.0}
    assert len(warnings) == 1
    body = warnings[0]
    assert "hold_joint_quiet" in body
    assert "SILENTLY-DEAD REWARD TERM" in body
    # The warning must name the actual root-cause class, not just "is zero".
    assert "UNDERFLOW" in body
    assert "PM_DEAD_TERM_CANARY_IGNORE" in body


def test_canary_re_warns_periodically_but_not_every_epoch():
    canary = DeadTermCanary(epochs=3)
    dead = _log(dead_term=(0.0, 0.0, 0.0))
    warn_epochs = [
        epoch for epoch in range(1, 13) if canary.update(dead)[1]
    ]
    assert warn_epochs == [3, 6, 9, 12]
    # ...and stays flagged in the TB surface on every epoch in between.
    assert canary.streaks["dead_term"] == 12


def test_canary_never_flags_a_live_term_however_small():
    """Zero false positives against sparse-but-live terms.

    ``fall_penalty`` ran at 1.7e-4 for the whole of v57 and is NOT dead.
    """
    canary = DeadTermCanary(epochs=2)
    live = _log(
        fall_penalty=(1.69436e-4, 1.3e-2, -3.38872e-4),
        global_wrist_pos=(0.958434, 0.218323, 0.383374),
    )
    for _ in range(50):
        extras, warnings = canary.update(live)
        assert extras == {} and warnings == []
    assert canary.streaks == {}


def test_canary_ignores_a_zero_mean_term_that_still_has_spread():
    """A signed term that nets to zero is BALANCED, not dead."""
    canary = DeadTermCanary(epochs=2)
    balanced = _log(signed_term=(0.0, 0.4, 0.0))
    for _ in range(20):
        assert canary.update(balanced) == ({}, [])


def test_canary_ignores_dormant_weight_zero_components():
    """No ``scaled_r`` key == not weighted == expected to read zero."""
    canary = DeadTermCanary(epochs=2)
    dormant = _log(dormant_term=(0.0, 0.0, None))
    for _ in range(20):
        assert canary.update(dormant) == ({}, [])


def test_canary_streak_resets_the_moment_the_term_comes_alive():
    canary = DeadTermCanary(epochs=3)
    dead = _log(t=(0.0, 0.0, 0.0))
    alive = _log(t=(0.02, 0.05, 0.005))
    canary.update(dead)
    canary.update(dead)
    canary.update(alive)
    assert canary.streaks == {}
    extras, warnings = canary.update(dead)
    assert extras == {} and warnings == []


def test_canary_honours_ignore_list_and_disable_switch():
    silenced = DeadTermCanary(epochs=1, ignore=["t"])
    assert silenced.update(_log(t=(0.0, 0.0, 0.0))) == ({}, [])

    off = DeadTermCanary(epochs=0)
    assert off.update(_log(t=(0.0, 0.0, 0.0))) == ({}, [])


def test_canary_from_env_defaults_and_overrides():
    assert DeadTermCanary.from_env({}).epochs == DEFAULT_EPOCHS
    assert DeadTermCanary.from_env({}).ignore == set()
    built = DeadTermCanary.from_env(
        {
            "PM_DEAD_TERM_CANARY_EPOCHS": "7",
            "PM_DEAD_TERM_CANARY_IGNORE": " a , b ,",
        }
    )
    assert built.epochs == 7
    assert built.ignore == {"a", "b"}


def test_canary_accepts_plain_floats_as_well_as_tensors():
    canary = DeadTermCanary(epochs=1)
    extras, warnings = canary.update(
        {
            "env/raw_r/t_mean": 0.0,
            "env/raw_r/t_std": 0.0,
            "env/scaled_r/t_mean": 0.0,
        }
    )
    assert extras == {"canary/dead_term_epochs/t": 1.0}
    assert len(warnings) == 1


def test_agent_post_epoch_logging_wires_the_canary_generically():
    """The hook must live in the agent's shared logging path, not in one term."""
    import inspect

    from protomotions.agents.base_agent.agent import BaseAgent

    src = inspect.getsource(BaseAgent.post_epoch_logging)
    assert "DeadTermCanary" in src
    assert "canary.update(log_dict)" in src
    # ...and must never special-case the term that motivated it.
    assert "hold_joint_quiet" not in src
