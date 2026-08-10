# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard tests for the RESUME reward-component injection gate
(``PM_RESUME_INJECT_COMPONENTS``) and for the launcher banner's world size.

DEFECT 1 (2026-08-10). ``resume_inject_reward_components`` CREATES reward
components absent from the frozen config whenever a ``PM_*_WEIGHT`` var is set.
``launch_protomotions_ddp.sh`` exports defaults for three of them, so every lane
that sourced that launcher and then resumed anything inherited reward surgery
*by omission*; the only suppression was ``launch_mm_v2.sh`` exporting the three
vars as empty strings, a trick no other caller performs. A 4-rank masked-mimic
smoke ate 12 ``RESUME INJECT`` lines (3 components x 4 ranks) into a
pure-supervision student whose frozen config had no reward components at all.
The contract pinned here:

1. **Hard no-op without the opt-in.** All three weight vars carrying non-empty
   values and no gate => not one component created, not one weight patched, the
   frozen dict byte-identical.
2. **Never silent.** A set-but-inert weight var earns exactly ONE loud
   ``RESUME INJECT SKIPPED`` line naming it.
3. **Armed, the historical behaviour is exactly what it was** -- same
   components, same weights, same ``RESUME INJECT`` proof lines.
4. **Fresh builds are untouched.** The launcher still defaults the three
   weights to 0.1 / -0.5 / -0.1, and never arms the gate for anyone.

DEFECT 2 (same day). The banner printed ``world_size=$WORLD`` as fact, but on a
resume ``detect_checkpoint_mode`` restores the FROZEN ``ngpu`` onto args, so the
launcher's request is discarded. Observed: banner said ``world_size=2`` while
Lightning ran 4 ranks. The banner may no longer assert a world size it cannot
know.
"""

import pathlib
import pickle

import pytest

import protomotions.envs.component_factories as factories

# tests/ -> protomotions/ -> ProtoMotions/ -> third_party/ -> <run tree>/
# The launchers belong to the RUN tree that vendors this repo, not to this
# repo, so a bare ProtoMotions checkout legitimately has no such file and skips.
_RUN_TREE = pathlib.Path(__file__).resolve().parents[4]

_WEIGHT_VARS = (
    "PM_CONTACT_MATCH_WEIGHT",
    "PM_LIFTOFF_PENALTY_WEIGHT",
    "PM_ACTION_SMOOTH_LME_WEIGHT",
)
_ALL_SET = {
    "PM_CONTACT_MATCH_WEIGHT": "0.1",
    "PM_LIFTOFF_PENALTY_WEIGHT": "-0.5",
    "PM_ACTION_SMOOTH_LME_WEIGHT": "-0.1",
}


def _run_tree_text(relpath):
    """Read a run-tree file, or skip on a bare ProtoMotions checkout."""
    path = _RUN_TREE / relpath
    if not path.is_file():
        pytest.skip(f"no vendoring run tree at {_RUN_TREE} (bare checkout)")
    return path.read_text()


# =============================================================================
# DEFECT 1 -- the opt-in gate
# =============================================================================


def test_injection_is_a_hard_noop_without_the_opt_in_gate():
    """The three launcher defaults set, no gate => NOTHING happens."""
    frozen = {"some_existing": factories.pow_rew_factory(weight=-1e-4)}
    before = pickle.dumps(frozen["some_existing"].static_params)
    lines = []

    changed = factories.resume_inject_reward_components(
        frozen, env=dict(_ALL_SET), log_fn=lines.append
    )

    assert changed is False
    # No component created ...
    assert set(frozen) == {"some_existing"}
    # ... and the surviving one is byte-identical, not merely equal.
    assert pickle.dumps(frozen["some_existing"].static_params) == before
    # ... and nothing claimed otherwise.
    assert not [l for l in lines if l.startswith("RESUME INJECT component ")]
    assert not [l for l in lines if l.startswith("RESUME override ")]


def test_the_gate_also_blocks_the_override_patch_path():
    """An already-present component is not silently re-priced either.

    Patching the weight of a frozen component looks safer than creating one,
    but it is the same defect wearing a different hat: a lane that never asked
    to touch rewards must not re-price them because of a launcher default.
    """
    frozen = {"contact_match": factories.contact_match_rew_factory(weight=0.03)}
    lines = []

    changed = factories.resume_inject_reward_components(
        frozen, env={"PM_CONTACT_MATCH_WEIGHT": "0.9"}, log_fn=lines.append
    )

    assert changed is False
    assert frozen["contact_match"].static_params["weight"] == 0.03
    assert len([l for l in lines if l.startswith("RESUME INJECT SKIPPED")]) == 1


def test_a_set_but_inert_weight_var_is_never_silent():
    """ONE loud SKIPPED line, naming every var it disarmed."""
    lines = []
    factories.resume_inject_reward_components(
        {}, env=dict(_ALL_SET), log_fn=lines.append
    )
    skipped = [l for l in lines if l.startswith("RESUME INJECT SKIPPED")]
    assert len(skipped) == 1, lines
    for var in _WEIGHT_VARS:
        assert var in skipped[0]
    # It must say what to do about it.
    assert factories.RESUME_INJECT_GATE_VAR in skipped[0]

    # A single set var is enough to earn the line, and only it is named.
    lines = []
    factories.resume_inject_reward_components(
        {}, env={"PM_LIFTOFF_PENALTY_WEIGHT": "-0.5"}, log_fn=lines.append
    )
    assert len(lines) == 1
    assert "PM_LIFTOFF_PENALTY_WEIGHT" in lines[0]
    assert "PM_CONTACT_MATCH_WEIGHT" not in lines[0]

    # Nothing set at all => nothing to warn about; total silence.
    lines = []
    assert (
        factories.resume_inject_reward_components({}, env={}, log_fn=lines.append)
        is False
    )
    assert lines == []

    # An EMPTY weight var is "not set" and must not produce a SKIPPED line
    # either -- it is already inert by intent (launch_mm_v2.sh's old trick).
    lines = []
    factories.resume_inject_reward_components(
        {},
        env={k: "" for k in _WEIGHT_VARS},
        log_fn=lines.append,
    )
    assert lines == []


def test_with_the_gate_armed_injection_works_exactly_as_before():
    """The historical behaviour, unchanged, behind one explicit opt-in."""
    frozen = {}
    lines = []
    changed = factories.resume_inject_reward_components(
        frozen,
        env=dict(_ALL_SET, PM_RESUME_INJECT_COMPONENTS="1"),
        log_fn=lines.append,
    )

    assert changed is True
    assert set(frozen) == {"contact_match", "liftoff_penalty", "action_smooth_lme"}
    assert frozen["contact_match"].static_params["weight"] == 0.1
    assert frozen["contact_match"].static_params["match_reward"] is True
    assert frozen["liftoff_penalty"].static_params["weight"] == -0.5
    assert frozen["action_smooth_lme"].static_params["weight"] == -0.1
    assert len([l for l in lines if l.startswith("RESUME INJECT component ")]) == 3
    assert not [l for l in lines if l.startswith("RESUME INJECT SKIPPED")]

    # Armed, the override path rides later resumes too.
    lines2 = []
    assert factories.resume_inject_reward_components(
        frozen,
        env=dict(_ALL_SET, PM_RESUME_INJECT_COMPONENTS="1",
                 PM_CONTACT_MATCH_WEIGHT="0.2"),
        log_fn=lines2.append,
    )
    assert frozen["contact_match"].static_params["weight"] == 0.2
    assert any("RESUME override contact_match.weight = 0.2" in l for l in lines2)


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " 1 "])
def test_gate_true_spellings(raw):
    assert factories.resume_inject_gate_requested({"PM_RESUME_INJECT_COMPONENTS": raw})


@pytest.mark.parametrize("raw", ["", "0", "false", "NO", "off", None])
def test_gate_false_spellings(raw):
    env = {} if raw is None else {"PM_RESUME_INJECT_COMPONENTS": raw}
    assert not factories.resume_inject_gate_requested(env)


@pytest.mark.parametrize("raw", ["2", "contact_match", "sure", "-1"])
def test_a_gate_you_can_typo_into_silence_is_not_a_gate(raw):
    with pytest.raises(ValueError) as exc:
        factories.resume_inject_gate_requested({"PM_RESUME_INJECT_COMPONENTS": raw})
    assert "PM_RESUME_INJECT_COMPONENTS" in str(exc.value)


def test_the_weight_var_table_covers_every_injectable_component():
    """READER/WRITER LAW: the SKIPPED reporter and the injector share a set.

    The SKIPPED line is built from ``RESUME_INJECTABLE_WEIGHT_VARS`` on a path
    that returns before the injector's own ``specs`` table exists, so a new
    injectable added to one table and not the other would be a component that
    can be injected but can never be reported as skipped.
    """
    assert set(factories.RESUME_INJECTABLE_WEIGHT_VARS) == set(
        factories.RESUME_INJECTABLE_COMPONENTS
    )
    source = (
        pathlib.Path(factories.__file__).read_text()
    )
    for name, var in factories.RESUME_INJECTABLE_WEIGHT_VARS.items():
        # Each pair must actually appear in the injector's specs table.
        assert f'("{name}", "{var}"' in source, f"{name}/{var} not in specs"


def test_train_agent_gates_the_deferral_promise_too():
    """"the pass below will create it" is a lie when the pass is disarmed."""
    source = (
        pathlib.Path(factories.__file__).resolve().parents[1] / "train_agent.py"
    ).read_text()
    assert "resume_inject_gate_requested" in source
    assert "RESUME override DEFERRED" in source
    # The deferral branch must be conditioned on the gate, not on membership
    # in the injectable set alone.
    assert "_comp in _INJECTABLE and _inject_armed()" in source


# =============================================================================
# DEFECT 1 -- the launchers (run-tree files; skipped on a bare checkout)
# =============================================================================


def test_fresh_build_weight_defaults_are_unchanged():
    """A teacher run is training off this lineage; fresh builds must not move."""
    text = _run_tree_text("launch_protomotions_ddp.sh")
    for var, default in (
        ("PM_CONTACT_MATCH_WEIGHT", "0.1"),
        ("PM_LIFTOFF_PENALTY_WEIGHT", "-0.5"),
        ("PM_ACTION_SMOOTH_LME_WEIGHT", "-0.1"),
    ):
        assert f'export {var}="${{{var}-{default}}}"' in text, (
            f"{var} fresh-build default is no longer {default}"
        )


def test_the_launcher_never_arms_the_gate_for_its_callers():
    """Opt-in means the OPERATOR opts in, not a launcher every lane sources."""
    text = _run_tree_text("launch_protomotions_ddp.sh")
    for armed in (
        "export PM_RESUME_INJECT_COMPONENTS=1",
        'export PM_RESUME_INJECT_COMPONENTS="1"',
        "export PM_RESUME_INJECT_COMPONENTS=${PM_RESUME_INJECT_COMPONENTS:-1}",
    ):
        assert armed not in text, f"launcher arms the gate: {armed}"
    # It must still SHOW the gate state, so a resume log proves it.
    assert "PM_RESUME_INJECT_COMPONENTS" in text


def test_mm_lane_no_longer_depends_on_the_empty_string_trick():
    """The mm lane's correctness must come from the gate, not from blanking."""
    text = _run_tree_text("launch_mm_v2.sh")
    assert "PM_RESUME_INJECT_COMPONENTS=0" in text


# =============================================================================
# DEFECT 2 -- the banner may not assert a world size it cannot know
# =============================================================================


def test_banner_never_prints_an_unqualified_world_size():
    text = _run_tree_text("launch_protomotions_ddp.sh")
    banner_lines = [
        l for l in text.splitlines() if l.lstrip().startswith('echo "[pm_ddp]')
    ]
    assert banner_lines
    offenders = [l for l in banner_lines if "world_size=" in l]
    assert not offenders, (
        "banner asserts a world size it cannot know on a resume: " + repr(offenders)
    )


def test_banner_reads_the_frozen_ngpu_when_a_resume_is_detected():
    text = _run_tree_text("launch_protomotions_ddp.sh")
    # It detects a resume the same way train_agent.py does ...
    assert 'results/$RUN_NAME' in text
    assert 'last.ckpt' in text
    # ... reads the frozen ngpu out of the saved args ...
    assert 'config.yaml' in text
    assert '"ngpu"' in text
    # ... and prints BOTH values, plus a loud line when they disagree.
    assert "world_size_requested=" in text
    assert "world_size_effective=" in text
    assert "WORLD-SIZE MISMATCH" in text
    # Degrades gracefully rather than guessing.
    assert "world_size_effective=unknown" in text
