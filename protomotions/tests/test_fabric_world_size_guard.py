# SPDX-License-Identifier: Apache-2.0
"""Guard tests for the world-size request/reality invariant.

Root cause these protect (2026-08-11, job 4863416 step 15): `--ngpu 8` is only
a REQUEST. Lightning decides the real world size from the ClusterEnvironment it
auto-detects. `srun` exports SLURM_NTASKS; a plain `sbatch` that never asked
for `--ntasks` does not. So the SAME launcher that builds 8 ranks under sbatch
built ONE rank under `srun --overlap --ntasks=1`, spawned nobody, and OOMed
trying to put 8x8192 environments on GPU 0 -- with "Starting with 1 processes"
as the only warning.

Two halves, and both must hold or the guard proves nothing:
  * the CHECK (fabric_config.assert_world_size_matches_request) turns the
    mismatch into a loud refusal before any training happens, and
  * the FIX (launch_protomotions_ddp.sh exporting SLURM_JOB_NAME=interactive,
    Lightning's documented manual-launch switch) stops the mismatch arising in
    the first place, for every lane, however it was invoked.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from protomotions.utils.fabric_config import assert_world_size_matches_request

LAUNCHER = Path("/oscar/data/stellex/glvov/mm_run_v2/launch_protomotions_ddp.sh")


# --------------------------------------------------------------------------
# the CHECK
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "world_size,ngpu,nodes",
    [(1, 1, 1), (8, 8, 1), (4, 4, 1), (16, 8, 2), (24, 8, 3)],
)
def test_matching_world_size_is_accepted(world_size, ngpu, nodes):
    assert_world_size_matches_request(world_size, ngpu, nodes, env={}) is None


def test_srun_collapse_to_one_rank_is_refused():
    """The exact 2026-08-11 failure: 8 requested, SLURMEnvironment gave 1."""
    env = {
        "SLURM_JOB_ID": "4863416",
        "SLURM_JOB_NAME": "mm_dofprofile",
        "SLURM_STEP_ID": "15",
        "SLURM_NTASKS": "1",
    }
    with pytest.raises(RuntimeError) as exc:
        assert_world_size_matches_request(1, 8, 1, env=env)
    msg = str(exc.value)
    assert "world_size=1" in msg and "= 8 rank(s)" in msg
    # It must name the cause and the two escapes, or the operator is left
    # staring at an OOM traceback again.
    assert "SLURMEnvironment" in msg
    assert "SLURM_JOB_NAME=interactive" in msg
    assert "--ntasks=8" in msg
    assert "SLURM_NTASKS" in msg and "4863416" in msg


def test_world_larger_than_request_is_also_refused():
    with pytest.raises(RuntimeError):
        assert_world_size_matches_request(8, 4, 1, env={})


def test_multinode_request_is_ngpu_times_nodes():
    with pytest.raises(RuntimeError) as exc:
        assert_world_size_matches_request(8, 8, 2, env={})
    assert "= 16 rank(s)" in str(exc.value)


def test_no_slurm_env_still_refuses_but_offers_no_slurm_hint():
    """A non-SLURM mismatch is still fatal; it just must not blame srun."""
    with pytest.raises(RuntimeError) as exc:
        assert_world_size_matches_request(1, 8, 1, env={})
    msg = str(exc.value)
    assert "SLURM env: <none>" in msg
    assert "SLURMEnvironment" not in msg


def test_interactive_job_name_suppresses_the_srun_hint():
    """With the fix applied the cause is elsewhere, so do not misattribute it."""
    env = {"SLURM_NTASKS": "1", "SLURM_JOB_NAME": "interactive"}
    with pytest.raises(RuntimeError) as exc:
        assert_world_size_matches_request(1, 8, 1, env=env)
    assert "SLURMEnvironment" not in str(exc.value)


# --------------------------------------------------------------------------
# the FIX
# --------------------------------------------------------------------------
@pytest.mark.skipif(not LAUNCHER.is_file(), reason="launcher not on this host")
def test_launcher_neutralizes_inherited_srun_task_environment():
    text = LAUNCHER.read_text()
    assert "export SLURM_JOB_NAME=interactive" in text, (
        "launch_protomotions_ddp.sh must neutralize an inherited srun task "
        "environment; without it, running the launcher inside `srun` collapses "
        "the world size to SLURM_NTASKS"
    )
    # ...and it must happen BEFORE python is started, or it is decorative.
    idx_fix = text.index("export SLURM_JOB_NAME=interactive")
    idx_py = text.index('"$PY" protomotions/train_agent.py')
    assert idx_fix < idx_py


@pytest.mark.skipif(not LAUNCHER.is_file(), reason="launcher not on this host")
def test_train_agent_calls_the_guard_after_fabric_launch():
    train_agent = LAUNCHER.parent / "third_party/ProtoMotions/protomotions/train_agent.py"
    text = train_agent.read_text()
    assert "assert_world_size_matches_request" in text
    assert re.search(
        r"fabric\.launch\(\)[\s\S]{0,800}?assert_world_size_matches_request\(", text
    ), "the guard must run immediately after fabric.launch(), before any env build"
