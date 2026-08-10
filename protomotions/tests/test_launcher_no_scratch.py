# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard: the shared launcher must never route work through /oscar/scratch.

/oscar/scratch hit GRACE_EXPIRED during the 2026-08 campaign and destroyed the
eval harness plus three runs. Until 2026-08-10 ``launch_protomotions_ddp.sh``
still DEFAULTED ``MOTION_FILE``, ``PM_HELDOUT_FILE`` and ``LOGDIR`` into it.

The dangerous part was not that those paths were dead -- it is that they were
ALIVE. The scratch corpus path resolves (through a symlink) to a file
byte-identical in size to the data-local copy, with an md5-identical heldout
sidecar. A launch would have read the condemned filesystem and worked, right up
until the next quota expiry killed it mid-run. Stale-but-working is the state
that bites; stale-and-broken announces itself.

This test pins the repaired state so it cannot silently rot back. It is a
source-grep guard in the style of ``test_noise_scale_env_gates.py``.
"""

from pathlib import Path

import pytest

BANNED = "/oscar/scratch"

#: The launcher lives in the RUN TREE / imprint repo that vendors this package,
#: not in this package. Candidate locations, nearest first.
CANDIDATES = (
    Path(__file__).resolve().parents[2] / "launch_protomotions_ddp.sh",
    Path(__file__).resolve().parents[3] / "launch_protomotions_ddp.sh",
    Path(__file__).resolve().parents[4] / "launch_protomotions_ddp.sh",
)


def _launcher() -> Path:
    for path in CANDIDATES:
        if path.is_file():
            return path
    pytest.skip(
        "launch_protomotions_ddp.sh not found next to this checkout "
        "(bare ProtoMotions clone); the launcher guard is exercised where the "
        "launcher actually lives."
    )


def _scratch_lines(text: str):
    """Lines mentioning the banned prefix, minus the guards that BAN it.

    A guard has to name the thing it forbids, so a naive grep can never reach
    zero. Only lines that could make the launcher USE such a path are failures:
    the refusal patterns, their diagnostics and their prose are the fix, not the
    defect.
    """
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if BANNED not in line:
            continue
        stripped = line.strip()
        if stripped.startswith("#"):            # explanatory prose
            continue
        if "FATAL" in line or "GRACE_EXPIRED" in line:
            continue
        if stripped.startswith(f"{BANNED}*)") or stripped.startswith(f"*/{BANNED.lstrip('/')}*)"):
            continue                            # case-statement refusal patterns
        if "*/oscar/scratch*)" in stripped:
            continue
        out.append((i, stripped))
    return out


def test_launcher_defines_no_scratch_paths():
    path = _launcher()
    offenders = _scratch_lines(path.read_text())
    assert offenders == [], (
        f"{path} routes work through {BANNED}:\n"
        + "\n".join(f"  line {i}: {t}" for i, t in offenders)
        + f"\nThat filesystem hit GRACE_EXPIRED and destroyed runs this campaign. "
        "Use a path under /oscar/data/stellex/glvov/."
    )


def test_launcher_carries_a_runtime_no_scratch_guard():
    """Repointing the defaults is not enough -- an export could reintroduce it."""
    text = _launcher().read_text()
    assert "NO-SCRATCH RUNTIME GUARD" in text
    for var in ("MOTION_FILE", "PM_HELDOUT_FILE", "LOGDIR"):
        assert var in text, f"{var} must be covered by the no-scratch guard"
    # Symlinks are how a banned path hides: the scratch corpus WAS one.
    assert "readlink -f" in text, (
        "the guard must resolve symlinks before checking -- the scratch corpus "
        "path was itself a symlink to a differently-named file"
    )


def test_launcher_shader_cache_refuses_scratch():
    """The managed shader cache must not be relocatable onto scratch either."""
    text = _launcher().read_text()
    assert "PM_SHADER_CACHE_ROOT" in text
    assert "/oscar/scratch*)" in text, (
        "PM_SHADER_CACHE_ROOT must hard-refuse a scratch root"
    )
