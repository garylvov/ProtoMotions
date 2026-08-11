# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth + boot-time PROOF for the FK Cartesian loss weights.

Why this exists
---------------
The FK Cartesian wrist loss is the student-side twin of the mechanism the
teacher buys almost a quarter of its reward income with
(``wrist_relative_body_pos`` / ``global_wrist_pos`` /
``wrist_relative_body_ori``). Its weights therefore have to be *set*, and a run
whose set weight and trained weight disagree is worse than useless -- it
produces a number nobody can trust.

That disagreement is easy to produce here, because
``train_agent.py::detect_checkpoint_mode`` does NOT re-execute the experiment
file on a resume: ``results/<EXP>/config.yaml`` is written back onto ``args``
and ``resolved_configs.pt`` supplies the real config objects. So on a resume,
``--overrides agent.fk_wrist_pos_weight=0.3`` and any environment variable are
INERT, while the launcher script happily echoes the value the operator
intended. The only way to change the weight is to rewrite the frozen pickle
(``stage_resume_config.py --fk-wrist-pos-weight``), and the only way to KNOW
which weight is in force is to read it off the config object the agent will
actually train with.

This module provides both halves so they cannot drift:

* :func:`fk_loss_weights` -- the ONE accessor. The loss
  (``SupervisedAgent.calculate_extra_loss``) reads its weights through it, and
  so does the proof below. There is no second code path that could resolve a
  different number.
* :func:`log_fk_loss_proof` -- called once per boot from ``train_agent.py``,
  on the FINAL config object, for BOTH the fresh and resume wiring paths.

The proof line is emitted even when every weight is zero. "No line in the log"
must mean "this binary predates the proof", never "the term is off" -- an
absent line that could mean either is not a proof.
"""

from __future__ import annotations

from typing import Callable, Dict

#: Every FK Cartesian loss weight, in the order the proof prints them.
#: ``calculate_extra_loss`` reads exactly these fields and no others.
FK_LOSS_WEIGHT_FIELDS = (
    "fk_wrist_pos_weight",
    "fk_wrist_ori_weight",
    "fk_global_pos_weight",
)

#: Non-weight FK settings worth printing alongside, because a correct weight
#: against the wrong reference key still trains the wrong thing (the
#: ``masked_mimic_target_poses`` incident inflated fk_wrist_pos_loss to ~7.7).
FK_LOSS_CONTEXT_FIELDS = (
    ("fk_wrist_ref_key", "mimic_target_poses"),
    ("fk_wrist_root_rot_obs_key", "max_coords_obs"),
    ("fk_wrist_body_names", None),
    ("fk_global_body_names", None),
)


def fk_loss_weights(config) -> Dict[str, float]:
    """Return the EFFECTIVE FK loss weights for ``config``.

    ``getattr`` with a 0.0 default is load-bearing: ``resolved_configs.pt``
    pickles written before these fields existed unpickle into objects that
    simply lack them, and such a run must pay zero FK cost rather than crash.
    A ``None`` stored in the pickle is likewise treated as off.
    """
    weights = {}
    for name in FK_LOSS_WEIGHT_FIELDS:
        value = getattr(config, name, 0.0)
        weights[name] = 0.0 if value is None else float(value)
    return weights


def format_fk_loss_proof(config, label: str) -> list:
    """Render the boot-time proof lines for ``config``. Pure, for testing."""
    weights = fk_loss_weights(config)
    active = {k: v for k, v in weights.items() if v > 0}
    lines = []
    if not active:
        lines.append(
            f"[FK-LOSS] {label}: FK Cartesian loss OFF -- "
            + " ".join(f"{k}={v:g}" for k, v in weights.items())
        )
        return lines

    lines.append(
        f"[FK-LOSS] {label}: FK Cartesian loss ACTIVE (source: the config "
        "object this run will train with, read through "
        "fk_wrist_proof.fk_loss_weights -- the same accessor the loss uses)"
    )
    for name, value in weights.items():
        state = "ACTIVE" if value > 0 else "off"
        lines.append(f"[FK-LOSS] {label}:   {name:22s} = {value:<8g} {state}")
    for name, default in FK_LOSS_CONTEXT_FIELDS:
        lines.append(
            f"[FK-LOSS] {label}:   {name:22s} = {getattr(config, name, default)}"
        )
    lines.append(
        f"[FK-LOSS] {label}: READER -- supervised/fk_wrist_pos_loss, "
        "supervised/fk_wrist_ori_loss (both logged whenever the wrist pass "
        "runs, at ANY weight incl. 0) and supervised/fk_global_pos_loss in "
        "tensorboard. Realized loss share = weight * loss / "
        "losses/supervised_loss."
    )
    return lines


def log_fk_loss_proof(config, log_fn: Callable[[str], None], label: str) -> None:
    """Emit the boot-time proof. Never raises; never mutates ``config``."""
    for line in format_fk_loss_proof(config, label):
        log_fn(line)
