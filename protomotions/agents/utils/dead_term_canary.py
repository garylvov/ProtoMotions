"""Generic SILENTLY-DEAD reward-term canary.

WHY THIS EXISTS (v57 post-mortem, 2026-08-05). ``hold_joint_quiet`` shipped
registered, weighted 0.25, with a working still-mask writer -- and produced
``env/raw_r/hold_joint_quiet_mean == 0.00000`` (std also exactly 0) at step 1
and at step 5267 alike. Its Gaussian kernel ``exp(-25 * mean_j v^2)`` had been
sized against an eval-derived 0.06-0.15 rad/s while the live hold-window rms
was 6-8 rad/s, so ``exp(-1432)`` flushed to a hard zero with a hard-zero
gradient. Nothing complained for 5267 consecutive logging steps.

The guard that DID exist checked that the still-mask WRITER was present. It
passed. That is the whole lesson: a per-term structural precondition check
cannot see a numerical death downstream of it. The only check that catches
this class is the OUTPUT check -- and it must be GENERIC, because the next
term to die this way will not be ``hold_joint_quiet``.

WHAT COUNTS AS DEAD. A term is flagged only when ALL of these hold for
``epochs`` CONSECUTIVE logging epochs:

- it is REGISTERED and WEIGHTED: both ``raw_r/<name>_mean`` and
  ``scaled_r/<name>_mean`` are present (the framework only emits ``scaled_r``
  for a component whose weight is non-zero, so dormant weight-0 components --
  which are *supposed* to read zero -- never trip the canary);
- ``raw_r/<name>_mean`` is EXACTLY 0.0;
- ``raw_r/<name>_std`` is EXACTLY 0.0 when present. A term with zero mean but
  non-zero spread is alive and merely balanced, not dead. This second test is
  what keeps signed penalties and rare indicators out of the alarm.

Exact-zero, not near-zero, on purpose: this canary must have a ZERO false
positive rate against sparse-but-live terms (``fall_penalty`` runs at 1.7e-4
in v57 and must never be flagged). A term that is genuinely, permanently
identically zero across every env and every step of an epoch is either dead or
mis-gated, and both deserve the same loud line.

Rule 10: this module NEVER touches a tensor that feeds training. It reads the
already-reduced scalar log dict and returns extra log keys ONLY for terms it
actually flags, so a healthy unset run's TB surface is byte-identical.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

RAW_PREFIX = "env/raw_r/"
SCALED_PREFIX = "env/scaled_r/"
MEAN_SUFFIX = "_mean"
STD_SUFFIX = "_std"

DEFAULT_EPOCHS = 50


def _as_float(value) -> Optional[float]:
    """Best-effort scalar extraction from a float / 0-d or 1-elem Tensor."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    item = getattr(value, "item", None)
    if item is None:
        return None
    try:
        return float(item())
    except (ValueError, RuntimeError):
        return None


class DeadTermCanary:
    """Streak counter over exactly-zero weighted reward terms.

    Args:
        epochs: Consecutive all-zero logging epochs before the first warning
            (and the re-warn period thereafter). 0 disables the canary.
        ignore: Term names never flagged (for a term whose exact zero is a
            deliberate, documented invariant).
    """

    def __init__(self, epochs: int = DEFAULT_EPOCHS, ignore: Iterable[str] = ()):
        self.epochs = max(0, int(epochs))
        self.ignore = set(ignore)
        self.streaks: Dict[str, int] = {}

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "DeadTermCanary":
        """Build from ``PM_DEAD_TERM_CANARY_EPOCHS`` / ``..._IGNORE``."""
        env = os.environ if env is None else env
        raw_epochs = env.get("PM_DEAD_TERM_CANARY_EPOCHS")
        epochs = int(raw_epochs) if raw_epochs not in (None, "") else DEFAULT_EPOCHS
        raw_ignore = env.get("PM_DEAD_TERM_CANARY_IGNORE") or ""
        ignore = [n.strip() for n in raw_ignore.split(",") if n.strip()]
        return cls(epochs=epochs, ignore=ignore)

    def weighted_term_names(self, log_dict: Dict) -> List[str]:
        """Terms that are BOTH registered (raw_r) and weighted (scaled_r)."""
        raw = {
            k[len(RAW_PREFIX):-len(MEAN_SUFFIX)]
            for k in log_dict
            if k.startswith(RAW_PREFIX) and k.endswith(MEAN_SUFFIX)
        }
        scaled = {
            k[len(SCALED_PREFIX):-len(MEAN_SUFFIX)]
            for k in log_dict
            if k.startswith(SCALED_PREFIX) and k.endswith(MEAN_SUFFIX)
        }
        return sorted(raw & scaled)

    def is_exactly_zero(self, log_dict: Dict, name: str) -> bool:
        """True when this epoch's raw mean AND (present) std are exactly 0.0."""
        mean = _as_float(log_dict.get(f"{RAW_PREFIX}{name}{MEAN_SUFFIX}"))
        if mean is None or mean != 0.0:
            return False
        std = _as_float(log_dict.get(f"{RAW_PREFIX}{name}{STD_SUFFIX}"))
        return std is None or std == 0.0

    def update(self, log_dict: Dict) -> Tuple[Dict[str, float], List[str]]:
        """Advance one logging epoch.

        Args:
            log_dict: The assembled scalar log dict (``env/raw_r/...`` keys).

        Returns:
            ``(extra_log_keys, warnings)``. ``extra_log_keys`` carries
            ``canary/dead_term_epochs/<name>`` for every currently-flagged term
            and is EMPTY when nothing is flagged. ``warnings`` holds the loud
            multi-line blocks to print (empty on a healthy epoch).
        """
        if self.epochs <= 0:
            return {}, []

        extras: Dict[str, float] = {}
        warnings: List[str] = []
        for name in self.weighted_term_names(log_dict):
            if name in self.ignore:
                continue
            if not self.is_exactly_zero(log_dict, name):
                self.streaks.pop(name, None)
                continue
            streak = self.streaks.get(name, 0) + 1
            self.streaks[name] = streak
            if streak < self.epochs:
                continue
            extras[f"canary/dead_term_epochs/{name}"] = float(streak)
            if streak % self.epochs == 0:
                warnings.append(self._format_warning(name, streak, log_dict))
        return extras, warnings

    def _format_warning(self, name: str, streak: int, log_dict: Dict) -> str:
        weight = _as_float(log_dict.get(f"{SCALED_PREFIX}{name}{MEAN_SUFFIX}"))
        return (
            "\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "[dead-term-canary] SILENTLY-DEAD REWARD TERM: "
            f"'{name}'\n"
            f"  env/raw_r/{name}_mean AND _std have been EXACTLY 0.0 for "
            f"{streak} consecutive logging epochs.\n"
            f"  The term IS registered and IS weighted (scaled_r present, "
            f"contributing {weight}), so it is costing the return nothing "
            "while occupying a weight.\n"
            "  Likely causes, in the order they have actually happened here:\n"
            "    1. KERNEL UNDERFLOW -- exp(-c*x) sized for a distribution the\n"
            "       live run is orders of magnitude away from (hold_joint_quiet,\n"
            "       v57: exp(-25*57) == 0.0 exactly, zero gradient).\n"
            "    2. A GATE/MASK that is never True (check the mask's own stat).\n"
            "    3. A dynamic_var wired to a buffer nobody writes.\n"
            "  Check the term's own diagnostic stats BEFORE re-scaling anything:\n"
            "  a dead term with a healthy mask is a SCALING failure, and a\n"
            "  rescaled constant is not a root-cause fix.\n"
            "  Silence this ONLY with PM_DEAD_TERM_CANARY_IGNORE="
            f"{name} and a written justification.\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        )
