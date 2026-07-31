from __future__ import annotations

from pathlib import Path

_DEBUGFS_ROOT = Path("/sys/kernel/debug/tcp_bbr_qrl")
_PACING_GAIN_OVERRIDE_PATH = _DEBUGFS_ROOT / "pacing_gain_override"
_CRUISE_ACTIVE_PATH = _DEBUGFS_ROOT / "cruise_active"
_NO_OVERRIDE_SENTINEL = -1.0


def set_pacing_gain(gain: float) -> None:
    value = _NO_OVERRIDE_SENTINEL if gain is None else gain
    _PACING_GAIN_OVERRIDE_PATH.write_text(f"{value:.6f}\n")


def clear_pacing_gain_override() -> None:
    set_pacing_gain(None)


def in_probe_bw_cruise() -> bool:
    return _CRUISE_ACTIVE_PATH.read_text().strip() == "1"
