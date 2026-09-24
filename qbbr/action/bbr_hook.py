from __future__ import annotations

from pathlib import Path

_DEBUGFS_ROOT = Path("/sys/kernel/debug/tcp_bbr_qrl")
_PACING_GAIN_OVERRIDE_PATH = _DEBUGFS_ROOT / "pacing_gain_override"
_CRUISE_ACTIVE_PATH = _DEBUGFS_ROOT / "cruise_active"
_NO_OVERRIDE_SENTINEL = -1.0
_GAIN_LEVELS = (0.75, 0.90, 1.00, 1.10, 1.25)


def set_pacing_gain(gain: float | None) -> None:
    """Refresh the kernel's one-second command lease; None disables it.

    Requests latch on CRUISE entry. Kernel readback is quantized (e.g.
    1.10 becomes 282/256); it is not a new admissible command value.
    """
    value = _NO_OVERRIDE_SENTINEL if gain is None else gain
    if isinstance(value, bool) or value not in (*_GAIN_LEVELS, _NO_OVERRIDE_SENTINEL):
        raise ValueError("Expected one of five declared gains, or None/-1 to clear.")
    _PACING_GAIN_OVERRIDE_PATH.write_text(f"{value:.6f}\n")


def clear_pacing_gain_override() -> None:
    set_pacing_gain(None)


def in_probe_bw_cruise() -> bool:
    return _CRUISE_ACTIVE_PATH.read_text().strip() == "1"
