import pytest

from qbbr.validation.bandwidth_recovery import fit_bandwidth_recovery


def _run_rows(run_id: str, start: float) -> list[dict]:
    # Target rate is 100; it first reaches 90 at t=2 after each event.
    return [
        {"run_id": run_id, "geographic_location": "Melbourne", "direction": "downlink", "offset_s": start, "handover_event": True, "delivery_rate_bps": 40.0},
        {"run_id": run_id, "geographic_location": "Melbourne", "direction": "downlink", "offset_s": start + 1, "handover_event": False, "delivery_rate_bps": 70.0},
        {"run_id": run_id, "geographic_location": "Melbourne", "direction": "downlink", "offset_s": start + 2, "handover_event": False, "delivery_rate_bps": 90.0},
        {"run_id": run_id, "geographic_location": "Melbourne", "direction": "downlink", "offset_s": start + 3, "handover_event": False, "delivery_rate_bps": 100.0},
    ]


def test_fit_bandwidth_recovery_requires_observed_handover_recovery() -> None:
    rows = _run_rows("one", 0.0) + _run_rows("two", 10.0) + _run_rows("three", 20.0)
    fit = fit_bandwidth_recovery(rows, min_events=3)[0]
    assert fit.events_used == 3
    assert fit.median_recovery_to_target_s == 2.0
    assert fit.recovery_time_constant_s > 0.0


def test_fit_bandwidth_recovery_refuses_insufficient_events() -> None:
    with pytest.raises(ValueError, match="enough handover"):
        fit_bandwidth_recovery(_run_rows("one", 0.0), min_events=3)
