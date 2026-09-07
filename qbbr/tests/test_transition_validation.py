from qbbr.validation.transitions import transition_fit_by_scenario


def test_transition_fit_is_reported_per_location_and_direction() -> None:
    rows = [
        {"geographic_location": "Melbourne", "direction": "uplink", "offset_s": 0, "predicted_bbr_state": "PROBE_BW", "bbr_state": "PROBE_BW"},
        {"geographic_location": "Melbourne", "direction": "uplink", "offset_s": 1, "predicted_bbr_state": "PROBE_RTT", "bbr_state": "PROBE_RTT"},
        {"geographic_location": "Melbourne", "direction": "uplink", "offset_s": 2, "predicted_bbr_state": "PROBE_BW", "bbr_state": "PROBE_BW"},
    ]
    result = transition_fit_by_scenario(rows)
    assert len(result) == 1
    assert result[0].precision == 1.0
    assert result[0].recall == 1.0
    assert result[0].f1 == 1.0
