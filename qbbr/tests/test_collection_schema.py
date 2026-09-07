from qbbr.data.collection.schema import validate_collection_rows


def _complete_row(run_id: str, terminal: str, location: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "terminal_id": terminal,
        "geographic_location": location,
        "started_at_utc": "2026-09-01T00:00:00+00:00",
        "direction": "uplink",
        "cca": "bbr-v3",
        "cloud_region": "ap-southeast-2",
        "path_fingerprint": "a-b-c",
        "elevation_deg": 45.0,
        "weather": "clear",
        "handover_event": False,
        "snr_db": 11.0,
        "traceroute": "a b c",
        "mtr": "stable",
    }


def test_collection_readiness_requires_true_multi_location_telemetry() -> None:
    report = validate_collection_rows(
        [_complete_row("one", "dish-one", "Melbourne"), _complete_row("two", "dish-two", "Geelong")]
    )
    assert report.is_minimally_ready is True
    assert report.terminals == 2
    assert report.locations == 2


def test_missing_telemetry_is_reported_not_imputed() -> None:
    row = _complete_row("one", "dish-one", "Melbourne")
    row["snr_db"] = None
    report = validate_collection_rows([row])
    assert report.physical_telemetry_coverage["snr_db"] == 0.0
    assert report.is_minimally_ready is False
