from qbbr.control.contracts import RunManifest
from qbbr.data.collection.records import FieldRun, IntervalObservation


def test_field_run_flattens_manifest_with_interval_observations() -> None:
    run = FieldRun(
        manifest=RunManifest.now(
            run_id="run-1",
            terminal_id="dish-1",
            geographic_location="Melbourne",
            direction="uplink",
            cca="bbr-v3",
            contract_id="native-rl-bbr-v1",
        ),
        intervals=(
            IntervalObservation(
                offset_s=0.0,
                delivery_rate_bps=1.0,
                rtt_s=0.02,
                retransmission_rate=0.0,
                bbr_state="PROBE_BW",
            ),
        ),
    )
    rows = run.flattened_rows()
    assert rows[0]["terminal_id"] == "dish-1"
    assert rows[0]["bbr_state"] == "PROBE_BW"
