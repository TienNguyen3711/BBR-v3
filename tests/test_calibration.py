from __future__ import annotations

from qbbr.features.calibration import compute_calibration


def test_calibration_ranking_matches_known_paths(dataset_root):
    """Sanity-checks ranking against real data, not exact match to the paper's
    rounded illustrative figures (which Table 1 explicitly says should be
    re-estimated from the dataset)."""
    calib = compute_calibration(dataset_root)

    b_max_downlink = {loc: v["downlink"]["B_max_mbps"] for loc, v in calib.items()}
    rtt_min_downlink = {loc: v["downlink"]["RTT_min_ms"] for loc, v in calib.items()}
    rtt_max_downlink = {loc: v["downlink"]["RTT_max_ms"] for loc, v in calib.items()}

    assert max(b_max_downlink, key=b_max_downlink.get) == "Sydney"
    assert min(rtt_min_downlink, key=rtt_min_downlink.get) == "Sydney"
    assert max(rtt_max_downlink, key=rtt_max_downlink.get) == "SaoPaulo"

    for loc, directions in calib.items():
        for direction, constants in directions.items():
            assert constants["B_max_mbps"] > 0
            assert 0 < constants["RTT_min_ms"] < constants["RTT_max_ms"]
