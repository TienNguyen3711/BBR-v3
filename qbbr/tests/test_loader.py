from __future__ import annotations

import pytest

from qbbr.data.catalog import build_catalog, iter_file_records
from qbbr.data.loader import load_trace


def test_load_single_downlink_trace(dataset_root):
    df = build_catalog(dataset_root)
    row = df[(df.cca == "bbr") & (df.direction == "downlink")].iloc[0]
    record = next(iter_file_records(df[df.index == row.name]))
    trace = load_trace(record)

    assert trace.summary["sender_tcp_congestion"] == "bbr"
    assert 300 <= len(trace.intervals) <= 319
    assert not trace.intervals["rtt_ms"].isna().any()
    assert not trace.intervals["bits_per_second"].isna().any()


def test_load_single_uplink_trace(dataset_root):
    df = build_catalog(dataset_root)
    row = df[(df.cca == "bbr") & (df.direction == "uplink")].iloc[0]
    record = next(iter_file_records(df[df.index == row.name]))
    trace = load_trace(record)

    assert trace.summary["sender_tcp_congestion"] == "bbr"
    assert trace.summary["sum_received"]["bytes"] > 0


@pytest.mark.slow
def test_full_corpus_loads_consistently(dataset_root):
    df = build_catalog(dataset_root)
    uplink_zero_received = 0
    uplink_total = 0

    for record in iter_file_records(df):
        trace = load_trace(record)

        assert trace.summary.get("sender_tcp_congestion") == record.cca
        assert 300 <= len(trace.intervals) <= 319

        sum_received = trace.summary.get("sum_received", {}).get("bytes")
        if record.direction == "downlink":
            assert sum_received == 0
        else:
            uplink_total += 1
            if not (sum_received is not None and sum_received > 0):
                uplink_zero_received += 1

    assert uplink_zero_received / uplink_total < 0.10
