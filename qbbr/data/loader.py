"""Turn a parsed iperf3 JSON object into a tidy per-interval Trace.

Each file is single-stream (test_start.num_streams == 1); intervals[i]
["streams"][0] is always the transmitting endpoint's telemetry (the server
for downlink/REV, the client for uplink/FWD) -- confirmed empirically via
"sender": true on every interval across the corpus.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from qbbr.data.catalog import FileRecord
from qbbr.data.parse import ParseDiagnostics, parse_iperf3_file

_INTERVAL_COLUMNS = [
    "t_start",
    "t_end",
    "seconds",
    "bytes",
    "bits_per_second",
    "retransmits",
    "snd_cwnd",
    "snd_wnd",
    "rtt_ms",
    "rttvar_ms",
    "pmtu",
    "omitted",
]


@dataclass(frozen=True)
class Trace:
    meta: FileRecord
    intervals: pd.DataFrame
    summary: dict[str, Any]
    diagnostics: ParseDiagnostics


def _interval_row(interval: dict[str, Any]) -> dict[str, Any] | None:
    streams = interval.get("streams") or []
    if not streams:
        return None
    s = streams[0]
    return {
        "t_start": s.get("start"),
        "t_end": s.get("end"),
        "seconds": s.get("seconds"),
        "bytes": s.get("bytes"),
        "bits_per_second": s.get("bits_per_second"),
        "retransmits": s.get("retransmits"),
        "snd_cwnd": s.get("snd_cwnd"),
        "snd_wnd": s.get("snd_wnd"),
        "rtt_ms": (s["rtt"] / 1000.0) if s.get("rtt") is not None else None,
        "rttvar_ms": (s["rttvar"] / 1000.0) if s.get("rttvar") is not None else None,
        "pmtu": s.get("pmtu"),
        "omitted": s.get("omitted"),
    }


def load_trace(record: FileRecord) -> Trace:
    parsed = parse_iperf3_file(record.path)
    obj = parsed.raw

    rows = [row for i in obj.get("intervals", []) if (row := _interval_row(i)) is not None]
    intervals = pd.DataFrame(rows, columns=_INTERVAL_COLUMNS)

    summary = obj.get("end", {}) or {}
    return Trace(meta=record, intervals=intervals, summary=summary, diagnostics=parsed.diagnostics)
