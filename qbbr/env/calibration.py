"""Per-location, per-direction calibration constants (Table 1 of the design doc).

Computed from stock BBR-v3 ("bbr") traces only, per location and direction,
using 1st/99th percentiles rather than the paper's illustrative figures --
exactly as Table 1 in quantum_bbr_pipeline_v2.tex specifies ("re-estimated
from the training dataset (per-location 99th percentiles) rather than
hard-coded").
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qbbr.data.catalog import build_catalog, iter_file_records
from qbbr.data.loader import load_trace

_LOW_PCT = 1.0
_HIGH_PCT = 99.0


def compute_calibration(
    dataset_root: str | Path, cca: str = "bbr"
) -> dict[str, dict[str, dict[str, float]]]:
    """Return {location: {direction: {B_max_mbps, RTT_min_ms, RTT_max_ms}}}."""
    catalog = build_catalog(dataset_root)
    subset = catalog[catalog["cca"] == cca]
    if subset.empty:
        raise ValueError(f"no traces found for cca={cca!r} under {dataset_root}")

    result: dict[str, dict[str, dict[str, float]]] = {}
    for (location, direction), group in subset.groupby(["location", "direction"]):
        bps_parts, rtt_parts = [], []
        for record in iter_file_records(group):
            trace = load_trace(record)
            bps_parts.append(trace.intervals["bits_per_second"].dropna())
            rtt_parts.append(trace.intervals["rtt_ms"].dropna())
        bps = pd.concat(bps_parts)
        rtt = pd.concat(rtt_parts)

        result.setdefault(location, {})[direction] = {
            "B_max_mbps": float(np.percentile(bps, _HIGH_PCT) / 1e6),
            "RTT_min_ms": float(np.percentile(rtt, _LOW_PCT)),
            "RTT_max_ms": float(np.percentile(rtt, _HIGH_PCT)),
        }
    return result


def save_calibration(calibration: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(calibration, indent=2, sort_keys=True))


def load_calibration(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
