"""Per-path RTT tolerance for the pass criteria: run-to-run half-IQR of the reference controller's RTT p90.

For each location and direction, RTT p90 is taken per measured sequential run of the reference congestion
control (BBR on downlinks, BBRv2 on uplinks), using the same per-second RTT and interval filter as
trace_forcing. The tolerance is half the interquartile range of those per-run values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CITIES = ["Sydney", "Tokyo", "Mumbai", "Ohio", "London", "SaoPaulo"]
REFERENCE_CCA = {"downlink": "bbr", "uplink": "bbr2"}
DEFAULT_OUT = ROOT / "outputs" / "rq_study" / "final-v3" / "rtt_tolerance.json"


def load_tolerance(path: Path = DEFAULT_OUT) -> dict:
    """{location: {direction: half-IQR in ms}}; raises a clear error if the script has not been run."""
    if not path.exists():
        raise SystemExit(f"{path} not found; run `python -m qbbr.scripts.rtt_tolerance` first.")
    return {loc: {d: v["half_iqr_ms"] for d, v in dirs.items()} for loc, dirs in json.loads(path.read_text()).items()}


def main() -> None:
    from qbbr.data.catalog import build_catalog
    from qbbr.data.loader import load_trace
    from qbbr.scripts.train_on_traces import iter_file_records
    from qbbr.scripts.trace_replay import trace_forcing

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "qbbr" / "data" / "raw")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    catalog = build_catalog(args.dataset)
    out = {}
    for city in CITIES:
        out[city] = {}
        for direction, cca in REFERENCE_CCA.items():
            sub = catalog[(catalog.category == "sequential") & (catalog.location == city)
                          & (catalog.direction == direction) & (catalog.cca == cca)]
            runs = {int(r.run): trace_forcing(load_trace(r).intervals)["realised_rtt_p90_ms"] for r in iter_file_records(sub)}
            p90 = np.array([runs[k] for k in sorted(runs)])
            q1, q3 = np.percentile(p90, [25, 75])
            out[city][direction] = dict(reference_cca=cca, runs=sorted(runs), rtt_p90_ms=p90.round(3).tolist(),
                                        half_iqr_ms=round(float((q3 - q1) / 2), 2))
            print(f"{city:9s} {direction:9s} {cca:5s} n={len(p90):2d} half-IQR={out[city][direction]['half_iqr_ms']:6.2f} ms")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
