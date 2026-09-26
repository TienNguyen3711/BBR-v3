"""Tier 3: simulated stock minus real Linux BBR on the same tc-replayed downlink capacity (runs 1-5 per city)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CITIES = ["Sydney", "Tokyo", "Mumbai", "Ohio", "London", "SaoPaulo"]
SEEDS = [1000, 1001, 1002, 1003, 1004]


def main() -> None:
    from qbbr.data.catalog import build_catalog
    from qbbr.scripts.train_on_traces import build_pool
    from qbbr.study.runner import stock_rollout

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testbed", type=Path, default=ROOT / "reports" / "kernel_testbed_6city_clean.json")
    parser.add_argument("--study", type=Path, default=ROOT / "outputs" / "rq_study" / "final-v3" / "1a_full")
    parser.add_argument("--calibration", type=Path, default=ROOT / "qbbr" / "data" / "calibrated" / "per_location_constants_v14.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "qbbr" / "data" / "raw")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "rq_study" / "final-v3" / "tier3_model_error.json")
    args = parser.parse_args()

    kernel_rows = json.loads(args.testbed.read_text())["rows"]
    calibration = json.loads(args.calibration.read_text())
    catalog = build_catalog(args.dataset)
    out = {}
    for city in CITIES:
        result = json.loads(next(args.study.glob(f"*__{city}__downlink__qa2c__full__0/result.json")).read_text())
        study, job = result["identity"]["protocol"], result["job"]
        kernel = [r for r in kernel_rows if r["location"] == city and r["direction"] == "downlink"]
        runs = sorted(r["run"] for r in kernel)
        pool = build_pool(catalog, city, "downlink", study["trace_split"]["reference_cca"]["downlink"], runs,
                          study["trace_split"]["capacity_proxy"])
        sim = [stock_rollout(study, job, calibration, entry, seed) for entry in pool for seed in SEEDS]
        k_thr = np.mean([r["kernel"]["throughput_mbps_mean"] for r in kernel])
        k_rtt = np.mean([r["kernel"]["rtt_p90_ms"] for r in kernel])
        s_thr = np.mean([r["throughput_mbps"] for r in sim])
        s_rtt = np.mean([r["rtt_p90_ms"] for r in sim])
        out[city] = dict(runs=runs, kernel_thr=k_thr, kernel_rtt_p90=k_rtt, sim_thr=s_thr, sim_rtt_p90=s_rtt,
                         thr_err_pct=100 * (s_thr / k_thr - 1), rtt_err_ms=s_rtt - k_rtt,
                         trace_rtt_p90=float(np.mean([r["trace_realised_rtt_p90_ms"] for r in kernel])))
        print(city, {k: round(v, 2) if isinstance(v, float) else v for k, v in out[city].items()}, flush=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
