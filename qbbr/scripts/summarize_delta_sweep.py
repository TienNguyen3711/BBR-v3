"""Summarise the difference-reward RTT-weight (delta) sweep.

Reads outputs/rq_study/delta-sweep-a2c/delta_<d>/*/result.json and reports,
per delta and city, the paired policy-minus-stock deltas averaged over the
validation holdout seeds, then over training seeds. The RTT gate budget is the
one the v14 protocol derives (stock RTT p90 run half-IQR, from the v14
calibration), so the sweep is judged by the same rule as the screen.

Validation seeds only (2000-2004): delta is chosen here, and the manuscript
holdout (1000-1004) stays untouched until the chosen delta is re-run.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SWEEP = Path(os.environ.get("QBBR_DELTA_SWEEP", ROOT / "outputs" / "rq_study" / "delta-sweep-a2c"))
CALIBRATION = ROOT / "qbbr" / "data" / "calibrated" / "per_location_constants_v14.json"
CITIES = ["Sydney", "Tokyo", "Mumbai", "Ohio", "London", "SaoPaulo"]


def rtt_budgets() -> dict[str, float]:
    """The v14 gate: stock RTT p90 run half-IQR, resolved per location."""
    calibration = json.loads(CALIBRATION.read_text())
    return {city: float(calibration[city]["downlink"]["stock_rtt_p90_run_half_iqr_ms"])
            for city in CITIES if "stock_rtt_p90_run_half_iqr_ms" in calibration.get(city, {}).get("downlink", {})}


def main() -> None:
    budgets = rtt_budgets()
    rows = []
    for delta_dir in sorted(SWEEP.glob("delta_*")):
        delta = float(delta_dir.name.split("_", 1)[1])
        cells = defaultdict(list)
        for path in delta_dir.glob("*/result.json"):
            result = json.loads(path.read_text())
            if (result["job"]["core"], result["job"]["variant"]) != ("a2c", "full"):
                continue
            evaluation = result["evaluation"]
            cells[result["job"]["location"]].append(dict(
                thr=np.mean([e["throughput_delta_pct"] for e in evaluation]),
                rtt=np.mean([e["rtt_p90_delta_ms"] for e in evaluation]),
                rtx=np.mean([e["retransmits_delta_per_s"] for e in evaluation]),
                high_gain=sum(int(e["policy"]["action_counts"].get(k, 0)) for e in evaluation for k in ("3", "4"))
                / max(sum(sum(e["policy"]["action_counts"].values()) for e in evaluation), 1),
            ))
        for city in CITIES:
            seeds = cells.get(city, [])
            if not seeds:
                continue
            thr = [s["thr"] for s in seeds]
            rtt = [s["rtt"] for s in seeds]
            budget = budgets.get(city)
            rows.append(dict(
                delta=delta, city=city, seeds=len(seeds),
                thr_pct=float(np.mean(thr)), thr_positive=int(sum(t > 0 for t in thr)),
                rtt_ms=float(np.mean(rtt)), rtt_budget_ms=budget,
                rtt_gate=None if budget is None else bool(np.median(rtt) <= budget),
                rtt_below_stock=int(sum(r <= 0 for r in rtt)),
                rtx_per_s=float(np.mean([s["rtx"] for s in seeds])),
                high_gain_share=float(np.mean([s["high_gain"] for s in seeds])),
            ))
    (SWEEP / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"{'delta':>5} {'city':<9} {'n':>2} {'thr %':>7} {'+':>3} {'dRTT ms':>8} "
          f"{'budget':>7} {'gate':>5} {'<=stk':>5} {'dRtx/s':>7} {'hi-gain':>7}")
    for r in rows:
        budget = "-" if r["rtt_budget_ms"] is None else f"{r['rtt_budget_ms']:.2f}"
        gate = "-" if r["rtt_gate"] is None else ("PASS" if r["rtt_gate"] else "FAIL")
        print(f"{r['delta']:>5.3g} {r['city']:<9} {r['seeds']:>2} {r['thr_pct']:>+7.2f} {r['thr_positive']:>3} "
              f"{r['rtt_ms']:>+8.2f} {budget:>7} {gate:>5} {r['rtt_below_stock']:>5} "
              f"{r['rtx_per_s']:>+7.2f} {r['high_gain_share']:>7.1%}")


if __name__ == "__main__":
    main()
