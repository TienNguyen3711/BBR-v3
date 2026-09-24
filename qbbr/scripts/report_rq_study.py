"""Aggregate an executed RQ study matrix into the RQ1/RQ2/RQ3 contrasts.

Read-only. It reports what is on disk, states how many jobs are missing, and
never fills a gap with an assumption.

Design notes that affect every number below:

* **Pairing.** Arms are compared at a matched (location, core, training seed,
  holdout seed). A holdout seed fixes the capacity forcing, which is the
  dominant variance term, so a paired difference removes it. Comparing arm
  medians instead would drown the effect in forcing noise.
* **Replicate unit.** Holdout seeds are repeated measurements of one trained
  policy, not independent replicates. They are aggregated to a median within
  each (location, training seed) first; that pair is the unit that enters the
  bootstrap and the sign count.
* **n is small.** Three training seeds per cell is enough to resolve a 2/3 from
  a 3/3 pattern, not to place a tight interval. With 18 (location, seed) units
  the bootstrap interval is reported, but the sign count is the more honest
  summary at this sample size and is printed beside it.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from qbbr.eval.successor_protocol import bootstrap_median_ci

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("throughput_delta_pct", "rtt_p90_delta_ms", "retransmits_delta_per_s")


def load_rows(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, jobs, expected = [], [], None
    for manifest in sorted(out.glob("manifest*.json")):
        declared = json.loads(manifest.read_text())
        expected = expected or declared["identity"]["protocol"]
        jobs.extend(declared["jobs"])
    for path in sorted(out.glob("*/result.json")):
        result = json.loads(path.read_text())
        job = result["job"]
        for entry in result["evaluation"]:
            rows.append({**{k: job[k] for k in ("location", "direction", "core", "variant", "seed")},
                         "holdout_seed": entry["holdout_seed"],
                         "action_counts": entry["policy"]["action_counts"],
                         **{m: entry[m] for m in METRICS}})
    return rows, {"declared_jobs": len(jobs), "completed_jobs": len(list(out.glob("*/result.json"))),
                  "protocol": expected}


def by_cell(rows, **filters):
    """Median across holdout seeds, keyed by (location, training seed)."""
    grouped = defaultdict(list)
    for row in rows:
        if all(row[key] == value for key, value in filters.items()):
            grouped[(row["location"], row["seed"])].append(row)
    return {key: {m: median(r[m] for r in group) for m in METRICS}
            for key, group in grouped.items()}


def paired(rows, metric, treatment: dict, control: dict):
    """Per-(location, seed) difference between two arms at matched holdout seeds."""
    left = defaultdict(dict)
    right = defaultdict(dict)
    for row in rows:
        target = (left if all(row[k] == v for k, v in treatment.items()) else
                  right if all(row[k] == v for k, v in control.items()) else None)
        if target is not None:
            target[(row["location"], row["seed"])][row["holdout_seed"]] = row[metric]
    differences = {}
    for key, values in left.items():
        shared = values.keys() & right.get(key, {}).keys()
        if shared:
            differences[key] = median(values[s] - right[key][s] for s in shared)
    return differences


def describe(differences: dict, label: str) -> dict[str, Any]:
    values = list(differences.values())
    if not values:
        return {"contrast": label, "n": 0, "note": "no matched pairs on disk"}
    positive = sum(1 for v in values if v > 0)
    summary = {"contrast": label, "n": len(values), "median": round(median(values), 4),
               "positive": f"{positive}/{len(values)}",
               "min": round(min(values), 4), "max": round(max(values), 4)}
    if len(values) >= 2:
        ci = bootstrap_median_ci(values)
        summary["ci95"] = [round(ci["lower"], 4), round(ci["upper"], 4)]
        summary["excludes_zero"] = ci["lower"] > 0 or ci["upper"] < 0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs" / "rq_study" / "native-bbr-rq-screen-v2-sixcity")
    parser.add_argument("--json", type=Path, help="also write the full report as JSON")
    args = parser.parse_args()

    rows, status = load_rows(args.out)
    if not rows:
        raise SystemExit(f"No results under {args.out}")
    report: dict[str, Any] = {"status": {k: v for k, v in status.items() if k != "protocol"}}

    # RQ1 -- the deployed arm against stock BBR-v3, per location.
    rq1 = {}
    for location in sorted({row["location"] for row in rows}):
        cells = by_cell(rows, core="qa2c", variant="full", location=location)
        if cells:
            deltas = [v["throughput_delta_pct"] for v in cells.values()]
            rq1[location] = {
                "seeds": len(deltas),
                "throughput_median_pct": round(median(deltas), 3),
                "positive_seeds": f"{sum(1 for d in deltas if d > 0)}/{len(deltas)}",
                "rtt_p90_median_ms": round(median(v["rtt_p90_delta_ms"] for v in cells.values()), 3),
                "retransmits_median_per_s": round(
                    median(v["retransmits_delta_per_s"] for v in cells.values()), 4),
            }
    report["rq1_qa2c_vs_stock"] = rq1

    # RQ2 -- does removing the anticipatory features cost anything?
    report["rq2_state_ablation"] = [
        describe(paired(rows, metric, {"core": core, "variant": "full"},
                        {"core": core, "variant": variant}),
                 f"{core}: full minus {variant} ({metric})")
        for metric in ("throughput_delta_pct", "rtt_p90_delta_ms")
        for core in ("qa2c", "a2c")
        for variant in ("telemetry", "telemetry_queue")
    ]

    # RQ3 -- quantum versus matched classical, and the recurrent estimator.
    report["rq3_representation"] = [
        describe(paired(rows, "throughput_delta_pct", {"core": "qa2c", "variant": variant},
                        {"core": "a2c", "variant": variant}),
                 f"qa2c minus a2c ({variant}, 126 params each)")
        for variant in ("full", "telemetry", "telemetry_queue")
    ] + [describe(paired(rows, "throughput_delta_pct", {"core": "qdqn", "variant": "full"},
                         {"core": core, "variant": "full"}), f"qdqn minus {core} (full)")
         for core in ("qa2c", "a2c")]

    # A policy pinned to the stock action reports exactly 0.000% and is not a
    # result; report the share explicitly so a frozen arm cannot read as a null.
    frozen = sum(1 for row in rows if row["action_counts"].get("2", 0)
                 == sum(row["action_counts"].values()))
    report["stock_pinned_evaluation_share"] = f"{frozen}/{len(rows)}"

    print(json.dumps(report, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
