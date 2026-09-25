"""Measured mixed-flow coexistence from the competitive iperf3 logs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from qbbr.data.catalog import build_catalog
from qbbr.data.parse import parse_iperf3_file
from qbbr.eval.metrics import alpha_fair_efficiency_ratio, min_per_flow_throughput

ROOT = Path(__file__).resolve().parents[2]
ALPHAS = (0.0, 1.0, float("inf"))
MIN_WINDOW_S = 120.0


def load_flow(path: str) -> dict[str, Any] | None:
    """Absolute-time interval series for one run file, or None if unusable."""
    raw = parse_iperf3_file(path).raw
    intervals = raw.get("intervals") or []
    base = raw.get("start", {}).get("timestamp", {}).get("timesecs")
    if base is None or not intervals:
        return None
    rows = []
    for entry in intervals:
        total = entry.get("sum") or {}
        if total.get("omitted"):
            continue
        rows.append((base + float(total["start"]), base + float(total["end"]),
                     float(total["bytes"]), float(total.get("retransmits") or 0.0)))
    if not rows:
        return None
    return {"rows": rows, "t0": rows[0][0], "t1": rows[-1][1]}


def bytes_in_window(flow: dict[str, Any], t0: float, t1: float) -> tuple[float, float]:
    """Bytes and retransmits inside [t0, t1], splitting partially covered intervals."""
    total_bytes = total_retransmits = 0.0
    for start, end, size, retransmits in flow["rows"]:
        overlap = min(end, t1) - max(start, t0)
        if overlap <= 0:
            continue
        fraction = overlap / (end - start) if end > start else 0.0
        total_bytes += size * fraction
        total_retransmits += retransmits * fraction
    return total_bytes, total_retransmits


def competitive_run(paths: dict[str, str]) -> dict[str, Any] | None:
    flows = {cca: load_flow(path) for cca, path in paths.items()}
    flows = {cca: flow for cca, flow in flows.items() if flow is not None}
    if len(flows) < 2:
        return None
    t0 = max(flow["t0"] for flow in flows.values())
    t1 = min(flow["t1"] for flow in flows.values())
    window = t1 - t0
    if window < MIN_WINDOW_S:
        return {"excluded": True, "window_s": round(window, 1), "flows": sorted(flows)}
    throughput, retransmits = {}, {}
    for cca, flow in flows.items():
        size, retransmitted = bytes_in_window(flow, t0, t1)
        throughput[cca] = size * 8 / window
        retransmits[cca] = retransmitted / window
    ordered = [throughput[cca] for cca in sorted(throughput)]
    return {
        "excluded": False, "window_s": round(window, 1),
        "throughput_bps": throughput, "retransmits_per_s": retransmits,
        "rho_alpha": {("inf" if a == float("inf") else str(a)):
                      alpha_fair_efficiency_ratio(ordered, a) for a in ALPHAS},
        "min_flow_bps": min_per_flow_throughput(ordered),
        "total_bps": sum(ordered),
    }


def sequential_baseline(catalog, location: str, direction: str) -> dict[str, float]:
    """Median isolated throughput per CCA on the same path."""
    rows = catalog[(catalog.category == "sequential") & (catalog.location == location)
                   & (catalog.direction == direction)]
    baseline = defaultdict(list)
    for row in rows.itertuples():
        flow = load_flow(row.path)
        if flow is None:
            continue
        duration = flow["t1"] - flow["t0"]
        if duration <= 0:
            continue
        size, _ = bytes_in_window(flow, flow["t0"], flow["t1"])
        baseline[row.cca].append(size * 8 / duration)
    return {cca: median(values) for cca, values in baseline.items() if values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=ROOT / "qbbr" / "data" / "raw")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "measured_coexistence.json")
    parser.add_argument("--directions", nargs="+", default=["downlink", "uplink"])
    args = parser.parse_args()

    catalog = build_catalog(args.dataset)
    competitive = catalog[catalog.category == "competitive"]
    report: dict[str, Any] = {
        "method": "throughput measured inside the window where every flow of a run is active",
        "min_window_s": MIN_WINDOW_S, "alphas": ["0.0", "1.0", "inf"],
        "alpha_note": "alpha>1 omitted: the efficiency ratio is unbounded there",
        "cells": {},
    }
    for direction in args.directions:
        for location in sorted(competitive.location.unique()):
            rows = competitive[(competitive.location == location)
                               & (competitive.direction == direction)]
            if rows.empty:
                continue
            baseline = sequential_baseline(catalog, location, direction)
            per_run, excluded = [], []
            for run in sorted(rows.run.unique()):
                paths = {r.cca: r.path for r in rows[rows.run == run].itertuples()}
                result = competitive_run(paths)
                if result is None:
                    continue
                (excluded if result["excluded"] else per_run).append({"run": int(run), **result})
            if not per_run:
                report["cells"][f"{location}/{direction}"] = {
                    "usable_runs": 0, "excluded_runs": [e["run"] for e in excluded]}
                continue
            ccas = sorted(per_run[0]["throughput_bps"])
            share = {}
            for cca in ccas:
                contended = median(r["throughput_bps"][cca] for r in per_run if cca in r["throughput_bps"])
                isolated = baseline.get(cca)
                share[cca] = {
                    "contended_mbps": round(contended / 1e6, 3),
                    "isolated_mbps": round(isolated / 1e6, 3) if isolated else None,
                    "retained_pct": round(100 * contended / isolated, 1) if isolated else None,
                    "share_of_total_pct": round(
                        100 * contended / median(r["total_bps"] for r in per_run), 1),
                }
            report["cells"][f"{location}/{direction}"] = {
                "usable_runs": len(per_run),
                "excluded_runs": [e["run"] for e in excluded],
                "median_window_s": round(median(r["window_s"] for r in per_run), 1),
                "rho_alpha": {a: round(median(r["rho_alpha"][a] for r in per_run), 4)
                              for a in report["alphas"]},
                "min_flow_mbps": round(median(r["min_flow_bps"] for r in per_run) / 1e6, 3),
                "per_cca": share,
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["cells"], indent=2, sort_keys=True)[:2000])
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
