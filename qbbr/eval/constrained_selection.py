"""Select throughput using hard paired safety constraints, not reward weights."""
from __future__ import annotations

import math
from statistics import median

METRICS = ("throughput_delta_pct", "rtt_p90_delta_ms", "retransmits_delta_per_s")


def summarize_candidate(rows: list[dict], expected_pairs: set[tuple[int, int]]) -> dict:
    """Every trace/seed must satisfy both safety limits (zero increase).

    Tolerance is for floating point equality only, not a practical budget.
    Equal-weight seeds and runs; deterministic baselines have seed -1.
    """
    pairs = [(r["run"], r["seed"]) for r in rows]
    complete = len(pairs) == len(set(pairs)) and set(pairs) == expected_pairs
    finite = bool(rows) and all(
        isinstance(r.get(k), (int, float)) and math.isfinite(r[k])
        for r in rows for k in METRICS
    )
    if not complete or not finite:
        return {"eligible": False, "complete": complete, "finite": finite}
    medians = {k: median(median(r[k] for r in rows if r["seed"] == seed)
                         for seed in sorted({r["seed"] for r in rows})) for k in METRICS}
    worst_rtt = max(r["rtt_p90_delta_ms"] for r in rows)
    worst_rtx = max(r["retransmits_delta_per_s"] for r in rows)
    return {
        "complete": True, "finite": True, "medians": medians,
        "worst_rtt_p90_delta_ms": worst_rtt,
        "worst_retransmits_delta_per_s": worst_rtx,
        "eligible": worst_rtt <= 1e-9 and worst_rtx <= 1e-9,
        "positive_throughput_pair_fraction": sum(r[METRICS[0]] > 1e-9 for r in rows) / len(rows),
        "strict_three_metric_improvement": all(
            r[METRICS[0]] > 1e-9 and r[METRICS[1]] < -1e-9 and r[METRICS[2]] < -1e-9
            for r in rows),
    }


def select_candidate(summaries: dict[str, dict]) -> str:
    """Validation-only selection; preserve stock when no safe improvement exists."""
    candidates = [name for name, s in summaries.items() if s.get("eligible")
                  and s["medians"]["throughput_delta_pct"] > 1e-9]
    return max(sorted(candidates), key=lambda n: summaries[n]["medians"]["throughput_delta_pct"]) if candidates else "stock"
