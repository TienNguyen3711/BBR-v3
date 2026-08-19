from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
from qbbr.agents.quantum.qa2c import QA2CAgent
from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import minrtt_100_decision_interval_s

DEFAULT_CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
N_QUBITS = 6
N_WARMUP = 20
N_TRIALS = 500


def build_agent(core: str, n_layers: int = 2, reupload: bool = False):
    if core == "quantum":
        return QA2CAgent(n_layers=n_layers, reupload=reupload)
    return MLPA2CAgent(n_layers=n_layers)


def measure_act_latency_ms(agent, n_warmup: int = N_WARMUP, n_trials: int = N_TRIALS) -> dict[str, float]:
    rng = np.random.default_rng(0)
    states = rng.uniform(0.0, 1.0, size=(n_warmup + n_trials, N_QUBITS)).astype(np.float32)

    for i in range(n_warmup):
        agent.act(states[i])

    latencies_ms = []
    for i in range(n_warmup, n_warmup + n_trials):
        t0 = time.perf_counter()
        agent.act(states[i])
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms)
    return {
        "n_trials": n_trials,
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
        "min_ms": float(arr.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "bench_inference.json")
    args = parser.parse_args()

    # Matches the realistic online-deployment scenario: an agent embedded
    # in/near a kernel hook is not free to spawn worker threads competing
    # with the rest of the system for CPU -- a multi-threaded benchmark
    # would understate the latency this guardrail is actually meant to bound.
    torch.set_num_threads(1)

    calibration = load_calibration(args.calibration_path)
    t_dec_by_location: dict[str, float] = {}
    for location, directions in calibration.items():
        for direction, values in directions.items():
            t_dec_by_location[f"{location}/{direction}"] = (
                minrtt_100_decision_interval_s(values["RTT_min_ms"]) * 1000.0
            )
    tightest_label = min(t_dec_by_location, key=t_dec_by_location.get)
    tightest_t_dec_ms = t_dec_by_location[tightest_label]
    threshold_ms = tightest_t_dec_ms / 10.0

    results = {}
    for core in ("classical", "quantum"):
        print(f"benchmarking {core} agent.act() ({args.n_warmup} warmup + {args.n_trials} timed calls)...")
        agent = build_agent(core, n_layers=args.n_layers)
        stats = measure_act_latency_ms(agent, args.n_warmup, args.n_trials)
        results[core] = stats
        print(f"  median={stats['median_ms']:.4f}ms  p95={stats['p95_ms']:.4f}ms  "
              f"p99={stats['p99_ms']:.4f}ms  max={stats['max_ms']:.4f}ms")

    print(f"\nTightest guardrail across locations: {tightest_label}, "
          f"T_dec={tightest_t_dec_ms:.2f}ms, T_dec/10={threshold_ms:.3f}ms")
    for core, stats in results.items():
        for stat_name in ("median_ms", "p95_ms", "p99_ms", "max_ms"):
            verdict = "PASS" if stats[stat_name] < threshold_ms else "FAIL"
            print(f"  {core:10s} {stat_name:10s} {stats[stat_name]:9.4f}ms  vs {threshold_ms:.3f}ms  [{verdict}]")

    out_payload = {
        "t_dec_ms_by_location": t_dec_by_location,
        "tightest_guardrail": {
            "label": tightest_label, "t_dec_ms": tightest_t_dec_ms, "threshold_ms": threshold_ms,
        },
        "latency": results,
        "n_warmup": args.n_warmup, "n_qubits": N_QUBITS, "n_layers": args.n_layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2, sort_keys=True))
    try:
        shown_path = args.out.relative_to(PROJECT_ROOT)
    except ValueError:
        shown_path = args.out
    print(f"\nsaved -> {shown_path}")


if __name__ == "__main__":
    main()
