"""Tier-2 deployment gate: validate synthetic-trained checkpoints on trace runs 1-7, fall back to stock if they fail.

Rule (fixed before the test results were examined): an agent is deployed if, over validation runs 1-7 x seeds
2000-2004, its median throughput delta is > 0, its median RTT p90 delta is <= the path's stock half-IQR, and its
median retransmission delta is <= 0.25/s. Otherwise it falls back to stock and its Tier-2 test deltas are zero.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION = ROOT / "qbbr" / "data" / "calibrated" / "per_location_constants_v14.json"
DEFAULT_DATASET = ROOT / "qbbr" / "data" / "raw"
CITIES = ["Sydney", "Tokyo", "Mumbai", "Ohio", "London", "SaoPaulo"]
DIRECTIONS = ["downlink", "uplink"]
VALIDATION_SEEDS = [2000, 2001, 2002, 2003, 2004]
RTX_LIMIT = 0.25


def validation_job(args: tuple[str, str, str]) -> str:
    import torch
    from qbbr.study.runner import build_agent, evaluate, trace_pools

    result_path, calibration_path, dataset = map(Path, args)
    out = result_path.with_name("validation.json")
    if out.exists():
        return str(out)
    result = json.loads(result_path.read_text())
    study, job = result["identity"]["protocol"], result["job"]
    calibration = json.loads(calibration_path.read_text())
    saved = torch.load(result_path.with_name("checkpoint.pt"), map_location="cpu", weights_only=False)
    agent = build_agent(study, job["core"], job["seed"])
    agent.load_training_state_dict(saved["agent"])
    validation_pool, _ = trace_pools(study, job, dataset)
    rows = evaluate(dict(study, holdout_seeds=VALIDATION_SEEDS), job, calibration, agent, validation_pool, stock_cache={})
    out.write_text(json.dumps(dict(job=job, validation_runs=[e["run"] for e in validation_pool], evaluation=rows)))
    return str(out)


def bootstrap(values, n=10_000, seed=0) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(values, (n, len(values))), axis=1)
    return float(np.median(values)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def summarize(study_dir: Path, calibration: dict) -> dict:
    cells = []
    for path in sorted(study_dir.glob("*/validation.json")):
        job = json.loads(path.read_text())["job"]
        if job["variant"] != "full":
            continue
        val = json.loads(path.read_text())["evaluation"]
        test = json.loads(path.with_name("result.json").read_text())["transfer_evaluation"]
        tol = calibration[job["location"]][job["direction"]]["stock_rtt_p90_run_half_iqr_ms"]
        med = {k: float(np.median([e[k] for e in val])) for k in ("throughput_delta_pct", "rtt_p90_delta_ms", "retransmits_delta_per_s")}
        deploy = med["throughput_delta_pct"] > 0 and med["rtt_p90_delta_ms"] <= tol and med["retransmits_delta_per_s"] <= RTX_LIMIT
        raw = {k: float(np.mean([e[k] for e in test])) for k in med}
        cells.append(dict(job=job, validation=med, deployed=deploy, raw=raw,
                          gated={k: (v if deploy else 0.0) for k, v in raw.items()}))
    out = {}
    for direction in DIRECTIONS:
        for core in ("qa2c", "a2c"):
            sel = [c for c in cells if c["job"]["direction"] == direction and c["job"]["core"] == core]
            if not sel:
                continue
            per_city = {}
            for city in CITIES:
                cs = [c for c in sel if c["job"]["location"] == city]
                per_city[city] = dict(
                    deployed=sum(c["deployed"] for c in cs), seeds=len(cs),
                    raw_thr=float(np.median([c["raw"]["throughput_delta_pct"] for c in cs])),
                    gated_thr=float(np.median([c["gated"]["throughput_delta_pct"] for c in cs])),
                    gated_rtt=float(np.median([c["gated"]["rtt_p90_delta_ms"] for c in cs])))
            out[f"{direction}/{core}"] = dict(
                deployed=sum(c["deployed"] for c in sel), agents=len(sel),
                raw_thr=bootstrap([c["raw"]["throughput_delta_pct"] for c in sel]),
                gated_thr=bootstrap([c["gated"]["throughput_delta_pct"] for c in sel]),
                raw_rtt=bootstrap([c["raw"]["rtt_p90_delta_ms"] for c in sel]),
                gated_rtt=bootstrap([c["gated"]["rtt_p90_delta_ms"] for c in sel]),
                deployed_test_thr_positive=sum(c["raw"]["throughput_delta_pct"] > 0 for c in sel if c["deployed"]),
                per_city=per_city)
    (study_dir / "validation_gate_summary.json").write_text(json.dumps(dict(cells=cells, pooled=out), indent=1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "outputs" / "rq_study" / "final-v3" / "1a_full")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if not args.summarize_only:
        jobs = [(str(p), str(args.calibration), str(args.dataset)) for p in sorted(args.study.glob("*__full__*/result.json"))]
        jobs = jobs[: args.limit] if args.limit else jobs
        with Pool(args.workers) as pool:
            for i, done in enumerate(pool.imap_unordered(validation_job, jobs), 1):
                print(f"[{i}/{len(jobs)}] {done}", flush=True)
    for key, row in summarize(args.study, json.loads(args.calibration.read_text())).items():
        print(key, {k: v for k, v in row.items() if k != "per_city"})


if __name__ == "__main__":
    main()
