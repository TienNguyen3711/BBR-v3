"""Tier-2 transfer: evaluate finished synthetic-trained checkpoints on replayed real capacity."""
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


def transfer_job(args: tuple[str, str, str]) -> str:
    import torch
    from qbbr.study.runner import build_agent, evaluate, trace_pools

    result_path, calibration_path, dataset = map(Path, args)
    out = result_path.with_name("transfer.json")
    if out.exists():
        return str(out)
    result = json.loads(result_path.read_text())
    study, job = result["identity"]["protocol"], result["job"]
    calibration = json.loads(calibration_path.read_text())
    saved = torch.load(result_path.with_name("checkpoint.pt"), map_location="cpu", weights_only=False)
    agent = build_agent(study, job["core"], job["seed"])
    agent.load_training_state_dict(saved["agent"])
    _, test_pool = trace_pools(study, job, dataset)
    rows = evaluate(study, job, calibration, agent, test_pool, stock_cache={})
    out.write_text(json.dumps(dict(job=job, test_runs=[e["run"] for e in test_pool], evaluation=rows)))
    return str(out)


def summarize(study_dir: Path) -> list[dict]:
    rows = []
    for city in CITIES:
        for core in ("qa2c", "a2c"):
            seeds = []
            for path in sorted(study_dir.glob(f"*__{city}__*__{core}__*/transfer.json")):
                evaluation = json.loads(path.read_text())["evaluation"]
                seeds.append(dict(
                    thr=np.mean([e["throughput_delta_pct"] for e in evaluation]),
                    rtt=np.mean([e["rtt_p90_delta_ms"] for e in evaluation]),
                    rtx=np.mean([e["retransmits_delta_per_s"] for e in evaluation]),
                    pairs_positive=np.mean([e["throughput_delta_pct"] > 0 for e in evaluation]),
                ))
            if not seeds:
                continue
            rows.append(dict(
                city=city, core=core, seeds=len(seeds),
                throughput_delta_pct_median=float(np.median([s["thr"] for s in seeds])),
                throughput_positive_seeds=int(sum(s["thr"] > 0 for s in seeds)),
                pairs_positive_fraction=float(np.mean([s["pairs_positive"] for s in seeds])),
                rtt_p90_delta_ms_median=float(np.median([s["rtt"] for s in seeds])),
                retransmits_delta_per_s_median=float(np.median([s["rtx"] for s in seeds])),
            ))
    (study_dir / "transfer_summary.json").write_text(json.dumps(rows, indent=2))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()

    paths = sorted(args.study.glob("*/result.json"))
    with Pool(args.workers) as pool:
        for done in pool.imap_unordered(transfer_job, [(str(p), str(args.calibration), str(args.dataset))
                                                        for p in paths]):
            print(done, flush=True)
    print(f"{'city':<9} {'core':<5} {'thr %':>7} {'+seeds':>6} {'+pairs':>6} {'dRTTp90':>8} {'dRtx/s':>7}")
    for r in summarize(args.study):
        print(f"{r['city']:<9} {r['core']:<5} {r['throughput_delta_pct_median']:>+7.2f} "
              f"{r['throughput_positive_seeds']:>4}/{r['seeds']} {r['pairs_positive_fraction']:>6.0%} "
              f"{r['rtt_p90_delta_ms_median']:>+8.2f} {r['retransmits_delta_per_s_median']:>+7.3f}")


if __name__ == "__main__":
    main()
