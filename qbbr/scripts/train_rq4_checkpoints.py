"""Train the RQ4 quantum-vs-classical comparison at the full seeded protocol,
parallelised over (location x seed x core).

RQ4 asks whether a quantum policy core outperforms, matches, or trails a
classical network *under an equal trainable-parameter budget*, with state,
action space, reward, environment and seeds held identical. The classical
arm is built through ``build_full_parameter_matched_classical`` (the
``--rq4-full-match`` contract: every actor/critic parameter matched, not just
the variational block); the quantum arm is the ``QA2CAgent`` VQC core.

This mirrors ``train_rq3_checkpoints.py``: it exists so the RQ4
checkpoint-building step is a committed, parameterised entry point that
fans jobs out across a process pool, rather than ``run_ablation.py``'s
sequential per-(location,core) seed loop. Checkpoints land at
``<out-root>/<core>/<location>/seed<N>.pt``; a summary JSON with per-location
quantum-vs-classical medians, IQRs and a Mann-Whitney test is written to
``--out``.

    # the current seven-input L=2 grid point (matches archived tab:qvc scope)
    python -m qbbr.scripts.train_rq4_checkpoints \
        --out-root outputs/checkpoints_rq4_full --out outputs/rq4_full_report.json

    # Sydney at fewer seeds (its low RTT makes the quantum arm the wall-clock
    # bottleneck; the other five locations stay at the full 10)
    python -m qbbr.scripts.train_rq4_checkpoints --locations Sydney --seeds 0 1 2 \
        --out-root outputs/checkpoints_rq4_full --out outputs/rq4_full_sydney.json

The fast simulator backend (lightning.qubit + adjoint, see qbbr/agents/
quantum/qnn.py) is used automatically for the quantum arm; set
QBBR_QUANTUM_DEVICE=default.qubit to force the reference backend.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "base.yaml"
DEFAULT_CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
ALL_LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
CORES = ["quantum", "classical"]
RISK_MODES = ["stub_constant", "empirical_proxy", "closed_form", "closed_form_dynamic"]


def _train_one(job: dict) -> dict:
    import numpy as np
    import torch

    torch.set_num_threads(1)

    from qbbr.agents.matching import build_full_parameter_matched_classical
    from qbbr.agents.quantum.qa2c import QA2CAgent
    from qbbr.env.calibration import load_calibration
    from qbbr.env.fluid_env import FluidSimEnv
    from qbbr.train.loop import train

    config = dict(job["base_config"])
    config["seed"] = job["seed"]
    torch.manual_seed(job["seed"])
    np.random.seed(job["seed"])

    calibration = load_calibration(job["calibration_path"])
    reward_kwargs = dict(config.get("reward", {}))
    reward_kwargs["alpha"] = job["alpha"]
    env = FluidSimEnv(
        job["location"], job["direction"], calibration, risk_mode=job["risk_mode"],
        episode_s=config.get("episode", {}).get("duration_s", 300.0),
        reward_kwargs=reward_kwargs,
    )

    action_dims = (5,)  # pacing_gain-only, same as the archived RQ4 grid point
    lr = config.get("learning_rate", 1e-3)
    gamma = config.get("gamma", 0.99)
    if job["core"] == "quantum":
        agent = QA2CAgent(
            n_qubits=job["n_qubits"], n_layers=job["n_layers"], action_dims=action_dims,
            lr=lr, gamma=gamma, reupload=job["reupload"],
        )
        match_params = agent.param_count()
    else:
        agent, match = build_full_parameter_matched_classical(
            n_qubits=job["n_qubits"], n_layers=job["n_layers"], action_dims=action_dims,
            reupload=job["reupload"], lr=lr, gamma=gamma,
        )
        match_params = match.classical_params

    t0 = time.time()
    result = train(agent, env, job["n_episodes"], config, run_dir=None)
    elapsed = time.time() - t0

    out_dir = Path(job["out_root"]) / job["core"] / job["location"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed{job['seed']}.pt"
    agent.save(str(out_path))

    return {
        "location": job["location"], "seed": job["seed"], "core": job["core"],
        "elapsed_s": elapsed, "final_mean_reward": result["final_mean_reward"],
        "n_episodes": job["n_episodes"], "param_count": match_params,
        "checkpoint": str(out_path),
    }


def _summarise(records: list[dict], locations: list[str]) -> list[dict]:
    from qbbr.eval.stats import mann_whitney_test, summarize_median_iqr

    rows = []
    for loc in locations:
        by_core = {
            core: sorted(
                (r for r in records if r["location"] == loc and r["core"] == core),
                key=lambda r: r["seed"],
            )
            for core in CORES
        }
        q = [r["final_mean_reward"] for r in by_core["quantum"]]
        c = [r["final_mean_reward"] for r in by_core["classical"]]
        row = {
            "location": loc,
            "quantum": summarize_median_iqr(q) if q else None,
            "classical": summarize_median_iqr(c) if c else None,
            "quantum_param_count": by_core["quantum"][0]["param_count"] if by_core["quantum"] else None,
            "classical_param_count": by_core["classical"][0]["param_count"] if by_core["classical"] else None,
        }
        if len(q) >= 2 and len(c) >= 2:
            row["mann_whitney"] = mann_whitney_test(q, c)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-root", type=Path, required=True,
                        help="checkpoints -> <out-root>/<core>/<location>/seed<N>.pt")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "rq4_full_report.json",
                        help="summary JSON (per-location quantum-vs-classical medians + Mann-Whitney)")
    parser.add_argument("--locations", nargs="+", default=ALL_LOCATIONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--cores", nargs="+", default=CORES, choices=CORES)
    parser.add_argument("--direction", default="downlink", choices=["downlink", "uplink"])
    parser.add_argument("--n-qubits", type=int, default=7)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--reupload", action="store_true", help="data re-uploading in the VQC (quantum arm)")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--risk-mode", default="closed_form", choices=RISK_MODES,
                        help="archived RQ4 grid point is risk-on (closed_form)")
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="default: config's episode.target_episodes[0] (200)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()

    base_config = yaml.safe_load(args.config.read_text())
    n_episodes = args.n_episodes or base_config.get("episode", {}).get("target_episodes", [200])[0]

    jobs = [
        {
            "location": loc, "seed": seed, "core": core, "base_config": base_config,
            "risk_mode": args.risk_mode, "direction": args.direction,
            "out_root": str(args.out_root), "n_episodes": n_episodes,
            "calibration_path": args.calibration_path, "n_qubits": args.n_qubits,
            "n_layers": args.n_layers, "reupload": args.reupload, "alpha": args.alpha,
        }
        for loc in args.locations for core in args.cores for seed in args.seeds
    ]
    # Longest jobs first so the slow tail (low-RTT locations => more decision
    # steps/episode; quantum >> classical) does not strand the pool at the end.
    _loc_cost = {"Sydney": 6, "Tokyo": 3, "Mumbai": 3, "Ohio": 2, "London": 2, "SaoPaulo": 1}
    jobs.sort(key=lambda j: (_loc_cost.get(j["location"], 2), j["core"] == "quantum"), reverse=True)
    print(f"{len(jobs)} training job(s): {len(args.locations)} location(s) x {len(args.cores)} core(s) "
          f"x {len(args.seeds)} seed(s), n_qubits={args.n_qubits}, L={args.n_layers}, "
          f"reupload={args.reupload}, alpha={args.alpha}, risk_mode={args.risk_mode}, "
          f"{n_episodes} episodes each -> {args.out_root}/<core>/<location>/seed<N>.pt")

    t0 = time.time()
    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_train_one, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            r = future.result()
            records.append(r)
            elapsed = time.time() - t0
            try:
                shown = Path(r["checkpoint"]).relative_to(PROJECT_ROOT)
            except ValueError:
                shown = r["checkpoint"]
            print(f"[{i}/{len(jobs)}] {elapsed:8.1f}s  {r['core']:9s} {r['location']:10s} "
                  f"seed={r['seed']}  run_time={r['elapsed_s']:8.1f}s  "
                  f"reward={r['final_mean_reward']:.4f} ({r['param_count']}p)  -> {shown}")

    summary = _summarise(records, args.locations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "meta": {
            "locations": args.locations, "seeds": args.seeds, "cores": args.cores,
            "direction": args.direction, "n_qubits": args.n_qubits, "n_layers": args.n_layers,
            "reupload": args.reupload, "alpha": args.alpha, "risk_mode": args.risk_mode,
            "n_episodes": n_episodes, "total_elapsed_s": time.time() - t0,
        },
        "per_location": summary,
        "records": sorted(records, key=lambda r: (r["location"], r["core"], r["seed"])),
    }, indent=2, sort_keys=True))

    print(f"\ndone in {time.time() - t0:.1f}s -> {args.out}")
    for row in summary:
        q, c, mw = row.get("quantum"), row.get("classical"), row.get("mann_whitney")
        if q and c:
            p = f"{mw['p_value']:.3f}" if mw else "n/a"
            print(f"  {row['location']:10s}  Q {q['median']:.4f} (IQR {q['iqr']:.4f})  "
                  f"C {c['median']:.4f} (IQR {c['iqr']:.4f})  p={p}")


if __name__ == "__main__":
    main()
