"""Exploratory Sydney baseline/action audit; no retraining or kernel writes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
import torch
import yaml

from qbbr.data.catalog import build_catalog, iter_file_records
from qbbr.data.loader import load_trace
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.eval.constrained_selection import summarize_candidate, select_candidate
from qbbr.scripts.run_native_qa2c_successor import _selector
from qbbr.scripts.trace_replay import trace_forcing, align_phase_offset, _load_agent

ROOT = Path(__file__).resolve().parents[2]
SIMPLE = ("stock", "highest_permitted", "gain110", "balanced110", "balanced125")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def simple_action(name, admitted):
    if name == "stock":
        return 2
    if name == "highest_permitted":
        return max(admitted)
    target = 3 if name in ("gain110", "balanced110") else 4
    if target in admitted:
        return target
    if name.startswith("balanced") and 1 in admitted:
        return 1
    return 2


def _episode(task):
    torch.set_num_threads(1)
    cfg, calibration, entry = task["config"], task["calibration"], task["entry"]
    overrides = dict(cfg["simulator"]["dynamics_overrides"])
    overrides["native_cruise_override"] = task["semantics"] == "native_cruise_proxy"
    env = FluidSimEnv("Sydney", "downlink", calibration,
        episode_s=entry["duration_s"], substep_s=task["substep_s"],
        reward_mode="throughput_only", risk_mode=cfg["simulator"]["risk_mode"],
        dynamics_overrides=overrides, probe_bw_phase_gate=True,
        capacity_trace=(entry["times_s"], entry["capacity_bytes_s"]),
        phase_offset_s=entry["phase_offset_s"])
    selector = _selector(cfg)
    model = None
    if task["policy"] in ("quantum", "classical"):
        model = _load_agent(cfg, env, Path(task["checkpoint_root"]), "Sydney", "downlink",
                            task["seed"], task["policy"])
    state, done = env.reset(seed=0), False
    total_bytes = total_retx = elapsed = 0.0
    rtts, counts, selected = [], np.zeros(5, dtype=int), set()
    phase_s = {str(i): 0.0 for i in range(4)}
    while not done:
        allowed = env.allowed_action_indices()
        admitted = selector.admissible_actions(torch.zeros(5), torch.as_tensor(state), allowed)
        if model is None:
            action = simple_action(task["policy"], admitted)
        else:
            action = model.act(state, allowed, deterministic=True, deployment=True)[0]
        if action not in admitted:
            raise RuntimeError("Policy escaped the shared hard mask")
        state, _, done, info = env.step(action)
        counts[action] += 1
        selected.add(round(info["selected_pacing_gain"], 9))
        elapsed += info["t_dec_s"]
        total_bytes += info["delivered_bytes"]
        total_retx += info["retransmits"]
        rtts.append(info["rtt_ms"])
        for phase, seconds in info["probe_phase_durations_s"].items():
            phase_s[phase] += seconds
    return {
        "semantics": task["semantics"], "policy": task["policy"], "seed": task["seed"],
        "run": entry["run"], "split": task["split"],
        "throughput_mbps": total_bytes * 8 / elapsed / 1e6,
        "rtt_p90_ms": float(np.percentile(rtts, 90)),
        "retransmits_per_s": total_retx / elapsed,
        "duration_s": elapsed, "decisions": int(counts.sum()),
        "action_shares": dict(zip(map(str, range(5)), (counts / counts.sum()).tolist())),
        "phase_shares": {k: v / elapsed for k, v in phase_s.items()},
        "selected_gain_count": len(selected), "selected_gains": sorted(selected),
    }


def paired_rows(rows):
    stock = {(r["semantics"], r["run"]): r for r in rows if r["policy"] == "stock"}
    out = []
    for row in rows:
        baseline = stock[(row["semantics"], row["run"])]
        out.append({**row,
            "throughput_delta_pct": 100 * (row["throughput_mbps"] / baseline["throughput_mbps"] - 1),
            "rtt_p90_delta_ms": row["rtt_p90_ms"] - baseline["rtt_p90_ms"],
            "retransmits_delta_per_s": row["retransmits_per_s"] - baseline["retransmits_per_s"],
        })
    return out


def assess(rows, runs, seeds):
    summaries = {}
    for semantics in ("legacy_v14", "native_cruise_proxy"):
        summaries[semantics] = {}
        for policy in SIMPLE + ("quantum", "classical"):
            expected = {(run, seed) for run in runs
                        for seed in (seeds if policy in ("quantum", "classical") else [-1])}
            subset = [r for r in rows if r["semantics"] == semantics and r["policy"] == policy]
            summaries[semantics][policy] = summarize_candidate(subset, expected)
    return summaries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/sydney_semantics_audit")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--substep-s", type=float, default=0.02)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if args.substep_s <= 0 or args.workers < 1:
        ap.error("substep and workers must be positive")
    config_path = ROOT / "qbbr/configs/tier1_v14_fidelity.yaml"
    calib_path = ROOT / "qbbr/data/calibrated/per_location_constants_v14.json"
    checkpoint_root = ROOT / "outputs/checkpoints_v14"
    cfg = yaml.safe_load(config_path.read_text())
    calibration = json.loads(calib_path.read_text())
    seeds, validation_runs, evaluation_runs = list(range(5)), [6, 7], [8, 9, 10]
    catalog = build_catalog(ROOT / "qbbr/data/raw")
    records = list(iter_file_records(catalog[(catalog.location == "Sydney") &
        (catalog.direction == "downlink") & (catalog.category == "sequential") &
        (catalog.cca == "bbr") & catalog.run.isin(validation_runs + evaluation_runs)]))
    if sorted(r.run for r in records) != sorted(validation_runs + evaluation_runs):
        raise ValueError("Require exactly one Sydney BBR trace per declared run")
    entries, provenance = [], {}
    for record in records:
        forcing = trace_forcing(load_trace(record).intervals, "envq90_w5")
        entries.append({**forcing, "run": record.run,
            "phase_offset_s": align_phase_offset(forcing["times_s"], forcing["capacity_bytes_s"])})
        provenance[str(record.path.relative_to(ROOT))] = _hash(record.path)
    for core in ("quantum", "classical"):
        for seed in seeds:
            path = checkpoint_root / core / "Sydney/downlink" / f"seed{seed}.pt"
            provenance[str(path.relative_to(ROOT))] = _hash(path)
    plan = {
        "protocol_id": "sydney-baseline-semantics-audit-v1", "evidence_tier": "exploratory_simulator_proxy",
        "validation_runs": validation_runs, "evaluation_runs": evaluation_runs, "seeds": seeds,
        "policies": list(SIMPLE) + ["quantum", "classical"], "config": cfg,
        "calibration": calibration["Sydney"]["downlink"],
        "config_sha256": _hash(config_path), "calibration_sha256": _hash(calib_path),
        "files_sha256": provenance, "substep_s": args.substep_s,
        "selection": "maximize median throughput delta; every validation run/seed RTT p90 and retransmissions <= stock; stock fallback",
        "limitations": [
            "Reused traces and full-dataset calibration: not a fresh confirmatory test.",
            "Replay capacity is inferred from achieved throughput, not independently measured capacity.",
            "Full-trace phase alignment is offline information, not a deployable predictor.",
            "Native mode is a fixed-duration fluid phase proxy, not validated BBR-v3.",
            "Old v14 synthetic-trained checkpoints are transferred without retraining.",
            "Trace-retrained learners were not checkpointed by their runner; their reported scores are not reusable weights.",
            "No bootstrap claim from two validation and three reused evaluation traces.",
            "Historical >1.5% threshold is not a calibrated error bound for the changed model.",
        ],
    }
    if not args.execute:
        print(json.dumps({k: v for k, v in plan.items() if k not in ("config", "files_sha256")}, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2))
    all_rows, summaries, selected = [], {}, {}
    for split, runs in (("validation", validation_runs), ("evaluation", evaluation_runs)):
        tasks = []
        for semantics in ("legacy_v14", "native_cruise_proxy"):
            for entry in entries:
                if entry["run"] not in runs:
                    continue
                for policy in SIMPLE + ("quantum", "classical"):
                    for seed in (seeds if policy in ("quantum", "classical") else [-1]):
                        tasks.append(dict(config=cfg, calibration=calibration, entry=entry,
                            semantics=semantics, policy=policy, seed=seed, split=split,
                            checkpoint_root=str(checkpoint_root), substep_s=args.substep_s))
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_episode, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                with (args.out / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(f"{split} {len(rows)}/{len(tasks)} {row['semantics']} {row['policy']} seed={row['seed']} run={row['run']}", flush=True)
        rows = paired_rows(rows)
        all_rows.extend(rows)
        summaries[split] = assess(rows, runs, seeds)
        if split == "validation":
            selected = {sem: select_candidate(s) for sem, s in summaries[split].items()}
            # Persist the choice BEFORE running evaluation.
            (args.out / "selection.json").write_text(json.dumps({"selected": selected,
                "validation": summaries[split]}, indent=2))
            print("Frozen validation selection: " + json.dumps(selected), flush=True)
    result = {"plan": plan, "selected_on_validation": selected, "summaries": summaries, "rows": all_rows}
    (args.out / "report.json").write_text(json.dumps(result, indent=2))
    lines = ["# Sydney baseline/action-semantics audit", "", "Exploratory fluid-model results; no kernel or field claim.", "",
        "Selection was frozen using runs 6/7 before evaluating reused runs 8/9/10.", "",
        "| Semantics | Policy | Throughput delta (%) | RTT p90 delta (ms) | Retransmission delta (/s) | Every evaluation pair safe? |",
        "|---|---|---:|---:|---:|---|"]
    for sem, policies in summaries["evaluation"].items():
        for policy, summary in policies.items():
            m = summary["medians"]
            lines.append(f"| {sem} | {policy} | {m['throughput_delta_pct']:+.4f} | {m['rtt_p90_delta_ms']:+.4f} | {m['retransmits_delta_per_s']:+.6f} | {summary['eligible']} |")
    lines += ["", "Validation-selected policies: " + json.dumps(selected), "", "All medians give equal weight to seeds and then traces. Safety checks use every pair, not medians.", ""]
    lines += ["- " + caveat for caveat in plan["limitations"]]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("Report: " + str(args.out / "summary.md"), flush=True)


if __name__ == "__main__":
    main()
