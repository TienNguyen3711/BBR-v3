"""Execution entry point for the RQ1/RQ2/RQ3 native study matrix."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from qbbr.study.protocol import atomic_json, job_id, jobs, load_protocol, validate_protocol

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "qbbr" / "configs" / "rq_study_screen.yaml"
DEFAULT_CALIBRATION = ROOT / "qbbr" / "data" / "calibrated" / "per_location_constants_v14.json"

MEASURED_SPEEDUP = {1: 1.0, 7: 5.0, 12: 4.56}
RECOMMENDED_PARALLELISM = 7
TRAIN_MS_PER_DECISION = {"qa2c": 12.9, "a2c": 4.3, "qdqn": 21.2}
EVAL_MS_PER_DECISION = {"qa2c": 6.09, "a2c": 4.18, "qdqn": 5.35}
STOCK_MS_PER_DECISION = 4.09

_SCOPE = {
    "locations": "locations", "directions": "directions", "cores": "cores",
    "variants": "variants", "forcing": "forcing",
    "training_seeds": "training_seeds", "holdout_seeds": "holdout_seeds",
}


def apply_scope(study: dict[str, Any], args: argparse.Namespace) -> None:
    """Narrow a declared study, then re-validate it as if it were declared."""
    for flag, key in _SCOPE.items():
        value = getattr(args, flag)
        if value is not None:
            study[key] = value
    for flag in ("episodes", "curve_every"):
        if getattr(args, flag) is not None:
            study[flag] = int(getattr(args, flag))
    if args.duration_s is not None:
        study["duration_s"] = float(args.duration_s)
    if args.reward_delta is not None:
        # The RTT weight of the difference reward. Written into the base
        # protocol itself, so it enters the digest and every run's identity.
        kwargs = dict(study["base"]["simulator"].get("reward_kwargs") or {})
        kwargs["delta"] = float(args.reward_delta)
        study["base"]["simulator"]["reward_kwargs"] = kwargs
    if args.reward_delay_form is not None:
        kwargs = dict(study["base"]["simulator"].get("reward_kwargs") or {})
        kwargs["delay_form"] = args.reward_delay_form
        if args.reward_delay_form == "queue":
            kwargs["queue_floor_ms"] = float(args.reward_queue_floor_ms)
        study["base"]["simulator"]["reward_kwargs"] = kwargs
    validate_protocol(study)


def speedup(shards: int) -> float:
    """Interpolate the measured scaling curve; hold the last point beyond it."""
    points = sorted(MEASURED_SPEEDUP)
    if shards <= points[0]:
        return MEASURED_SPEEDUP[points[0]]
    if shards >= points[-1]:
        return MEASURED_SPEEDUP[points[-1]]
    low = max(n for n in points if n <= shards)
    high = min(n for n in points if n >= shards)
    if low == high:
        return MEASURED_SPEEDUP[low]
    weight = (shards - low) / (high - low)
    return MEASURED_SPEEDUP[low] + weight * (MEASURED_SPEEDUP[high] - MEASURED_SPEEDUP[low])


def curve_points(study: dict[str, Any]) -> int:
    every, total = study["curve_every"], study["episodes"]
    return sum(1 for e in range(1, total + 1) if e % every == 0 or e == total)


def decisions_per_episode(study: dict[str, Any], calibration: dict[str, Any],
                          location: str, direction: str) -> float:
    from qbbr.env.fluid_env import minrtt_100_decision_interval_s

    try:
        min_rtt = calibration[location][direction]["RTT_min_ms"]
    except KeyError as exc:
        raise SystemExit(f"Calibration has no RTT_min_ms for {location}/{direction}") from exc
    return study["duration_s"] / minrtt_100_decision_interval_s(min_rtt)


def cost(study: dict[str, Any], selected: list[dict[str, Any]], calibration: dict[str, Any],
         with_dataset: bool, scale: float = 1.0) -> dict[str, Any]:
    """Wall-clock estimate for the selected jobs, by core and by location."""
    holdout = len(study["holdout_seeds"])
    test_runs = study["trace_split"]["test_runs"]
    train_s = policy_s = stock_decisions = 0.0
    stock_keys, by_location, by_core = {}, Counter(), Counter()
    for job in selected:
        decisions = decisions_per_episode(study, calibration, job["location"], job["direction"])
        core = job["core"]
        job_s = study["episodes"] * decisions * TRAIN_MS_PER_DECISION[core] / 1000
        train_s += job_s
        units = [(job["location"], job["direction"], run)
                 for run in (test_runs if job["forcing"] == "replay" else [None])]
        rollouts = curve_points(study) * len(units) * holdout
        if job["forcing"] == "synthetic" and with_dataset:  # unchanged-policy transfer
            units += [(job["location"], job["direction"], run) for run in test_runs]
            rollouts += len(test_runs) * holdout
        evaluation_s = rollouts * decisions * EVAL_MS_PER_DECISION[core] / 1000
        policy_s += evaluation_s
        stock_decisions += rollouts * decisions
        job_s += evaluation_s
        by_location[job["location"]] += job_s
        by_core[core] += job_s
        for unit in units:
            for seed in study["holdout_seeds"]:
                stock_keys[(unit, seed)] = decisions
    stock_cached_s = sum(stock_keys.values()) * STOCK_MS_PER_DECISION / 1000
    stock_uncached_s = stock_decisions * STOCK_MS_PER_DECISION / 1000
    total = (train_s + policy_s + stock_cached_s) * scale
    return {
        "jobs": len(selected),
        "training_wall_s": train_s * scale,
        "policy_holdout_wall_s": policy_s * scale,
        "stock_holdout_wall_s_cached": stock_cached_s * scale,
        "stock_holdout_wall_s_uncached": stock_uncached_s * scale,
        "total_wall_s": total,
        "total_wall_s_without_stock_cache":
            (train_s + policy_s + stock_uncached_s) * scale,
        "wall_hours_by_location": {name: round(value * scale / 3600, 2)
                                   for name, value in by_location.most_common()},
        "wall_hours_by_core": {name: round(value * scale / 3600, 2)
                               for name, value in by_core.most_common()},
    }


def shard_filter(study: dict[str, Any], index: int, count: int) -> Callable[[dict[str, Any]], bool]:
    """Round-robin over the declared order, so shards get comparable work."""
    chosen = {jid for k, jid in enumerate(job_id(job) for job in jobs(study)) if k % count == index}
    return lambda job: job_id(job) in chosen


def parse_shard(value: str) -> tuple[int, int]:
    index, _, count = value.partition("/")
    index, count = int(index), int(count)
    if count < 1 or not 1 <= index <= count:
        raise argparse.ArgumentTypeError("--shard must be I/N with 1 <= I <= N")
    return index - 1, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--dataset", type=Path,
                        help="measured-trace root; required for replay forcing, and adds the "
                             "unchanged-policy transfer evaluation to synthetic jobs")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-simulator-proxy", action="store_true",
                        help="acknowledge that every artefact produced is simulator-proxy evidence")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--shard", type=parse_shard, default=None, metavar="I/N")
    parser.add_argument("--cost-scale", type=float, default=1.0,
                        help="multiply the built-in per-decision timings, measured on an "
                             "Apple M5 Pro with lightning.qubit, for a different machine")
    parser.add_argument("--parallel", type=int, default=1,
                        help="shards you intend to run concurrently; scales the estimate by the "
                             "MEASURED speedup curve, which is sublinear, not by the shard count")
    for flag in ("locations", "directions", "cores", "variants", "forcing"):
        parser.add_argument(f"--{flag}", nargs="+", default=None)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=None, dest="training_seeds")
    parser.add_argument("--holdout-seeds", nargs="+", type=int, default=None, dest="holdout_seeds")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--duration-s", type=float, default=None, dest="duration_s")
    parser.add_argument("--curve-every", type=int, default=None, dest="curve_every")
    parser.add_argument("--reward-delta", type=float, default=None, dest="reward_delta",
                        help="override the difference reward's RTT weight (base reward_kwargs.delta)")
    parser.add_argument("--reward-delay-form", choices=("total", "queue"), default=None,
                        dest="reward_delay_form",
                        help="penalise total RTT (manuscript default) or queueing delay rtt - rtt_base")
    parser.add_argument("--reward-queue-floor-ms", type=float, default=5.0, dest="reward_queue_floor_ms",
                        help="q0 of the queue delay form")
    args = parser.parse_args()

    study = load_protocol(args.config)
    apply_scope(study, args)

    job_filter, manifest_name, plan_name = None, "manifest.json", "plan.json"
    if args.shard is not None:
        index, count = args.shard
        job_filter = shard_filter(study, index, count)
        manifest_name = f"manifest.shard-{index + 1}of{count}.json"
        plan_name = f"plan.shard-{index + 1}of{count}.json"
    selected = [job for job in jobs(study) if job_filter is None or job_filter(job)]
    if not selected:
        raise SystemExit("Scope or shard selected no jobs.")
    if args.max_jobs is not None:
        selected = selected[: args.max_jobs]

    out = args.out or ROOT / "outputs" / "rq_study" / study["study_id"]
    calibration = json.loads(args.calibration.read_text())
    estimate = cost(study, selected, calibration, args.dataset is not None, args.cost_scale)
    plan = {
        "study_id": study["study_id"],
        "base_protocol": study["base_protocol"],
        "config": str(args.config),
        "out": str(out),
        "shard": None if args.shard is None else f"{args.shard[0] + 1}/{args.shard[1]}",
        "scope": {key: study[key] for key in _SCOPE},
        "episodes": study["episodes"], "duration_s": study["duration_s"],
        "curve_every": study["curve_every"], "curve_points": curve_points(study),
        "jobs_by_core": dict(Counter(job["core"] for job in selected)),
        "jobs_by_variant": dict(Counter(job["variant"] for job in selected)),
        "jobs_by_forcing": dict(Counter(job["forcing"] for job in selected)),
        "cost": estimate,
        "decisions_per_episode": {
            location: round(decisions_per_episode(study, calibration, location, direction))
            for location in study["locations"] for direction in study["directions"]
        },
        "estimated_wall_hours": estimate["total_wall_s"] / 3600,
        "estimated_wall_hours_without_stock_cache":
            estimate["total_wall_s_without_stock_cache"] / 3600,
        "estimated_wall_hours_at_parallelism": {
            str(args.parallel): round(estimate["total_wall_s"] / speedup(args.parallel) / 3600, 2)},
        "measured_speedup_at_parallelism": round(speedup(args.parallel), 2),
        "recommended_parallelism": RECOMMENDED_PARALLELISM,
        "estimated_wall_hours_at_recommended_parallelism":
            round(estimate["total_wall_s"] / speedup(RECOMMENDED_PARALLELISM) / 3600, 2),
        "cost_model": "per-decision timings measured 2026-09-20 (Apple M5 Pro, lightning.qubit, "
                      "Sydney downlink); re-measure with --execute --max-jobs 1 on other hardware",
        "evidence_tier": "simulator_proxy; capacity_replay_fluid_transport where forcing is replay",
        "execution_mode": "execute" if args.execute else "preflight_only",
    }
    atomic_json(out / plan_name, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))

    if not args.execute:
        print("\nPreflight only. Add --execute --allow-simulator-proxy to train.")
        return
    if not args.allow_simulator_proxy:
        raise SystemExit("--execute requires --allow-simulator-proxy.")
    if "replay" in study["forcing"] and args.dataset is None:
        raise SystemExit("Replay forcing requires --dataset.")

    from qbbr.study.runner import execute  # imported late: pulls in torch and pennylane

    manifest = execute(study, args.calibration, args.dataset, out,
                       max_jobs=args.max_jobs, job_filter=job_filter, manifest_name=manifest_name)
    print(f"\n{manifest['status']}: {len(manifest['completed'])}/{len(manifest['jobs'])} jobs in {out}")


if __name__ == "__main__":
    main()
