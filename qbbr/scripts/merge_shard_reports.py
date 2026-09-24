"""Merge sharded successor-protocol reports and assess the gates once, globally."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from qbbr.env.calibration import load_calibration
from qbbr.eval.successor_protocol import assess_full_successor_records

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CALIBRATION = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION,
                        help="Per-path constants; needed to resolve derived_from criteria.")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Assess even if a cell is missing seeds (clearly labelled).")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    expected_seeds = set(config["training"]["training_seeds"])

    plan, records, stock_evaluations = None, [], []
    missing_files = []
    for shard_path in sorted(args.shards):
        if not shard_path.exists():
            missing_files.append(str(shard_path))
            continue
        if shard_path.name.endswith(".partial.json"):
            continue
        shard = json.loads(shard_path.read_text())
        if plan is None:
            plan = {k: v for k, v in shard.items()
                    if k not in {"records", "stock_evaluations", "selection_assessment",
                                 "assessment_skipped_reason"}}
        records.extend(shard.get("records", []))
        stock_evaluations.extend(shard.get("stock_evaluations", []))

    if plan is None:
        raise SystemExit("No shard reports could be read.")
    if missing_files:
        print(f"[merge] WARNING: {len(missing_files)} shard file(s) missing: {missing_files}")

    seen: dict[tuple, int] = defaultdict(int)
    for record in records:
        seen[(record["location"], record["direction"], record["core"], record["seed"])] += 1
    duplicated = {key: count for key, count in seen.items() if count > 1}
    if duplicated:
        raise SystemExit(
            f"Refusing to assess: {len(duplicated)} record(s) appear more than once "
            f"(e.g. {sorted(duplicated)[0]} x{list(duplicated.values())[0]}). "
            "The shard list overlaps; pass each shard report exactly once."
        )

    groups: dict[tuple, set] = defaultdict(set)
    for record in records:
        groups[(record["location"], record["direction"], record["core"])].add(record["seed"])
    incomplete = {key: sorted(expected_seeds - seeds) for key, seeds in groups.items()
                  if expected_seeds - seeds}
    if incomplete:
        print(f"[merge] INCOMPLETE seed groups ({len(incomplete)}):")
        for key, missing in sorted(incomplete.items()):
            print(f"          {key}: missing seeds {missing}")
        if not args.allow_incomplete:
            raise SystemExit(
                "Refusing to assess: the positive-seed fraction and bootstrap CI would be "
                "computed over an incomplete group. Re-run the missing shards, or pass "
                "--allow-incomplete to assess anyway."
            )

    plan["records"] = records
    plan["stock_evaluations"] = stock_evaluations
    plan["selection_assessment"] = assess_full_successor_records(
        records, config["selection_criteria"], config["bootstrap"], load_calibration(args.calibration)
    )
    plan["merged_from_shards"] = [str(p) for p in sorted(args.shards)]
    if incomplete:
        plan["incomplete_seed_groups"] = {str(k): v for k, v in incomplete.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2))

    print(f"[merge] {len(records)} records from {len(args.shards)} shards -> {args.out}")
    for assessment in plan["selection_assessment"]:
        gates = assessment["criteria_pass"]
        failed = [name for name, ok in gates.items() if not ok]
        status = "QUALIFIED" if assessment["qualified_simulator_proxy_result"] else "not qualified"
        print(f"  {assessment['core']:<9} {assessment['location']:<8} {assessment['direction']:<9} "
              f"seeds={assessment['independent_training_seed_count']}  {status}"
              + (f"   failed: {', '.join(failed)}" if failed else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
