"""Single entry point for the complete four-RQ study.

The default command is read-only and emits a machine-readable execution plan.
``--execute`` is required for simulator jobs.  Field measurements are never
started by this script: they are ingested through ``assess_collection`` after
an independently audited collector has produced JSONL records.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from qbbr.eval.active_protocol import validate_active_protocol
from qbbr.data.collection.jsonl import read_jsonl_rows
from qbbr.data.collection.schema import validate_collection_rows

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "qbbr" / "configs" / "active_study_protocol.yaml"
DEFAULT_RQ_CONFIG = ROOT / "qbbr" / "configs" / "rq_complete_study.yaml"


def build_plan(protocol: dict[str, Any], rq: dict[str, Any]) -> dict[str, Any]:
    readiness = validate_active_protocol(protocol)
    locations = rq["locations"]
    directions = rq["directions"]
    seeds = rq["training_seeds"]
    return {
        "study_id": rq["study_id"],
        "readiness": {"valid": readiness.valid, "missing": list(readiness.missing)},
        "experiments": {
            "rq1_baseline_transfer": {"locations": locations, "directions": directions, "forcing": rq["forcing"]},
            "rq2_state_ablation": {"variants": rq["variants"], "cores": ["qa2c", "a2c"], "seeds": seeds},
            "rq3_qdqn": {"cores": ["qa2c", "a2c", "qdqn"], "seeds": seeds},
            "rq4_generalisation": {
                "mixed_flow": True,
                "application_profiles": protocol["benchmark"]["application_profiles"],
                "field_gate": protocol["collection"],
            },
        },
        "field_evidence_boundary": "Field claims require assessed JSONL coverage and are never inferred from simulator output.",
        "commands": {
            "primary": "python -m qbbr.scripts.run_native_qa2c_successor --execute --allow-simulator-proxy",
            "qdqn": "python -m qbbr.scripts.run_recurrent_qdqn_ablation --execute --allow-simulator-proxy",
            "scenario_b": "python -m qbbr.scripts.train_scenario_b --config qbbr/configs/eval_scenarioB.yaml --location Sydney --direction downlink",
            "field_assessment": "python -m qbbr.scripts.assess_collection FIELD.jsonl --require-ready",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rq-config", type=Path, default=DEFAULT_RQ_CONFIG)
    parser.add_argument("--field-records", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "complete_rq_study_plan.json")
    parser.add_argument("--execute", action="store_true", help="allow downstream simulator commands to be run")
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    rq = yaml.safe_load(args.rq_config.read_text(encoding="utf-8"))
    plan = build_plan(protocol, rq)
    if args.field_records:
        report = validate_collection_rows(read_jsonl_rows(args.field_records))
        plan["field_collection"] = report.as_dict()
        field_cfg = rq["field"]
        longitudinal_ok = (report.longitudinal_span_days or 0.0) >= float(field_cfg["minimum_span_days"])
        per_terminal_ok = all(
            days >= int(field_cfg["minimum_distinct_days_per_terminal"])
            for days in (report.distinct_days_by_terminal or {}).values()
        )
        plan["field_collection"]["protocol_gate"] = bool(
            report.is_minimally_ready
            and report.terminals >= int(field_cfg["minimum_terminals"])
            and report.locations >= int(field_cfg["minimum_locations"])
            and longitudinal_ok and per_terminal_ok
        )
        if not plan["field_collection"]["protocol_gate"]:
            plan["field_collection_warning"] = "Field gate is incomplete; no field-performance claim is permitted."
    plan["execution_mode"] = "simulator_execute_acknowledged" if args.execute else "preflight_only"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not plan["readiness"]["valid"]:
        raise SystemExit("Protocol is incomplete; resolve the listed commitments before claiming a complete study.")


if __name__ == "__main__":
    main()
