"""Audit whether the repository is ready to run the four RQ families."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def audit() -> dict:
    rq = yaml.safe_load((ROOT / "qbbr/configs/rq_complete_study.yaml").read_text())
    active = yaml.safe_load((ROOT / "qbbr/configs/active_study_protocol.yaml").read_text())
    checks = {
        "rq1_entrypoint": (ROOT / "qbbr/scripts/run_native_qa2c_successor.py").exists(),
        "rq2_ablation_contract": (ROOT / "qbbr/eval/state_ablation.py").exists(),
        "rq2_all_variants_declared": set(rq["variants"]) >= {
            "full", "telemetry", "telemetry_queue", "no_queue", "no_handover", "no_failure", "no_phase"
        },
        "rq3_qdqn_entrypoint": (ROOT / "qbbr/scripts/run_recurrent_qdqn_ablation.py").exists(),
        "rq123_matrix_entrypoint": (ROOT / "qbbr/scripts/run_rq_study_matrix.py").exists(),
        "rq123_screen_protocol": (ROOT / "qbbr/configs/rq_study_screen.yaml").exists(),
        "rq4_mixed_flow_env_file_exists": (ROOT / "qbbr/env/multi_flow_env.py").exists(),
        "rq4_native_contract_mixed_flow_env": (ROOT / "qbbr/env/native_multi_flow_env.py").exists(),
        "rq4_measured_coexistence_analysis": (ROOT / "qbbr/scripts/analyze_measured_coexistence.py").exists(),
        "rq4_coexistence_overlay": (ROOT / "qbbr/configs/native_coexistence.yaml").exists(),
        "rq4_application_profiles_declared": set(active["benchmark"]["application_profiles"]) >= {
            "video_streaming", "voip", "gaming"
        },
        "field_collection_schema": (ROOT / "qbbr/data/collection/schema.py").exists(),
        "field_multi_terminal_gate": active["collection"]["minimum_independent_terminals"] >= 2,
        "field_longitudinal_gate": active["collection"].get("longitudinal_capture") == "multi_week",
    }
    return {
        "checks": checks,
        "ready_for_simulator_smoke": all(checks.values()),
        "rq123_execution": "run_rq_study_matrix executes the RQ1/RQ2/RQ3 matrix; "
                           "rq_study_screen.yaml is the affordable scope and "
                           "rq_complete_study.yaml is the full declared design",
        "rq4_execution": "PARTIAL: measured coexistence is available via "
                         "analyze_measured_coexistence. NativeMultiFlowEnv runs under the "
                         "frozen contract with the native_coexistence overlay, and the "
                         "agent's share is validated against measurement; the competitor "
                         "mix is NOT, so simulated fairness claims stay blocked and only "
                         "policy-versus-stock contrasts are supported. Application "
                         "profiles remain declared names with no traffic model.",
        "field_execution": "requires real JSONL telemetry; never inferred from simulator output",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="exit non-zero if any contract check fails")
    args = parser.parse_args()
    result = audit()
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.strict and not result["ready_for_simulator_smoke"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
