"""RQ4-under-gamma5 early picture (5/6 locations -- Sydney's quantum pilot
still training in the background, see train_gamma5_quantum_parallel.py).
Evaluates the quantum gamma=5.0 checkpoints (3-seed pilot) vs. stock BBR-v3
(same style as eval_gamma5_raw.py's classical arm) AND directly vs. the
already-evaluated classical gamma=5.0 checkpoints (outputs/rq1_gamma5_raw
.json), Mann-Whitney on retransmit rate -- mirrors RQ4's published design
(Table tab:qvc: quantum vs. classical, per location) but at gamma=5.0 and
pilot (n=3) scale, so p-values are read qualitatively (n=3's own floor is
p=2/C(6,3)=0.10, per main.tex's own caveat on the earlier 3-seed pilot).

Output -> outputs/rq4_gamma5_partial.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
ACTION_CONFIG_PATH = PACKAGE_ROOT / "configs" / "action_pacing_gain.yaml"
QUANTUM_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints_gamma5" / "pacing_only" / "quantum"
LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]  # all 6 -- Sydney's pilot finished
N_SEEDS = 3
N_EPISODES = 10
EPISODE_S = 300.0
RISK_MODE = "closed_form"
DIRECTION = "downlink"
OUT_PATH = PROJECT_ROOT / "outputs" / "rq4_gamma5_partial.json"


def _eval_one(checkpoint: Path, location: str, calibration, action_config) -> dict[str, Any]:
    from qbbr.agents.quantum.qa2c import QA2CAgent
    from qbbr.eval.scenario_a import simulated_agent_stats

    agent = QA2CAgent(n_layers=2, action_dims=(5,), reupload=False)
    agent.load(str(checkpoint))
    return simulated_agent_stats(
        agent, location, DIRECTION, calibration,
        n_episodes=N_EPISODES, episode_s=EPISODE_S, risk_mode=RISK_MODE, action_config=action_config,
    )


def main() -> None:
    import statistics

    from scipy import stats as scipy_stats

    from qbbr.action.registry import load_action_space
    from qbbr.env.calibration import load_calibration
    from qbbr.eval.metrics import real_cca_per_run_medians

    calibration = load_calibration(CALIBRATION_PATH)
    action_config = load_action_space(ACTION_CONFIG_PATH)
    classical_gamma5 = json.load(open(PROJECT_ROOT / "outputs" / "rq1_gamma5_raw.json"))["qbbr_gamma5"]

    print(f"{'=' * 100}\nRQ4 under gamma=5.0 (partial: {len(LOCATIONS)}/6 locations, Sydney pending)\n{'=' * 100}")
    report = []
    for location in LOCATIONS:
        rtx_q, tput_q = [], []
        for seed in range(N_SEEDS):
            checkpoint = QUANTUM_CHECKPOINT_ROOT / location / f"seed{seed}.pt"
            r = _eval_one(checkpoint, location, calibration, action_config)
            rtx_q.append(r["retransmits_per_s_median"])
            tput_q.append(r["throughput_mbps_median"])
            print(f"  {location:10s} seed={seed}  tput={r['throughput_mbps_median']:7.2f}Mbps  "
                  f"rtx/s={r['retransmits_per_s_median']:6.3f}")

        bbr = real_cca_per_run_medians(PACKAGE_ROOT / "data" / "raw", location, DIRECTION, "bbr")
        rtx_c = classical_gamma5[location]["retransmits_per_s"]

        q_rtx_med, c_rtx_med, bbr_rtx_med = (statistics.median(x) for x in (rtx_q, rtx_c, bbr["retransmits_per_s"]))
        q_tput_med, bbr_tput_med = statistics.median(tput_q), statistics.median(bbr["throughput_mbps"])
        rtx_reduction_vs_bbr = 1.0 - q_rtx_med / bbr_rtx_med if bbr_rtx_med > 0 else float("nan")
        tput_retention_vs_bbr = q_tput_med / bbr_tput_med if bbr_tput_med > 0 else float("nan")
        _u, p_q_vs_c = scipy_stats.mannwhitneyu(rtx_q, rtx_c, alternative="two-sided") if len(set(rtx_q + rtx_c)) > 1 else (None, 1.0)
        # RQ1a's actual significance test: quantum (n=3) vs. real stock BBR-v3 (n=10),
        # not quantum vs. classical -- this is what an RQ1a-style pass/fail needs.
        bbr_rtx_list = bbr["retransmits_per_s"]
        if len(set(rtx_q + bbr_rtx_list)) > 1:
            _u, p_q_vs_bbr = scipy_stats.mannwhitneyu(rtx_q, bbr_rtx_list, alternative="two-sided")
        else:
            p_q_vs_bbr = 1.0
        rq1a_pass = bool(rtx_reduction_vs_bbr >= 0.20 and tput_retention_vs_bbr >= 0.95 and p_q_vs_bbr < 0.05)

        row = {
            "location": location,
            "quantum_gamma5_rtx_median": q_rtx_med, "classical_gamma5_rtx_median": c_rtx_med,
            "bbr_rtx_median": bbr_rtx_med, "retransmit_reduction_vs_bbr": rtx_reduction_vs_bbr,
            "throughput_retention_vs_bbr": tput_retention_vs_bbr, "p_quantum_vs_classical": float(p_q_vs_c),
            "p_quantum_vs_bbr": float(p_q_vs_bbr), "rq1a_criteria_pass": rq1a_pass,
        }
        report.append(row)
        flag = "PASS" if rq1a_pass else "FAIL"
        print(f"{location:10s}  quantum_rtx={q_rtx_med:6.3f}  classical_rtx={c_rtx_med:6.3f}  bbr_rtx={bbr_rtx_med:6.2f}  "
              f"rtx_reduction(vs bbr)={rtx_reduction_vs_bbr:+7.1%}  tput_retention={tput_retention_vs_bbr:6.1%}  "
              f"p(quantum vs classical)={p_q_vs_c:.4f}  p(quantum vs bbr)={p_q_vs_bbr:.4f}  [{flag}]\n")

    OUT_PATH.write_text(json.dumps({"meta": {"n_seeds": N_SEEDS, "n_episodes": N_EPISODES, "gamma": 5.0,
                                              "locations_complete": LOCATIONS, "sydney_pending": False},
                                     "rq4_gamma5_partial": report}, indent=2))
    print(f"saved -> {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
