"""Corrected shifted-freeze test (supersedes the earlier single-offset check,
which was invalid -- it patched the single shared _HANDOVER_PHASE_PROFILE
object that BOTH in_reconfig_freeze_window (fluid_env.step()) and
retransmit_phase_multiplier (fluid_sim.step_fluid_state) read from, so
freeze's window and the retransmit-concentration point moved together and
could never show a difference regardless of whether freeze is genuinely
phase-specific).

This version decouples them: retransmit_phase_multiplier always sees the
REAL calibrated profile (mean_phase_s=10.5, r_bar=0.7368, unpatched --
retransmits concentrate here regardless of what freeze does), while
in_reconfig_freeze_window is monkeypatched (via the qbbr.env.fluid_env
module-level name it's imported under) to check against a DECOY profile
whose mean_phase_s is swept across several offsets spread around the 15s
cycle, mirroring the circular-shift null already used for the Rayleigh
test (validate_handover_cadence.py) but applied to the freeze intervention
rather than to an observable.

Freeze-at-real-phase (10.5, included as one of the sweep points) beating
the decoy offsets -> phase-specific signal, the anticipatory claim holds.
Freeze-at-real-phase indistinguishable from the decoys -> freeze is
generic "do less x% of the time" regularization/variance reduction, no
anticipatory content, matching what the earlier (invalid) test suggested
but now on a methodologically sound footing.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
ACTION_CONFIG_PATH = PACKAGE_ROOT / "configs" / "action_pacing_gain.yaml"
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints_final" / "pacing_only"
LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
N_SEEDS = 10  # matches the standard 10-seed protocol used everywhere else (RQ1-RQ4) -- the
# original 5-seed run's post-hoc MDE was ~12-17% at 5/6 locations (only Sydney, with much lower
# per-seed variance, was powered down to ~1.3%), well above the -1.3%..+6.8% decoy spread actually
# observed, so a real effect on the order of the freeze extension's own claimed magnitude (single-
# digit %) could have been missed at those 5 locations. 10 seeds roughly doubles n on both arms.
N_EPISODES = 5
EPISODE_S = 300.0
REAL_MEAN_PHASE_S = 10.5
R_BAR = 0.7368
# Decoy freeze-center offsets spread around the 15s cycle, each >=1s (the
# freeze half-width) from the real phase so decoy windows never overlap it.
DECOY_OFFSETS_S = [0.0, 2.5, 5.0, 7.5, 13.0]
ALL_OFFSETS_S = [REAL_MEAN_PHASE_S] + DECOY_OFFSETS_S
OUT_PATH = PROJECT_ROOT / "outputs" / "shifted_freeze_report.json"


def _eval_one_job(job: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)

    import qbbr.env.fluid_env as fe
    from qbbr.action.registry import load_action_space
    from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
    from qbbr.env.calibration import load_calibration
    from qbbr.env.fluid_sim import PhaseProfile, in_reconfig_freeze_window as _real_freeze_check
    from qbbr.eval.scenario_a import simulated_agent_stats

    decoy_profile = PhaseProfile(mean_phase_s=job["freeze_offset_s"], r_bar=R_BAR)

    def _patched_freeze_check(t_s, phase_offset_s, phase_profile, half_width_s=1.0):
        return _real_freeze_check(t_s, phase_offset_s, decoy_profile, half_width_s)

    fe.in_reconfig_freeze_window = _patched_freeze_check

    calibration = load_calibration(CALIBRATION_PATH)
    action_config = load_action_space(ACTION_CONFIG_PATH)
    agent = MLPA2CAgent(n_layers=2, action_dims=(5,))
    agent.load(job["checkpoint"])

    stats = simulated_agent_stats(
        agent, job["location"], "downlink", calibration,
        n_episodes=N_EPISODES, episode_s=EPISODE_S, risk_mode="closed_form",
        action_config=action_config,
    )
    return {"location": job["location"], "seed": job["seed"], "freeze_offset_s": job["freeze_offset_s"], **stats}


def build_jobs() -> list[dict[str, Any]]:
    jobs = []
    for location in LOCATIONS:
        for seed in range(N_SEEDS):
            checkpoint = CHECKPOINT_ROOT / "classical" / location / f"seed{seed}.pt"
            for offset in ALL_OFFSETS_S:
                jobs.append({
                    "location": location, "seed": seed, "checkpoint": str(checkpoint), "freeze_offset_s": offset,
                })
    return jobs


def mann_whitney(a: list[float], b: list[float]) -> float:
    from scipy import stats as scipy_stats

    if len(set(a + b)) == 1:
        return 1.0
    _u, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(p)


def main() -> None:
    jobs = build_jobs()
    print(f"{len(jobs)} eval job(s): {len(LOCATIONS)} location(s) x {N_SEEDS} seed(s) x {len(ALL_OFFSETS_S)} offset(s)")

    t0 = time.time()
    results: dict[tuple[str, float], list[float]] = {}
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_eval_one_job, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            r = future.result()
            key = (r["location"], r["freeze_offset_s"])
            results.setdefault(key, []).append(r["retransmits_per_s_median"])
            elapsed = time.time() - t0
            tag = "REAL" if r["freeze_offset_s"] == REAL_MEAN_PHASE_S else "decoy"
            print(f"[{i}/{len(jobs)}] {elapsed:7.1f}s  {job['location']:10s} seed={job['seed']}  "
                  f"offset={r['freeze_offset_s']:5.2f}s [{tag:5s}]  rtx/s={r['retransmits_per_s_median']:6.2f}")

    report = []
    print(f"\n{'=' * 100}\nShifted-freeze: real phase (10.5s) vs. pooled decoy offsets, per location\n{'=' * 100}")
    for location in LOCATIONS:
        real_vals = results[(location, REAL_MEAN_PHASE_S)]
        decoy_vals = []
        for offset in DECOY_OFFSETS_S:
            decoy_vals.extend(results[(location, offset)])
        real_med = sum(real_vals) / len(real_vals)
        decoy_med = sum(decoy_vals) / len(decoy_vals)
        p = mann_whitney(real_vals, decoy_vals)
        delta = (real_med / decoy_med - 1.0) if decoy_med > 0 else float("nan")
        row = {
            "location": location, "real_phase_rtx_mean": real_med, "decoy_pooled_rtx_mean": decoy_med,
            "delta": delta, "p": p, "real_better": (real_med < decoy_med) and (p < 0.05),
        }
        report.append(row)
        flag = "REAL PHASE SIGNIFICANTLY BETTER" if row["real_better"] else "n.s. (freeze is phase-independent here)"
        print(f"{location:10s}  real={real_med:7.2f}  decoy_pooled={decoy_med:7.2f}  delta={delta:+7.1%}  "
              f"p={p:.4f}  [{flag}]")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "meta": {"n_seeds": N_SEEDS, "n_episodes": N_EPISODES, "offsets_s": ALL_OFFSETS_S,
                  "real_phase_s": REAL_MEAN_PHASE_S, "total_elapsed_s": time.time() - t0},
        "report": report,
    }, indent=2))
    print(f"\ndone in {time.time() - t0:.1f}s -> {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
