"""Replay a trained native QA2C / A2C checkpoint and log the decision path.

For every simulator decision this records the canonical state, the masked
actor logits, the raw masked probabilities, the actions the shared
:class:`NativeActionSelector` still admits, the selector-effective
probabilities, the chosen action, and the resulting throughput / retransmit /
RTT / queue outcome.  The aggregate answers two diagnostic questions:

  * When does the policy pick a high native gain (1.10 / 1.25), stratified by
    ``s2`` (excess RTT), ``s3`` (inflight/BDP), ``s4`` (queue) and ``s7``
    (reconfiguration-phase proximity)?
  * How often does the simulator actually hand the agent a non-trivial
    decision, i.e. is the agent deciding far more often than a native BBR-v3
    ProbeBW override would apply?

This is a simulator diagnostic.  It produces no field-performance claim and
does not train.

    python -m qbbr.scripts.replay_native_policy \
        --checkpoint outputs/tier1_native_qa2c_checkpoints/quantum/London/downlink/seed0.pt \
        --core quantum --location London --direction downlink --seed 1000 \
        --recovery-proxy --out outputs/replay_qa2c_london_dl_seed0.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qbbr.control.native_action_selector import NativeActionSelector
from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.eval.native_qa2c_matched import build_matched_native_a2c_agents

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CALIBRATION = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
GAIN_BY_INDEX = {0: 0.75, 1: 0.90, 2: 1.00, 3: 1.10, 4: 1.25}
HIGH_GAIN = (3, 4)
STATE_NAMES = ["s1_bhat", "s2_rtt_ratio", "s3_inflight_bdp", "s4_queue",
               "s5_handover_eta", "s6_p_tot", "s7_reconfig_phase"]


def _default_selector() -> NativeActionSelector:
    """The selector-v2 pilot parameters, minus the opt-in RTT/reconfig gates."""
    return NativeActionSelector(
        stock_action=2, min_logit_advantage=0.15, confidence_temperature=0.10,
        max_inflight_state=0.45, max_queue_state=0.20, high_gain_actions=HIGH_GAIN,
    )


def _load_checkpoint(agent, core: str, path: Path) -> None:
    """Load a runner or agent checkpoint (trusted project file, torch>=2.6 safe)."""
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    state = raw.get("agent_state", raw)
    modules = (("actor_qnn", "actor_head", "critic_qnn", "critic_head")
               if core == "quantum" else ("actor", "critic"))
    for name in modules:
        getattr(agent, name).load_state_dict(state[name])
    if "optimizer" in state:
        try:
            agent.optimizer.load_state_dict(state["optimizer"])
        except (ValueError, KeyError):
            pass


def _raw_logits(agent, state_t: torch.Tensor) -> torch.Tensor:
    if hasattr(agent, "_actor_logits"):
        return agent._actor_logits(state_t)
    return agent.actor(state_t)


def _bucket(value: float, edges=(0.2, 0.4, 0.6, 0.8)) -> str:
    lo = 0.0
    for hi in edges:
        if value < hi:
            return f"[{lo:.1f},{hi:.1f})"
        lo = hi
    return f"[{lo:.1f},1.0]"


def replay(agent, selector, env, seed: int, deterministic: bool) -> dict:
    state = env.reset(seed=seed)
    rows = []
    done = False
    stock_forced = decisions = 0
    t_dec_values = []
    while not done:
        allowed = env.allowed_action_indices()
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        with torch.no_grad():
            logits = _raw_logits(agent, state_t)
            masked = torch.full_like(logits, float("-inf"))
            masked[list(allowed)] = logits[list(allowed)]
            raw_probs = torch.softmax(masked, dim=-1)
            admitted = selector.admissible_actions(logits, state_t, allowed)
            effective = selector.distribution(logits, state_t, allowed).probs
        action, _ = agent.act(state, allowed, deterministic=deterministic)
        next_state, reward, done, info = env.step(action)
        t_dec = info["t_dec_s"]
        t_dec_values.append(t_dec)
        decisions += 1
        if len(allowed) <= 1:
            stock_forced += 1
        rows.append({
            "t_s": round(float(info.get("t_s", 0.0)), 3),
            "state": {n: round(float(v), 4) for n, v in zip(STATE_NAMES, np.asarray(state))},
            "allowed": list(allowed),
            "n_allowed": len(allowed),
            "logits": [round(float(x), 4) for x in logits.tolist()],
            "raw_probs": [round(float(x), 4) for x in raw_probs.tolist()],
            "admitted": list(admitted),
            "effective_probs": [round(float(x), 4) for x in effective.tolist()],
            "action": int(action),
            "gain": GAIN_BY_INDEX[int(action)],
            "throughput_mbps": round(info["delivered_bytes"] * 8.0 / t_dec / 1e6, 4),
            "retransmits_per_s": round(info["retransmits"] / t_dec, 4),
            "rtt_ms": round(info["rtt_ms"], 2),
            "queue_i_dwn": round(max(info["i_dwn"], 0.0), 4),
            "bbr_bw_est_bps": round(info.get("bbr_bw_est_bps", 0.0), 1),
        })
        state = next_state

    n = len(rows)
    shares = {str(a): sum(1 for r in rows if r["action"] == a) / n for a in range(5)}
    high_rows = [r for r in rows if r["action"] in HIGH_GAIN]
    strat = {}
    for key in ("s2_rtt_ratio", "s3_inflight_bdp", "s4_queue", "s7_reconfig_phase"):
        overall_mean = float(np.mean([r["state"][key] for r in rows]))
        high_mean = float(np.mean([r["state"][key] for r in high_rows])) if high_rows else None
        buckets: dict[str, dict] = {}
        for r in rows:
            b = _bucket(r["state"][key])
            slot = buckets.setdefault(b, {"n": 0, "high_gain": 0})
            slot["n"] += 1
            if r["action"] in HIGH_GAIN:
                slot["high_gain"] += 1
        strat[key] = {
            "overall_mean": round(overall_mean, 4),
            "mean_when_high_gain": round(high_mean, 4) if high_mean is not None else None,
            "high_gain_share_by_bucket": {
                b: round(v["high_gain"] / v["n"], 3) for b, v in sorted(buckets.items())
            },
        }
    return {
        "seed": seed,
        "deterministic": deterministic,
        "decisions": n,
        "episode_s": round(sum(t_dec_values), 1),
        "mean_t_dec_s": round(float(np.mean(t_dec_values)), 4),
        "effective_decision_hz": round(1.0 / float(np.mean(t_dec_values)), 3),
        "stock_forced_decisions": stock_forced,
        "non_trivial_decisions": n - stock_forced,
        "probe_bw_phase_restriction": False,
        "probe_bw_note": (
            "FluidSimEnv.allowed_action_indices() gates only STARTUP and the "
            "reconfiguration freeze window; it does not restrict overrides to the "
            "PROBE_BW / CRUISE phase, so the agent decides every T_dec."
        ),
        "action_shares": {k: round(v, 4) for k, v in shares.items()},
        "high_gain_share": round(sum(shares[str(a)] for a in HIGH_GAIN), 4),
        "throughput_mbps_mean": round(float(np.mean([r["throughput_mbps"] for r in rows])), 4),
        "retransmits_per_s_mean": round(float(np.mean([r["retransmits_per_s"] for r in rows])), 4),
        "rtt_ms_mean": round(float(np.mean([r["rtt_ms"] for r in rows])), 2),
        "stratification": strat,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--core", choices=["quantum", "classical"], required=True)
    ap.add_argument("--location", required=True)
    ap.add_argument("--direction", required=True, choices=["downlink", "uplink"])
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--duration-s", type=float, default=300.0)
    ap.add_argument("--selector", choices=["none", "v2"], default="none",
                    help="'none' matches a checkpoint trained without the selector "
                         "(the v1 Tier-1 runs); 'v2' matches the selector-v2 pilot.")
    ap.add_argument("--recovery-proxy", action="store_true",
                    help="enable the uncalibrated bandwidth_estimate_recovery_s=3.0 proxy")
    ap.add_argument("--sample", action="store_true", help="sample actions instead of argmax")
    ap.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    reference_selector = _default_selector()   # always used for the diagnostic columns
    agent_selector = reference_selector if args.selector == "v2" else None
    calibration = load_calibration(args.calibration_path)
    quantum, classical, match = build_matched_native_a2c_agents(
        observation_dim=7, action_count=5, n_layers=2, selector=agent_selector, entropy_coef=0.005,
    )
    agent = quantum if args.core == "quantum" else classical
    _load_checkpoint(agent, args.core, args.checkpoint)

    overrides = {"bandwidth_estimate_recovery_s": 3.0} if args.recovery_proxy else None
    env = FluidSimEnv(
        args.location, args.direction, calibration, risk_mode="stub_constant",
        episode_s=args.duration_s, reward_mode="throughput_only", dynamics_overrides=overrides,
    )

    result = {
        "evidence_tier": "simulator_diagnostic_only",
        "warning": "Policy-replay diagnostic. Not an RL training or field-performance result.",
        "reward_contract": "supervisor-locked-throughput-only-v1",
        "checkpoint": str(args.checkpoint),
        "core": args.core,
        "location": args.location,
        "direction": args.direction,
        "recovery_proxy": bool(args.recovery_proxy),
        "parameter_match": {
            "quantum_params": match.quantum_params, "classical_params": match.classical_params,
            "actor_hidden_dim": match.actor_hidden_dim, "critic_hidden_dim": match.critic_hidden_dim,
        },
        "agent_selector": args.selector,
        "reference_selector": {
            "min_logit_advantage": reference_selector.min_logit_advantage,
            "confidence_temperature": reference_selector.confidence_temperature,
            "max_inflight_state": reference_selector.max_inflight_state,
            "max_queue_state": reference_selector.max_queue_state,
            "max_excess_rtt_state": reference_selector.max_excess_rtt_state,
            "max_reconfig_phase_proximity": reference_selector.max_reconfig_phase_proximity,
        },
        "replay": replay(agent, reference_selector, env, args.seed, deterministic=not args.sample),
    }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))

    rep = result["replay"]
    print(f"{args.core} {args.location}/{args.direction} seed={args.seed} "
          f"recovery_proxy={bool(args.recovery_proxy)}")
    print(f"  decisions={rep['decisions']}  mean_t_dec={rep['mean_t_dec_s']}s  "
          f"~{rep['effective_decision_hz']} Hz   stock_forced={rep['stock_forced_decisions']}  "
          f"(no PROBE_BW-phase restriction)")
    print(f"  action shares: " + "  ".join(f"{GAIN_BY_INDEX[a]}={rep['action_shares'][str(a)]:.2f}"
                                           for a in range(5)))
    print(f"  throughput={rep['throughput_mbps_mean']} Mbps   "
          f"retransmits={rep['retransmits_per_s_mean']}/s   rtt={rep['rtt_ms_mean']} ms")
    for key, s in rep["stratification"].items():
        print(f"  {key}: overall_mean={s['overall_mean']}  when_high_gain={s['mean_when_high_gain']}  "
              f"high_gain_share_by_bucket={s['high_gain_share_by_bucket']}")
    if args.out is not None:
        print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
