"""CDF + boxplot figures for the current Tier-1 native-QA2C results.

Re-evaluates every QA2C brain checkpoint (v3b -> v4 -> v5, quantum) and the
v5 matched Classical A2C checkpoints on the CURRENT simulator (v5 config:
drain_throughput_penalty 0.15, probe_bw_phase_gate on,
bandwidth_estimate_recovery_s 3.0), capturing throughput per holdout seed so
the distributions are genuine (train seed x 10 holdout seeds x 4 conditions).

One consistent policy lens: raw learned policy, masked argmax, selector OFF.
For v5 this coincides with the deployed policy (the v5 report shows identical
deployed/learned action shares); for v3b/v4 it exposes what the network
actually learned under the shield in force at the time.

Writes to figures/:
  tier1_v5_delta_boxplot_by_condition.png
  tier1_v5_throughput_cdf.png
  tier1_v5_perseed_strip.png
  tier1_iteration_delta_boxplot.png
  tier1_v5_throughput_vs_retransmit.png
and outputs/tier1_result_figure_data.json
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from qbbr.env.calibration import load_calibration
from qbbr.scripts.run_native_qa2c_successor import PACKAGE_ROOT, _env
from qbbr.agents.quantum.native_qa2c import NativeQA2CAgent
from qbbr.agents.classical.native_a2c import NativeMLPA2CAgent

FIGDIR = Path("figures"); FIGDIR.mkdir(exist_ok=True)
CONDS = [("London", "downlink"), ("London", "uplink"),
         ("Sydney", "downlink"), ("Sydney", "uplink")]
CORE_COLOR = {"quantum": "#54a24b", "classical": "#e45756"}
ITER_COLOR = {"v3b": "#b0b0b0", "v4": "#4c78a8", "v5": "#54a24b"}

CONFIG = yaml.safe_load(
    Path("qbbr/configs/tier1_native_qa2c_successor_protocol.yaml").read_text())
CALIB = load_calibration(PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json")
HOLDOUT = CONFIG["evaluation"]["holdout_seeds"]


def _rollout(agent, loc, direction, hseed, stock_action=None):
    env = _env(CALIB, loc, direction, CONFIG)
    state, done = env.reset(seed=hseed), False
    tp, rtx, rtt = [], [], []
    while not done:
        a = stock_action if stock_action is not None else agent.act(
            state, env.allowed_action_indices(), deterministic=True, deployment=False)[0]
        state, _r, done, info = env.step(a)
        tp.append(info["delivered_bytes"] * 8.0 / info["t_dec_s"] / 1e6)
        rtx.append(info["retransmits"] / info["t_dec_s"])
        rtt.append(info["rtt_ms"])
    return float(np.mean(tp)), float(np.mean(rtx)), float(np.mean(rtt))


def _load_quantum(path):
    st = torch.load(path, map_location="cpu", weights_only=False)["agent_state"]
    ag = NativeQA2CAgent(observation_dim=int(st["observation_dim"]),
                         native_action_count=int(st["native_action_count"]),
                         n_layers=2, entropy_coef=float(st["entropy_coef"]), selector=None)
    ag.load_training_state_dict(st)
    return ag


def _load_classical(path):
    st = torch.load(path, map_location="cpu", weights_only=False)["agent_state"]
    ag = NativeMLPA2CAgent(observation_dim=int(st["observation_dim"]),
                           native_action_count=int(st["native_action_count"]),
                           actor_hidden_dim=int(st["actor_hidden_dim"]),
                           critic_hidden_dim=int(st["critic_hidden_dim"]),
                           entropy_coef=float(st["entropy_coef"]), selector=None)
    ag.load_training_state_dict(st)
    return ag


def reeval() -> list[dict]:
    jobs = [("v3b", "quantum", _load_quantum), ("v4", "quantum", _load_quantum),
            ("v5", "quantum", _load_quantum), ("v5", "classical", _load_classical)]
    rows: list[dict] = []
    for loc, direction in CONDS:
        stock_tp = {}
        for hs in HOLDOUT:
            tp, rtx, rtt = _rollout(None, loc, direction, hs, stock_action=2)
            stock_tp[hs] = tp
            rows.append(dict(iter="stock", core="stock", seed=-1, location=loc,
                             direction=direction, holdout_seed=hs, tput_mbps=tp,
                             rtx_per_s=rtx, rtt_ms=rtt, delta_pct=0.0))
        for it, core, loader in jobs:
            for path in sorted(glob.glob(
                    f"outputs/tier1_{it}_checkpoints/{core}/{loc}/{direction}/seed*.pt")):
                seed = int(path.split("seed")[-1].split(".")[0])
                ag = loader(path)
                d = []
                for hs in HOLDOUT:
                    tp, rtx, rtt = _rollout(ag, loc, direction, hs)
                    delta = 100.0 * (tp / stock_tp[hs] - 1.0)
                    d.append(delta)
                    rows.append(dict(iter=it, core=core, seed=seed, location=loc,
                                     direction=direction, holdout_seed=hs, tput_mbps=tp,
                                     rtx_per_s=rtx, rtt_ms=rtt, delta_pct=delta))
                print(f"[reeval] {it:3s} {core:9s} {loc:7s} {direction:8s} seed{seed}  "
                      f"medΔ={np.median(d):+6.2f}%", flush=True)
    Path("outputs/tier1_result_figure_data.json").write_text(
        json.dumps({"config_simulator": CONFIG["simulator"], "rows": rows}, indent=2))
    print("outputs/tier1_result_figure_data.json written")
    return rows


def _sub(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def fig_by_condition(rows):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, (loc, direction) in zip(axes.flat, CONDS):
        data, labels, colors = [], [], []
        for core in ("classical", "quantum"):
            v = [r["delta_pct"] for r in _sub(rows, core=core, iter="v5",
                                              location=loc, direction=direction)]
            data.append(v); labels.append(f"{core}\nn={len(v)}"); colors.append(CORE_COLOR[core])
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", lw=2))
        for p, c in zip(bp["boxes"], colors):
            p.set_facecolor(c); p.set_alpha(0.35)
        for i, v in enumerate(data, 1):
            ax.scatter(np.random.default_rng(i).normal(i, 0.05, len(v)), v, s=16,
                       color=colors[i - 1], edgecolor="black", lw=0.3, zorder=3)
        ax.axhline(0, color="crimson", ls="--", lw=1)
        ax.set_title(f"{loc} / {direction}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_ylabel("throughput Δ vs stock (%)")
    fig.suptitle("Tier-1 v5 — throughput Δ vs stock BBR-v3, per holdout-seed "
                 "(5 train seeds × 10 holdout seeds; raw learned policy, current simulator)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(FIGDIR / "tier1_v5_delta_boxplot_by_condition.png", dpi=150)
    plt.close(fig); print("figures/tier1_v5_delta_boxplot_by_condition.png")


def fig_cdf(rows):
    fig, ax = plt.subplots(figsize=(8, 5))
    for core in ("quantum", "classical"):
        v = np.sort([r["delta_pct"] for r in _sub(rows, core=core, iter="v5")])
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", lw=2,
                color=CORE_COLOR[core], label=f"{core} A2C  (n={len(v)})")
    ax.axvline(0, color="crimson", ls="--", lw=1, label="stock BBR-v3")
    ax.set_xlabel("throughput Δ vs stock (%)  — per (train seed × holdout seed × condition)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Tier-1 v5 — CDF of throughput Δ vs stock (current simulator)")
    ax.grid(alpha=0.3); ax.legend(loc="center right", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGDIR / "tier1_v5_throughput_cdf.png", dpi=150)
    plt.close(fig); print("figures/tier1_v5_throughput_cdf.png")


def fig_perseed_strip(rows):
    seeds = sorted({r["seed"] for r in _sub(rows, iter="v5", core="quantum")})
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, core in enumerate(("quantum", "classical")):
        for k, s in enumerate(seeds):
            v = [r["delta_pct"] for r in _sub(rows, iter="v5", core=core, seed=s)]
            x = k + (j - 0.5) * 0.34
            ax.scatter(np.random.default_rng(k + j).normal(x, 0.03, len(v)), v, s=14,
                       color=CORE_COLOR[core], edgecolor="black", lw=0.25, alpha=0.7,
                       label=core if k == 0 else None)
            ax.hlines(np.median(v), x - 0.14, x + 0.14, color="black", lw=2, zorder=4)
    ax.axhline(0, color="crimson", ls="--", lw=1)
    ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("throughput Δ vs stock (%)  — 10 holdout × 4 conditions")
    ax.set_title("Tier-1 v5 — per training-seed throughput Δ vs stock")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGDIR / "tier1_v5_perseed_strip.png", dpi=150)
    plt.close(fig); print("figures/tier1_v5_perseed_strip.png")


def fig_iterations(rows):
    fig, ax = plt.subplots(figsize=(9, 6))
    data, labels = [], []
    for it in ("v3b", "v4", "v5"):
        v = [r["delta_pct"] for r in _sub(rows, iter=it, core="quantum")]
        data.append(v); labels.append(f"{it.upper()}\nn={len(v)}")
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", lw=2))
    for p, it in zip(bp["boxes"], ("v3b", "v4", "v5")):
        p.set_facecolor(ITER_COLOR[it]); p.set_alpha(0.4)
    for i, (it, v) in enumerate(zip(("v3b", "v4", "v5"), data), 1):
        ax.scatter(np.random.default_rng(i).normal(i, 0.05, len(v)), v, s=12,
                   color=ITER_COLOR[it], edgecolor="black", lw=0.25, zorder=3)
    ax.axhline(0, color="crimson", ls="--", lw=1, label="stock BBR-v3")
    ax.set_ylabel("throughput Δ vs stock (%)")
    ax.set_title("QA2C brain iterations — throughput Δ vs stock\n"
                 "(quantum; raw learned policy re-scored on the current simulator; "
                 "seeds × 10 holdout × 4 conditions)", fontsize=10)
    ax.legend(loc="lower left", fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGDIR / "tier1_iteration_delta_boxplot.png", dpi=150)
    plt.close(fig); print("figures/tier1_iteration_delta_boxplot.png")


def fig_tradeoff(rows):
    fig, ax = plt.subplots(figsize=(8, 6))
    for core in ("classical", "quantum"):
        pts = {}
        for r in _sub(rows, iter="v5", core=core):
            pts.setdefault((r["seed"], r["location"], r["direction"]), []).append(r)
        xs, ys = [], []
        for (seed, loc, d), rs in pts.items():
            base_rtx = np.mean([s["rtx_per_s"] for s in _sub(rows, core="stock",
                                                             location=loc, direction=d)])
            xs.append(np.mean([x["rtx_per_s"] for x in rs]) - base_rtx)
            ys.append(np.mean([x["delta_pct"] for x in rs]))
        ax.scatter(xs, ys, s=55, color=CORE_COLOR[core], edgecolor="black", lw=0.5,
                   alpha=0.8, label=core)
    ax.axhline(0, color="crimson", ls="--", lw=1); ax.axvline(0, color="grey", ls=":", lw=1)
    ax.set_xlabel("retransmits Δ vs stock (per s)")
    ax.set_ylabel("throughput Δ vs stock (%)")
    ax.set_title("Tier-1 v5 — throughput gain vs retransmit cost "
                 "(one point per train seed × condition)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGDIR / "tier1_v5_throughput_vs_retransmit.png", dpi=150)
    plt.close(fig); print("figures/tier1_v5_throughput_vs_retransmit.png")


def main():
    rows = reeval()
    fig_by_condition(rows)
    fig_cdf(rows)
    fig_perseed_strip(rows)
    fig_iterations(rows)
    fig_tradeoff(rows)
    print("\nall figures in figures/")


if __name__ == "__main__":
    main()
