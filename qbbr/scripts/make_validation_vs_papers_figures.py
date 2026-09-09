"""Validation figures: our calibrated 'stock BBR-v3' fluid model vs BBR-v3 as
reported in the Deakin Starlink measurement papers, plus the queue-buildup
time series analogous to those papers' Fig. 12 / Fig. 13.

Papers:
  desilva2026tmc  -- "Unveiling TCP BBR Dominance in Starlink Internet:
                      Experimental Insights and Analysis"
  desilva2026icoin -- "Understanding BBR-v3 Dynamics over Starlink"
Both benchmark BBR-v3 (their footnote: "BBR" == "BBR-v3") on the SAME six
cities / iperf3 / 300 s methodology this codebase's raw traces come from.

Reference values below are tagged:
  T = median stated in the paper text
  F = approximate value read off the paper's box-plot figure (Fig 5 / Fig 8)
The F values are eyeballed and are shown only as orientation, not ground truth.

Non-competing (single-flow) only -- FluidSimEnv has no multi-flow mode, so the
papers' competitive-stream figures (Fig 6/7/9/10/13) are out of scope here.

Writes to figures/:
  validation_stock_vs_paper_downlink.png
  queue_buildup_downlink.png
  queue_buildup_uplink.png
and outputs/validation_vs_papers.md
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
from qbbr.scripts.run_native_qa2c_successor import PACKAGE_ROOT, _env, _selector
from qbbr.agents.quantum.native_qa2c import NativeQA2CAgent

FIGDIR = Path("figures"); FIGDIR.mkdir(exist_ok=True)
CITIES = ["Tokyo", "SaoPaulo", "Ohio", "London", "Mumbai", "Sydney"]
HOLDOUT_TS = 1000
CONFIG = yaml.safe_load(
    Path("qbbr/configs/tier1_native_qa2c_successor_protocol.yaml").read_text())
CALIB = load_calibration(PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json")
SELECTOR = _selector(CONFIG)

# ---- paper BBR-v3 reference (T = text-stated median, F = read off figure) ----
PAPER_DL = {
    "throughput_mbps": {"Sydney": (236.0, "T"), "London": (160.0, "F"), "Mumbai": (200.0, "F"),
                        "Tokyo": (175.0, "F"), "Ohio": (175.0, "F"), "SaoPaulo": (110.0, "F")},
    "rtt_ms": {"Sydney": (47.0, "T"), "SaoPaulo": (355.0, "T"), "London": (255.0, "F"),
               "Ohio": (235.0, "F"), "Tokyo": (165.0, "F"), "Mumbai": (175.0, "F")},
    # Fig 5(b) retransmission COUNT over the 300 s run; /300 to compare with our per-s rate.
    "retransmits_per_s": {"Tokyo": (500.0 / 300, "F"), "Mumbai": (200.0 / 300, "F"),
                          "SaoPaulo": (300.0 / 300, "F"), "London": (120.0 / 300, "F"),
                          "Ohio": (120.0 / 300, "F"), "Sydney": (80.0 / 300, "F")},
}
PAPER_NOTE = ("desilva2026tmc/icoin, non-competing download.  T = median stated in text;  "
              "F ≈ read off Fig 5/8 box-plots (orientation only).  Retransmit ref = paper count/300 s.")


def _load_qa2c(path):
    st = torch.load(path, map_location="cpu", weights_only=False)["agent_state"]
    ag = NativeQA2CAgent(observation_dim=int(st["observation_dim"]),
                         native_action_count=int(st["native_action_count"]),
                         n_layers=2, entropy_coef=float(st["entropy_coef"]), selector=SELECTOR)
    ag.load_training_state_dict(st)
    return ag


def _trace(agent, city, direction, stock=False):
    env = _env(CALIB, city, direction, CONFIG)
    state, done = env.reset(seed=HOLDOUT_TS), False
    t, q, vob, tput = [], [], [], []
    while not done:
        a = 2 if stock else agent.act(state, env.allowed_action_indices(),
                                      deterministic=True, deployment=True)[0]
        state, _r, done, info = env.step(a)
        row = env._history[-1]
        t.append(info["t_start"]); q.append(float(row["q_packets"]))
        vob.append(float(row["v_over_bdp"]))
        tput.append(info["delivered_bytes"] * 8.0 / info["t_dec_s"] / 1e6)
    return {"t": t, "q_packets": q, "v_over_bdp": vob, "tput_mbps": tput}


def collect_traces():
    cache = Path("outputs/validation_traces.json")
    if cache.exists():
        print("[resume] loading cached traces from outputs/validation_traces.json", flush=True)
        return json.loads(cache.read_text())
    out = {}
    qa2c = {d: _load_qa2c(glob.glob(f"outputs/tier1_v5_checkpoints/quantum/London/{d}/seed0.pt")[0])
            for d in ("downlink", "uplink")}
    for direction in ("downlink", "uplink"):
        for city in CITIES:
            out[f"{city}/{direction}/stock"] = _trace(None, city, direction, stock=True)
            out[f"{city}/{direction}/qa2c"] = _trace(qa2c[direction], city, direction)
            print(f"[trace] {city:9s} {direction}", flush=True)
    Path("outputs/validation_traces.json").write_text(json.dumps(out))
    return out


# --------------------------------------------------------------------------
def fig_stock_vs_paper(rows):
    metrics = [("tput_mbps", "throughput_mbps", "throughput (Mbps)", "Throughput"),
               ("rtt_ms", "rtt_ms", "RTT (ms)", "RTT"),
               ("rtx_per_s", "retransmits_per_s", "retransmits (per s)", "Retransmissions")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, (rk, pk, ylab, title) in zip(axes, metrics):
        data = [[r[rk] for r in rows if r["model"] == "BBR-v3 (stock)"
                 and r["city"] == c and r["direction"] == "downlink"] for c in CITIES]
        bp = ax.boxplot(data, positions=range(len(CITIES)), widths=0.5, patch_artist=True,
                        medianprops=dict(color="black", lw=1.6), showfliers=False)
        for b in bp["boxes"]:
            b.set_facecolor("#4c78a8"); b.set_alpha(0.5)
        for i, c in enumerate(CITIES):
            if c in PAPER_DL[pk]:
                val, tag = PAPER_DL[pk][c]
                ax.scatter([i], [val], marker="D" if tag == "T" else "o", s=80 if tag == "T" else 55,
                           color="crimson", zorder=5, edgecolor="black", lw=0.6,
                           facecolor="crimson" if tag == "T" else "white",
                           label=None)
        ax.set_xticks(range(len(CITIES))); ax.set_xticklabels(CITIES, fontsize=9, rotation=15)
        ax.set_ylabel(ylab); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    from matplotlib.lines import Line2D
    leg = [plt.Rectangle((0, 0), 1, 1, fc="#4c78a8", alpha=0.5),
           Line2D([0], [0], marker="D", color="w", markerfacecolor="crimson", markeredgecolor="black", markersize=9),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="black", markersize=8)]
    fig.legend(leg, ["our calibrated stock BBR-v3 (fluid model, n=10 holdout seeds)",
                     "paper BBR-v3  — median stated in text (T)",
                     "paper BBR-v3  — ≈ read off Fig 5/8 (F)"],
               loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Validation — our stock BBR-v3 vs BBR-v3 as reported in the Deakin Starlink papers\n"
                 + PAPER_NOTE, fontsize=10)
    fig.tight_layout(rect=[0, 0.07, 1, 0.90])
    fig.savefig(FIGDIR / "validation_stock_vs_paper_downlink.png", dpi=150)
    plt.close(fig); print("figures/validation_stock_vs_paper_downlink.png")


_WARMUP_S = 15.0  # skip the STARTUP transient so the steady-state sawtooth is legible


def fig_queue_buildup(traces, direction):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for ax, city in zip(axes.flat, CITIES):
        s = traces[f"{city}/{direction}/stock"]; q = traces[f"{city}/{direction}/qa2c"]
        def post(tr):
            t = np.asarray(tr["t"]); qq = np.asarray(tr["q_packets"]); m = t >= _WARMUP_S
            return t[m], qq[m]
        ts, qs = post(s); tq, qq = post(q)
        ax.plot(ts, qs, lw=1.8, color="#4c78a8", label="BBR-v3 (stock)")
        ax.plot(tq, qq, lw=1.0, color="#54a24b", label="QA2C (quantum)", alpha=0.9)
        ax.axhline(50, color="crimson", ls="--", lw=0.9)
        ax.axhline(40, color="grey", ls=":", lw=0.9)
        cap = max(np.percentile(qq, 99) * 1.15, 60) if len(qq) else 60
        ax.set_ylim(-cap * 0.03, cap)
        peak = max(np.max(q["q_packets"]), 1.0)
        ax.text(0.97, 0.92, f"QA2C STARTUP peak ≈ {peak:,.0f} pkts", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="#54a24b")
        ax.set_title(city, fontsize=10); ax.grid(alpha=0.3); ax.set_ylabel("queue (packets)")
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Queue buildup over time — non-competing {direction} BBR streams, six cities  "
                 f"(holdout seed {HOLDOUT_TS}, t ≥ {_WARMUP_S:.0f}s;  dashed = paper's 50-pkt cap, dotted = Q≥40)\n"
                 f"paper desilva2026tmc Fig 12: BBR-v3 at the 50-pkt cap 70–77% of the time (single stream).  "
                 f"Here stock ≈ 0; QA2C runs a large standing queue to buy its throughput.",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIGDIR / f"queue_buildup_{direction}.png", dpi=150)
    plt.close(fig); print(f"figures/queue_buildup_{direction}.png")


def write_md(rows, traces):
    def med(model, city, d, k):
        return float(np.median([r[k] for r in rows if r["model"] == model
                                and r["city"] == city and r["direction"] == d]))
    lines = ["# Validation — our stock BBR-v3 fluid model vs the Deakin Starlink papers", "",
             "Non-competing (single-flow) download. Our values: median over 10 holdout seeds.",
             "Paper values: T = median stated in text, F ≈ read off Fig 5/8 box-plots.", "",
             "| city | our stock tput | paper BBR-v3 tput | our stock RTT | paper RTT | our mean q (pkts) | paper queue |",
             "|---|---|---|---|---|---|---|"]
    for c in CITIES:
        pt = PAPER_DL["throughput_mbps"].get(c); pr = PAPER_DL["rtt_ms"].get(c)
        qmean = float(np.mean(traces[f"{c}/downlink/stock"]["q_packets"]))
        lines.append(f"| {c} | {med('BBR-v3 (stock)', c, 'downlink', 'tput_mbps'):.0f} Mbps "
                     f"| {pt[0]:.0f} ({pt[1]}) | {med('BBR-v3 (stock)', c, 'downlink', 'rtt_ms'):.0f} ms "
                     f"| {pr[0]:.0f} ({pr[1]}) ms | {qmean:.1f} | ~50 (cap) 70–77% of time |")
    lines += ["",
              "## Reading",
              "- **Throughput / RTT**: our calibrated stock BBR-v3 is in the papers' ballpark but consistently "
              "conservative on downlink (e.g. Sydney 188 vs 236 Mbps).",
              "- **Queue**: clear mismatch. Our stock BBR-v3 barely queues (mean well under the 40/50-packet "
              "marks), whereas desilva2026tmc Fig 12 has BBR-v3 pinned at the 50-packet cap 70–77% of the time. "
              "Root cause: the `utilization_fraction` floor (0.05, `calibration.py:45`) throttles delivery so the "
              "pipe never fills — most visible on uplink, where 4/6 cities calibrate to ~2 Mbps.",
              "- **Consequence**: throughput/RTT improvement claims for QA2C are defensible; queueing / "
              "buffer-occupancy claims are NOT until the stock baseline is re-calibrated to match the papers' "
              "BBR-v3 queue behaviour.",
              "- **Out of scope here**: every competitive / parallel-stream figure (desilva2026tmc Fig 6/7/9/10/13) "
              "— `FluidSimEnv` is single-flow only.", ""]
    Path("outputs/validation_vs_papers.md").write_text("\n".join(lines))
    print("outputs/validation_vs_papers.md")


def main():
    rows = json.load(open("outputs/sixcity_result_rows.json"))
    traces = collect_traces()
    fig_stock_vs_paper(rows)
    fig_queue_buildup(traces, "downlink")
    fig_queue_buildup(traces, "uplink")
    write_md(rows, traces)
    print("\ndone")


if __name__ == "__main__":
    main()
