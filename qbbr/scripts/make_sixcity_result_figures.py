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
from qbbr.agents.classical.native_a2c import NativeMLPA2CAgent

FIGDIR = Path("figures"); FIGDIR.mkdir(exist_ok=True)
CITIES = ["London", "Sydney", "Tokyo", "SaoPaulo", "Ohio", "Mumbai"]
IN_DIST = {"London", "Sydney"}
DIRECTIONS = ["downlink", "uplink"]
MODELS = ["BBR-v3 (stock)", "Classical A2C", "QA2C (quantum)"]
MCOLOR = {"BBR-v3 (stock)": "#4c78a8", "Classical A2C": "#f58518", "QA2C (quantum)": "#54a24b"}
TS_TRAIN_SEED = 0
TS_HOLDOUT_SEED = 1000
TS_SECONDS = 60.0

CONFIG = yaml.safe_load(
    Path("qbbr/configs/tier1_native_qa2c_successor_protocol.yaml").read_text())
CALIB = load_calibration(PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json")
SELECTOR = _selector(CONFIG)
HOLDOUT = CONFIG["evaluation"]["holdout_seeds"]
TRAIN_SEEDS = CONFIG["training"]["training_seeds"]


def _load_quantum(path):
    st = torch.load(path, map_location="cpu", weights_only=False)["agent_state"]
    ag = NativeQA2CAgent(observation_dim=int(st["observation_dim"]),
                         native_action_count=int(st["native_action_count"]),
                         n_layers=2, entropy_coef=float(st["entropy_coef"]), selector=SELECTOR)
    ag.load_training_state_dict(st)
    return ag


def _load_classical(path):
    st = torch.load(path, map_location="cpu", weights_only=False)["agent_state"]
    ag = NativeMLPA2CAgent(observation_dim=int(st["observation_dim"]),
                           native_action_count=int(st["native_action_count"]),
                           actor_hidden_dim=int(st["actor_hidden_dim"]),
                           critic_hidden_dim=int(st["critic_hidden_dim"]),
                           entropy_coef=float(st["entropy_coef"]), selector=SELECTOR)
    ag.load_training_state_dict(st)
    return ag


def _rollout(agent, city, direction, hseed, stock_action=None, want_series=False):
    env = _env(CALIB, city, direction, CONFIG)
    state, done = env.reset(seed=hseed), False
    tp, rtx, rtt, vob = [], [], [], []
    ts = {"t": [], "tput_mbps": [], "v_over_bdp": [], "rtt_ms": []}
    while not done:
        a = stock_action if stock_action is not None else agent.act(
            state, env.allowed_action_indices(), deterministic=True, deployment=True)[0]
        state, _r, done, info = env.step(a)
        inst = info["delivered_bytes"] * 8.0 / info["t_dec_s"] / 1e6
        tp.append(inst)
        rtx.append(info["retransmits"] / info["t_dec_s"])
        rtt.append(info["rtt_ms"])
        vob.append(float(env._history[-1]["v_over_bdp"]))
        if want_series and info["t_start"] <= TS_SECONDS:
            ts["t"].append(info["t_start"])
            ts["tput_mbps"].append(inst)
            ts["v_over_bdp"].append(float(env._history[-1]["v_over_bdp"]))
            ts["rtt_ms"].append(info["rtt_ms"])
    ep = {"tput_mbps": float(np.mean(tp)), "rtx_per_s": float(np.mean(rtx)),
          "rtt_ms": float(np.mean(rtt)), "v_over_bdp": float(np.mean(vob))}
    return (ep, ts) if want_series else (ep, None)


def collect():
    rows_path = Path("outputs/sixcity_result_rows.json")
    series_path = Path("outputs/sixcity_timeseries.json")
    rows, series = [], []
    done_pairs: set[tuple[str, str]] = set()
    if rows_path.exists():
        rows = json.loads(rows_path.read_text())
        series = json.loads(series_path.read_text()) if series_path.exists() else []
        done_pairs = {(r["city"], r["direction"]) for r in rows}
        print(f"[resume] {len(rows)} rows on disk; skipping {sorted(done_pairs)}", flush=True)
    for city in CITIES:
        for direction in DIRECTIONS:
            if (city, direction) in done_pairs:
                continue
            # stock
            stock_ep = {}
            for hs in HOLDOUT:
                ep, ts = _rollout(None, city, direction, hs, stock_action=2,
                                  want_series=(hs == TS_HOLDOUT_SEED))
                stock_ep[hs] = ep["tput_mbps"]
                rows.append(dict(model="BBR-v3 (stock)", city=city, direction=direction,
                                 seed=-1, holdout_seed=hs, delta_pct=0.0, **ep))
                if ts is not None:
                    series.append(dict(model="BBR-v3 (stock)", city=city, direction=direction, **ts))
            # RL arms
            for model, core, loader in (("Classical A2C", "classical", _load_classical),
                                        ("QA2C (quantum)", "quantum", _load_quantum)):
                for seed in TRAIN_SEEDS:
                    hits = glob.glob(f"outputs/tier1_v5_checkpoints/{core}/{city}/{direction}/seed{seed}.pt")
                    # checkpoints only exist for the trained cities; reuse the
                    # London checkpoint set for transfer cities (same policy).
                    if not hits:
                        hits = glob.glob(f"outputs/tier1_v5_checkpoints/{core}/London/{direction}/seed{seed}.pt")
                    ag = loader(hits[0])
                    for hs in HOLDOUT:
                        want = (seed == TS_TRAIN_SEED and hs == TS_HOLDOUT_SEED)
                        ep, ts = _rollout(ag, city, direction, hs, want_series=want)
                        rows.append(dict(model=model, city=city, direction=direction,
                                         seed=seed, holdout_seed=hs,
                                         delta_pct=100.0 * (ep["tput_mbps"] / stock_ep[hs] - 1.0), **ep))
                        if ts is not None:
                            series.append(dict(model=model, city=city, direction=direction, **ts))
                    print(f"[collect] {model:16s} {city:9s} {direction:8s} seed{seed}", flush=True)
            Path("outputs/sixcity_result_rows.json").write_text(json.dumps(rows, indent=2))
            Path("outputs/sixcity_timeseries.json").write_text(json.dumps(series, indent=2))
            print(f"  -- checkpointed after {city}/{direction} ({len(rows)} rows) --", flush=True)
    return rows, series


# --------------------------------------------------------------------------
def _sub(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def _grouped_box(ax, rows, direction, metric, ylabel, clip_pct=None):
    step = 3.0
    positions, ticks = [], []
    for ci, city in enumerate(CITIES):
        for mi, model in enumerate(MODELS):
            vals = [r[metric] for r in _sub(rows, city=city, direction=direction, model=model)]
            pos = ci * step + (mi - 1) * 0.72
            bp = ax.boxplot([vals], positions=[pos], widths=0.6, patch_artist=True,
                            medianprops=dict(color="black", lw=1.4), showfliers=False)
            bp["boxes"][0].set_facecolor(MCOLOR[model]); bp["boxes"][0].set_alpha(0.6)
            ax.scatter(np.random.default_rng(ci * 3 + mi).normal(pos, 0.06, len(vals)), vals,
                       s=6, color=MCOLOR[model], edgecolor="none", alpha=0.35, zorder=3)
        positions.append(ci * step); ticks.append(city + ("" if city in IN_DIST else "\n(transfer)"))
    ax.set_xticks(positions); ax.set_xticklabels(ticks, fontsize=8)
    ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(-1.2, (len(CITIES) - 1) * step + 1.2)
    if clip_pct is not None:
        hi = np.percentile([r[metric] for r in _sub(rows, direction=direction)], clip_pct)
        ax.set_ylim(top=hi * 1.15)


def fig_boxplot_grid(rows, direction):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    _grouped_box(axes[0, 0], rows, direction, "tput_mbps", "throughput (Mbps)")
    axes[0, 0].set_title("Throughput")
    _grouped_box(axes[0, 1], rows, direction, "rtx_per_s", "retransmits (per s)", clip_pct=98)
    axes[0, 1].set_title("Retransmissions  (y clipped at p98)")
    _grouped_box(axes[1, 0], rows, direction, "rtt_ms", "RTT (ms)")
    axes[1, 0].set_title("RTT")
    _grouped_box(axes[1, 1], rows, direction, "v_over_bdp", "inflight / BDP")
    axes[1, 1].set_title("Inflight / BDP")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=MCOLOR[m], alpha=0.6) for m in MODELS]
    fig.legend(handles, MODELS, loc="lower center", ncol=3, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"Six-city comparison — dedicated {direction} over Starlink   "
                 f"(v5 deployed policy;  box = 5 train seeds × 10 holdout seeds,  stock = 10 holdout seeds;  "
                 f"Classical A2C coincides with stock)", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    out = FIGDIR / f"sixcity_boxplot_{direction}.png"
    fig.savefig(out, dpi=150); plt.close(fig); print(out)


def fig_throughput_timeseries(series, direction):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    style = {"BBR-v3 (stock)": dict(lw=2.6, ls="-", alpha=0.9),
             "Classical A2C": dict(lw=1.2, ls=(0, (4, 3)), alpha=0.95),
             "QA2C (quantum)": dict(lw=1.4, ls="-", alpha=0.95)}
    for ax, city in zip(axes.flat, CITIES):
        for model in MODELS:
            s = _sub(series, city=city, direction=direction, model=model)
            if not s:
                continue
            ax.plot(s[0]["t"], s[0]["tput_mbps"], color=MCOLOR[model], label=model, **style[model])
        ax.set_title(city + ("" if city in IN_DIST else "  (transfer)"), fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylabel("Mbps")
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Throughput time series — dedicated {direction}, first {TS_SECONDS:.0f}s "
                 f"(train seed {TS_TRAIN_SEED}, holdout seed {TS_HOLDOUT_SEED};  "
                 f"stock line thick, Classical A2C dashed on top — they coincide)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGDIR / f"sixcity_throughput_timeseries_{direction}.png"
    fig.savefig(out, dpi=150); plt.close(fig); print(out)


def fig_inflight_bdp(series, direction):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for ax, city in zip(axes.flat, CITIES):
        for model in ("BBR-v3 (stock)", "QA2C (quantum)"):
            s = _sub(series, city=city, direction=direction, model=model)
            if not s:
                continue
            s = s[0]
            ax.plot(s["t"], s["v_over_bdp"], lw=1.1, color=MCOLOR[model], label=model, alpha=0.9)
        ax.axhline(1.0, color="grey", ls="--", lw=0.8)
        ax.axhline(2.0, color="grey", ls=":", lw=0.8)
        ax.set_title(city + ("" if city in IN_DIST else "  (transfer)"), fontsize=10)
        ax.grid(alpha=0.3); ax.set_ylabel("inflight / BDP")
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Inflight / BDP time series — BBR-v3 vs QA2C, dedicated {direction}, "
                 f"first {TS_SECONDS:.0f}s (dashed 1×BDP, dotted 2×BDP)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGDIR / f"sixcity_inflight_bdp_{direction}.png"
    fig.savefig(out, dpi=150); plt.close(fig); print(out)


def fig_delta_cdf(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (grp, title) in zip(axes, [(IN_DIST, "in-distribution (London, Sydney)"),
                                       (set(CITIES) - IN_DIST, "zero-shot transfer (Tokyo, SaoPaulo, Ohio, Mumbai)")]):
        for model in ("QA2C (quantum)", "Classical A2C"):
            v = np.sort([r["delta_pct"] for r in rows if r["model"] == model and r["city"] in grp])
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", lw=2,
                    color=MCOLOR[model], label=f"{model}  (n={len(v)})")
        ax.axvline(0, color="crimson", ls="--", lw=1, label="stock BBR-v3")
        ax.set_title(title, fontsize=10); ax.grid(alpha=0.3)
        ax.set_xlabel("throughput Δ vs stock (%)")
    axes[0].set_ylabel("cumulative fraction"); axes[0].legend(fontsize=8, loc="center right")
    fig.suptitle("CDF of throughput Δ vs stock — v5 deployed policy, per (train seed × holdout seed × direction)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGDIR / "sixcity_delta_cdf.png"
    fig.savefig(out, dpi=150); plt.close(fig); print(out)


def main():
    rows, series = collect()
    for direction in DIRECTIONS:
        fig_boxplot_grid(rows, direction)
        fig_throughput_timeseries(series, direction)
        fig_inflight_bdp(series, direction)
    fig_delta_cdf(rows)
    print("\nall six-city figures in figures/")


if __name__ == "__main__":
    main()
