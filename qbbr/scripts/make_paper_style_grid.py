"""Paper-style 6-metric x 6-city box-plot grid (Throughput, Retransmissions,
Congestion Window, Receiver Advertised Window, RTT, RTT Variance), in the
layout of desilva2026tmc Fig. 5 / Fig. 8.

Two data sources:
  * measured CCAs  -- parsed straight from the iperf3 sequential (dedicated,
    single-flow) logs in qbbr/data/raw: BBR-v3 (`bbr`), BBRv1, BBRv2, Cubic,
    Hybla, Vegas. All six metrics (snd_cwnd, snd_wnd are in the logs).
  * our agents     -- stock BBR-v3 fluid model and the v5 QA2C brain, rolled
    out on FluidSimEnv. Only four metrics: the fluid model has NO congestion
    window and NO receiver window, so those two panels stay measured-only.

Writes figures/paper_style_grid_downlink.png, figures/paper_style_grid_uplink.png
and the underlying outputs/paper_style_grid_*.json.
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
MEASURED = [("bbr", "BBR-v3"), ("bbr1", "BBRv1"), ("bbr2", "BBRv2"),
            ("cubic", "Cubic"), ("hybla", "Hybla"), ("vegas", "Vegas")]
AGENTS = [("stock", "stock BBR-v3 (fluid)"), ("qa2c", "QA2C (v5)")]
COLORS = {"BBR-v3": "#4c78a8", "BBRv1": "#72b7b2", "BBRv2": "#54a24b", "Cubic": "#eeca3b",
          "Hybla": "#b279a2", "Vegas": "#ff9da6",
          "stock BBR-v3 (fluid)": "#9d9d9d", "QA2C (v5)": "#e45756"}
PANELS = [("throughput_mbps", "throughput (Mbps)", "Throughput", True),
          ("retransmits", "count", "Retransmissions", True),
          ("cwnd_mb", "MB", "Congestion Window", False),
          ("rwnd_mb", "MB", "Receiver Advertised Window", False),
          ("rtt_ms", "ms", "RTT", True),
          ("rttvar_ms", "ms", "RTT Variance", True)]

CONFIG = yaml.safe_load(Path("qbbr/configs/tier1_native_qa2c_successor_protocol.yaml").read_text())
CALIB = load_calibration(PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json")
SELECTOR = _selector(CONFIG)
HOLDOUT = CONFIG["evaluation"]["holdout_seeds"]
SEEDS = CONFIG["training"]["training_seeds"]


# ----- measured CCAs from raw iperf3 logs -----
def _read_json_tolerant(path):
    raw = Path(path).read_text()
    return json.JSONDecoder().raw_decode(raw.lstrip())[0]


def _seq_glob(cca, city, direction):
    if direction == "downlink":
        return glob.glob(f"qbbr/data/raw/downlink-sequential-logs/{city}/**/{cca}_{city}__REV_run*.json",
                         recursive=True)
    return glob.glob(f"qbbr/data/raw/uplink-sequential-logs/{city}/{cca}_{city}__FWD_run*.json")


def parse_measured():
    cache = Path("outputs/measured_cca_grid_metrics.json")
    if cache.exists():
        print("[resume] measured_cca_grid_metrics.json", flush=True)
        return json.loads(cache.read_text())
    rows = []
    for direction in ("downlink", "uplink"):
        for cca, label in MEASURED:
            for city in CITIES:
                for path in sorted(_seq_glob(cca, city, direction)):
                    try:
                        d = _read_json_tolerant(path)
                        ivs = [s for iv in d.get("intervals", []) if not iv.get("omitted", False)
                               for s in iv.get("streams", [])]
                        if not ivs:
                            continue
                        bps = np.array([s["bits_per_second"] for s in ivs], float)
                        cwnd = np.array([s.get("snd_cwnd", np.nan) for s in ivs], float)
                        wnd = np.array([s.get("snd_wnd") or np.nan for s in ivs], float)
                        rtt = np.array([s.get("rtt", np.nan) for s in ivs], float)
                        rttv = np.array([s.get("rttvar", np.nan) for s in ivs], float)
                        retr = d.get("end", {}).get("streams", [{}])[0].get("sender", {}).get("retransmits")
                        rows.append(dict(
                            source="measured", model=label, city=city, direction=direction,
                            throughput_mbps=float(np.median(bps) / 1e6),
                            retransmits=float(retr) if retr is not None else float("nan"),
                            cwnd_mb=float(np.nanmedian(cwnd) / 1e6),
                            rwnd_mb=float(np.nanmedian(wnd) / 1e6),
                            rtt_ms=float(np.nanmedian(rtt) / 1e3),
                            rttvar_ms=float(np.nanmedian(rttv) / 1e3),
                        ))
                    except Exception as exc:  # skip an unreadable run
                        print(f"  skip {path}: {exc}", flush=True)
            print(f"[measured] {direction} {label}: {sum(1 for r in rows if r['model']==label and r['direction']==direction)} runs", flush=True)
    cache.write_text(json.dumps(rows, indent=2))
    return rows


# ----- our agents on FluidSimEnv -----
def _load_qa2c(path):
    st = torch.load(path, map_location="cpu", weights_only=False)["agent_state"]
    ag = NativeQA2CAgent(observation_dim=int(st["observation_dim"]),
                         native_action_count=int(st["native_action_count"]),
                         n_layers=2, entropy_coef=float(st["entropy_coef"]), selector=SELECTOR)
    ag.load_training_state_dict(st)
    return ag


def _rollout(agent, city, direction, hseed, stock=False):
    env = _env(CALIB, city, direction, CONFIG)
    state, done = env.reset(seed=hseed), False
    tp, rtt, rtx = [], [], 0.0
    dur = 0.0
    while not done:
        a = 2 if stock else agent.act(state, env.allowed_action_indices(),
                                      deterministic=True, deployment=True)[0]
        state, _r, done, info = env.step(a)
        tp.append(info["delivered_bytes"] * 8.0 / info["t_dec_s"] / 1e6)
        rtt.append(info["rtt_ms"]); rtx += info["retransmits"]; dur += info["t_dec_s"]
    return dict(throughput_mbps=float(np.mean(tp)), retransmits=float(rtx),
                rtt_ms=float(np.mean(rtt)), rttvar_ms=float(np.std(rtt)))


def collect_agents():
    cache = Path("outputs/agent_grid_metrics.json")
    if cache.exists():
        print("[resume] agent_grid_metrics.json", flush=True)
        return json.loads(cache.read_text())
    rows = []
    qa2c = {d: {s: _load_qa2c(glob.glob(f"outputs/tier1_v5_checkpoints/quantum/London/{d}/seed{s}.pt")[0])
                for s in SEEDS} for d in ("downlink", "uplink")}
    for direction in ("downlink", "uplink"):
        for city in CITIES:
            for hs in HOLDOUT:
                rows.append(dict(source="agent", model="stock BBR-v3 (fluid)", city=city,
                                 direction=direction, **_rollout(None, city, direction, hs, stock=True)))
            for s in SEEDS:
                for hs in HOLDOUT:
                    rows.append(dict(source="agent", model="QA2C (v5)", city=city, direction=direction,
                                     **_rollout(qa2c[direction][s], city, direction, hs)))
            print(f"[agent] {direction} {city} done ({len(rows)} rows)", flush=True)
            cache.write_text(json.dumps(rows, indent=2))
    return rows


# ----- figure -----
def make_grid(rows, direction):
    groups = [lab for _, lab in MEASURED] + [lab for _, lab in AGENTS]
    fig, axes = plt.subplots(2, 3, figsize=(23, 12))
    for ax, (key, ylab, title, shared) in zip(axes.flat, PANELS):
        present = groups if shared else [lab for _, lab in MEASURED]
        step = len(present) + 1.5
        ticks = []
        for ci, city in enumerate(CITIES):
            for gi, g in enumerate(present):
                vals = [r[key] for r in rows if r["model"] == g and r["city"] == city
                        and r["direction"] == direction and np.isfinite(r.get(key, np.nan))]
                if not vals:
                    continue
                pos = ci * step + gi
                bp = ax.boxplot([vals], positions=[pos], widths=0.8, patch_artist=True,
                                medianprops=dict(color="black", lw=1.2), showfliers=False)
                bp["boxes"][0].set_facecolor(COLORS[g]); bp["boxes"][0].set_alpha(0.75)
            ticks.append(ci * step + (len(present) - 1) / 2)
        ax.set_xticks(ticks); ax.set_xticklabels(CITIES, fontsize=9)
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=12); ax.grid(axis="y", alpha=0.3)
        if not shared:
            ax.set_title(title + "  (measured CCAs only — fluid model has no cwnd/rwnd)", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[g], alpha=0.75) for g in groups]
    fig.legend(handles, groups, loc="lower center", ncol=len(groups), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.004))
    fig.suptitle(f"Dedicated {direction} over Starlink — six cities.  "
                 f"Measured CCAs from iperf3 sequential logs (qbbr/data/raw, 10 runs each);  "
                 f"stock BBR-v3 (fluid) & QA2C from FluidSimEnv (5 seeds × 10 holdout).", fontsize=12)
    fig.tight_layout(rect=[0, 0.045, 1, 0.96])
    out = FIGDIR / f"paper_style_grid_{direction}.png"
    fig.savefig(out, dpi=140); plt.close(fig); print(out)


def main():
    rows = parse_measured() + collect_agents()
    Path("outputs/paper_style_grid_rows.json").write_text(json.dumps(rows))
    for direction in ("downlink", "uplink"):
        make_grid(rows, direction)
    print("done")


if __name__ == "__main__":
    main()
