"""Compare our native-action QRL arms against the CCAs actually MEASURED on
Starlink in qbbr/data/raw.

Measured arms (bbr, bbr2, cubic, ...) are per-run statistics from the real
iperf3 sequential runs. Simulated arms (stock / A2C / QA2C) come from the
trace-replay condition, i.e. the fluid proxy driven by capacity forcing taken
from those same real traces -- validated by the stock-vs-real fidelity printed
in the figure, so the two families sit on a comparable scale.

They are NOT the same kind of evidence and the figure says so: measured arms
are drawn solid, simulated arms hatched. The only *controlled* contrast here
is agent-vs-stock inside the simulator; a simulated arm sitting near a
measured one is context, not a head-to-head win.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from qbbr.data.catalog import build_catalog, iter_file_records
from qbbr.data.loader import load_trace

PKG = Path(__file__).resolve().parent.parent
ROOT = PKG.parent
CITIES = ["Tokyo", "SaoPaulo", "Ohio", "London", "Mumbai", "Sydney"]
PANELS = [("throughput", "Throughput", "Mbps"), ("retx", "Retransmissions", "per s"),
          ("rtt", "RTT", "ms"), ("rttvar", "RTT variance", "ms")]


def measured_arms(catalog, city, direction, ccas) -> dict:
    out = {}
    for cca in ccas:
        sub = catalog[(catalog.category == "sequential") & (catalog.location == city)
                      & (catalog.direction == direction) & (catalog.cca == cca)]
        rows = {"throughput": [], "retx": [], "rtt": [], "rttvar": []}
        for rec in iter_file_records(sub):
            iv = load_trace(rec).intervals
            iv = iv[(iv["omitted"] != True) & iv["bits_per_second"].notna()]  # noqa: E712
            if not len(iv):
                continue
            secs = iv["seconds"].replace(0, np.nan)
            rtt = iv["rtt_ms"].dropna().to_numpy(dtype=float)
            rows["throughput"].append(float(iv["bits_per_second"].mean() / 1e6))
            rows["retx"].append(float((iv["retransmits"] / secs).mean()))
            rows["rtt"].append(float(np.median(rtt)) if rtt.size else np.nan)
            rows["rttvar"].append(float(np.std(rtt)) if rtt.size else np.nan)
        if rows["throughput"]:
            out[cca] = rows
    return out


def simulated_arms(replay_dir: Path, city, direction) -> dict:
    def collect(core, key):
        f = replay_dir / f"replay_{city}_{direction}_{core}.json"
        if not f.exists():
            return None, None
        data = json.loads(f.read_text())
        rows = {"throughput": [], "retx": [], "rtt": [], "rttvar": []}
        seen_stock = set()
        fid = []
        for r in data.get("rows", []):
            src = r[key]
            if key == "stock_sim":
                if r["run"] in seen_stock:
                    continue
                seen_stock.add(r["run"])
                fid.append(r["replay_fidelity_stock_over_real"])
            rows["throughput"].append(src["throughput_mbps_mean"])
            rows["retx"].append(src["retransmits_per_s_mean"])
            rows["rtt"].append(src.get("rtt_median_ms", src.get("rtt_p90_ms")))
            rows["rttvar"].append(src.get("rtt_std_ms", np.nan))
        return (rows if rows["throughput"] else None), (float(np.median(fid)) if fid else None)

    stock, fid = collect("quantum", "stock_sim")
    qa2c, _ = collect("quantum", "agent_sim")
    a2c, _ = collect("classical", "agent_sim")
    arms = {}
    for name, rows in (("stock-sim", stock), ("A2C-sim", a2c), ("QA2C-sim", qa2c)):
        if rows:
            arms[name] = rows
    return arms, fid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=PKG / "data" / "raw")
    ap.add_argument("--replay-dir", type=Path, default=ROOT / "outputs" / "replay6")
    ap.add_argument("--direction", default="downlink")
    ap.add_argument("--measured-ccas", nargs="+", default=["bbr", "bbr2", "cubic"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / "figures" / f"cca_comparison_{args.direction}.png"

    catalog = build_catalog(args.dataset_root)
    measured_names = [f"{c} (measured)" for c in args.measured_ccas]
    sim_names = ["stock-sim", "A2C-sim", "QA2C-sim"]
    arm_names = measured_names + sim_names
    colours = (["#4C72B0", "#DD8452", "#55A868"][:len(measured_names)]
               + ["#8172B3", "#937860", "#C44E52"])

    data = {}   # city -> arm -> metric -> values
    fidelity = {}
    for city in CITIES:
        m = measured_arms(catalog, city, args.direction, args.measured_ccas)
        s, fid = simulated_arms(args.replay_dir, city, args.direction)
        fidelity[city] = fid
        data[city] = {f"{k} (measured)": v for k, v in m.items()} | s

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    width = 0.8 / len(arm_names)
    for ax, (key, title, unit) in zip(axes.ravel(), PANELS):
        for j, arm in enumerate(arm_names):
            vals, pos = [], []
            for i, city in enumerate(CITIES):
                series = data[city].get(arm, {}).get(key)
                series = [v for v in (series or []) if v is not None and np.isfinite(v)]
                if series:
                    vals.append(series)
                    pos.append(i + (j - (len(arm_names) - 1) / 2) * width)
            if not vals:
                continue
            bp = ax.boxplot(vals, positions=pos, widths=width * 0.85, patch_artist=True,
                            showfliers=False, medianprops=dict(color="black", lw=1.2))
            for patch in bp["boxes"]:
                patch.set_facecolor(colours[j])
                patch.set_alpha(0.85)
                if arm in sim_names:
                    patch.set_hatch("///")
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_xticks(range(len(CITIES)))
        ax.set_xticklabels(CITIES, rotation=20)
        ax.grid(alpha=0.25, axis="y")

    handles = [Patch(facecolor=colours[j], alpha=0.85,
                     hatch="///" if arm in sim_names else None, label=arm)
               for j, arm in enumerate(arm_names)]
    fig.legend(handles=handles, loc="upper center", ncol=len(arm_names), frameon=False,
               bbox_to_anchor=(0.5, 0.985))
    fid_txt = "  ".join(f"{c}:{fidelity[c]:.2f}x" for c in CITIES if fidelity.get(c))
    fig.suptitle(
        f"{args.direction.capitalize()} over Starlink — measured CCAs (solid) vs simulated "
        f"native-action QRL (hatched, trace-replay conditioned)\n"
        f"Simulated arms are a fluid proxy driven by capacity forcing from the same real traces. "
        f"Stock-vs-real replay fidelity: {fid_txt}\n"
        f"The only controlled contrast is agent-vs-stock inside the simulator; proximity to a "
        f"measured arm is context, not a head-to-head result.",
        fontsize=9, y=0.06)
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    csv = out.with_suffix(".csv")
    with csv.open("w") as fh:
        fh.write("city,direction,arm,metric,value\n")
        for city in CITIES:
            for arm, metrics in data[city].items():
                for key, vals in metrics.items():
                    for v in vals:
                        if v is not None and np.isfinite(v):
                            fh.write(f"{city},{args.direction},{arm},{key},{v}\n")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
