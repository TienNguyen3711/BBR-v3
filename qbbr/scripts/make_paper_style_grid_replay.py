from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PKG = Path(__file__).resolve().parent.parent
ROOT = PKG.parent
CITIES = ["Tokyo", "SaoPaulo", "Ohio", "London", "Mumbai", "Sydney"]
MEASURED = [("bbr", "BBR-v3"), ("bbr2", "BBRv2"), ("cubic", "Cubic"),
            ("hybla", "Hybla"), ("vegas", "Vegas")]
AGENTS = [("stock_sim", "stock (fluid, replayed)"), ("a2c", "A2C (replayed)"),
          ("qa2c", "QA2C (replayed)")]
COLORS = {"BBR-v3": "#4c78a8", "BBRv2": "#54a24b", "Cubic": "#eeca3b",
          "Hybla": "#b279a2", "Vegas": "#ff9da6",
          "stock (fluid, replayed)": "#9d9d9d", "A2C (replayed)": "#8172b3",
          "QA2C (replayed)": "#e45756"}
# key, unit, title, simulated arms have it?
PANELS = [("throughput_mbps", "Mbps", "Throughput", True),
          ("retransmits_per_s", "per s", "Retransmissions", True),
          ("cwnd_mb", "MB", "Congestion Window", False),
          ("rwnd_mb", "MB", "Receiver Advertised Window", False),
          ("rtt_ms", "ms", "RTT (median)", True),
          ("rttvar_ms", "ms", "RTT Variance (TCP rttvar)", False)]


def _read_json_tolerant(path):
    return json.JSONDecoder().raw_decode(Path(path).read_text().lstrip())[0]


def _seq_glob(cca, city, direction):
    if direction == "downlink":
        return glob.glob(f"{PKG}/data/raw/downlink-sequential-logs/{city}/**/{cca}_{city}__REV_run*.json",
                         recursive=True)
    return glob.glob(f"{PKG}/data/raw/uplink-sequential-logs/{city}/{cca}_{city}__FWD_run*.json")


def parse_measured(direction: str) -> list[dict]:
    rows = []
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
                    sender = d.get("end", {}).get("streams", [{}])[0].get("sender", {})
                    retr, secs = sender.get("retransmits"), sender.get("seconds")
                    rows.append(dict(
                        source="measured", model=label, city=city, direction=direction,
                        throughput_mbps=float(np.mean(bps) / 1e6),
                        retransmits_per_s=(float(retr) / float(secs)
                                           if retr is not None and secs else float("nan")),
                        cwnd_mb=float(np.nanmedian(cwnd) / 1e6),
                        rwnd_mb=float(np.nanmedian(wnd) / 1e6),
                        rtt_ms=float(np.nanmedian(rtt) / 1e3),
                        rttvar_ms=float(np.nanmedian(rttv) / 1e3),
                    ))
                except Exception as exc:
                    print(f"  skip {path}: {exc}", flush=True)
        n = sum(1 for r in rows if r["model"] == label)
        print(f"[measured] {direction} {label}: {n} runs", flush=True)
    return rows


def collect_replayed(replay_dir: Path, direction: str) -> tuple[list[dict], dict]:
    """Agent arms from the trace-replay rollouts, plus per-city fidelity."""
    rows, fidelity = [], {}
    for city in CITIES:
        seen_stock = set()
        for core, label in (("quantum", "QA2C (replayed)"), ("classical", "A2C (replayed)")):
            f = replay_dir / f"replay_{city}_{direction}_{core}.json"
            if not f.exists():
                print(f"  missing {f.name}", flush=True)
                continue
            for r in json.loads(f.read_text()).get("rows", []):
                agent = r["agent_sim"]
                rows.append(dict(source="agent", model=label, city=city, direction=direction,
                                 throughput_mbps=agent["throughput_mbps_mean"],
                                 retransmits_per_s=agent["retransmits_per_s_mean"],
                                 rtt_ms=agent.get("rtt_median_ms", np.nan),
                                 cwnd_mb=np.nan, rwnd_mb=np.nan, rttvar_ms=np.nan))
                if core == "quantum" and r["run"] not in seen_stock:
                    seen_stock.add(r["run"])
                    stock = r["stock_sim"]
                    fidelity.setdefault(city, []).append(r["replay_fidelity_stock_over_real"])
                    rows.append(dict(source="agent", model="stock (fluid, replayed)", city=city,
                                     direction=direction,
                                     throughput_mbps=stock["throughput_mbps_mean"],
                                     retransmits_per_s=stock["retransmits_per_s_mean"],
                                     rtt_ms=stock.get("rtt_median_ms", np.nan),
                                     cwnd_mb=np.nan, rwnd_mb=np.nan, rttvar_ms=np.nan))
    return rows, {c: float(np.median(v)) for c, v in fidelity.items()}


def make_grid(rows, direction, fidelity, out: Path, note: str) -> None:
    arms = [lbl for _, lbl in MEASURED] + [lbl for _, lbl in AGENTS]
    fig, axes = plt.subplots(2, 3, figsize=(19, 9.5))
    width = 0.8 / len(arms)
    for ax, (key, unit, title, sim_has) in zip(axes.ravel(), PANELS):
        for j, arm in enumerate(arms):
            is_sim = arm in {lbl for _, lbl in AGENTS}
            if is_sim and not sim_has:
                continue
            vals, pos = [], []
            for i, city in enumerate(CITIES):
                series = [r[key] for r in rows
                          if r["model"] == arm and r["city"] == city and np.isfinite(r.get(key, np.nan))]
                if series:
                    vals.append(series)
                    pos.append(i + (j - (len(arms) - 1) / 2) * width)
            if not vals:
                continue
            bp = ax.boxplot(vals, positions=pos, widths=width * 0.85, patch_artist=True,
                            showfliers=False, medianprops=dict(color="black", lw=1.1))
            for patch in bp["boxes"]:
                patch.set_facecolor(COLORS.get(arm, "#999999"))
                patch.set_alpha(0.85)
                if is_sim:
                    patch.set_hatch("///")
        ax.set_title(title + ("" if sim_has else "   (measured only)"), fontsize=11)
        ax.set_ylabel(unit)
        ax.set_xticks(range(len(CITIES)))
        ax.set_xticklabels(CITIES, rotation=20)
        ax.grid(alpha=0.25, axis="y")

    handles = [plt.matplotlib.patches.Patch(
        facecolor=COLORS.get(a, "#999"), alpha=0.85,
        hatch="///" if a in {lbl for _, lbl in AGENTS} else None, label=a) for a in arms]
    fig.legend(handles=handles, loc="upper center", ncol=len(arms), frameon=False,
               bbox_to_anchor=(0.5, 0.99), fontsize=10)
    fid = "  ".join(f"{c}:{fidelity[c]:.2f}x" for c in CITIES if c in fidelity)
    fig.suptitle(
        f"Dedicated {direction} over Starlink — measured CCAs (solid) vs simulated native-action QRL "
        f"(hatched, trace-replay conditioned)\n"
        f"Simulated arms are a fluid proxy driven by capacity forcing taken from the same measured runs; "
        f"stock-vs-real replay fidelity: {fid}\n{note}",
        fontsize=9, y=0.055)
    fig.tight_layout(rect=[0, 0.095, 1, 0.945])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay-dir", type=Path, default=ROOT / "outputs" / "replay6")
    ap.add_argument("--directions", nargs="+", default=["downlink", "uplink"])
    ap.add_argument("--note", default="Agent checkpoints: 6-city screen (v7c). The only controlled "
                                      "contrast is agent-vs-stock inside the simulator.")
    args = ap.parse_args()
    for direction in args.directions:
        measured = parse_measured(direction)
        agents, fidelity = collect_replayed(args.replay_dir, direction)
        rows = measured + agents
        out = ROOT / "figures" / f"paper_style_grid_replay_{direction}.png"
        make_grid(rows, direction, fidelity, out, args.note)
        # csv.writer, not manual join: arm labels contain commas
        # ("stock (fluid, replayed)") and would otherwise split into two fields.
        csv_path = out.with_suffix(".csv")
        keys = [k for k, *_ in PANELS]
        with csv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["source", "model", "city", "direction", *keys])
            for r in rows:
                writer.writerow([r["source"], r["model"], r["city"], r["direction"],
                                 *(r.get(k, "") for k in keys)])
        print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
