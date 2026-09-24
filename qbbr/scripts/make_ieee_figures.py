"""Publication figures for main.tex, drawn at IEEEtran's physical sizes.

Why a separate script rather than shrinking the analysis figures: those were
drawn ~15 in wide, so at IEEE's 7.16 in text width their 8 pt labels would print
at ~4 pt. These are laid out for the page they will be printed on:

  figure*  7.16 in wide   (\textwidth in IEEEtran journal)
  figure   3.50 in wide   (\columnwidth)
  text     7-8 pt at final size, Times, vector PDF

Series are separated by line style or hatch as well as colour, because IEEE
papers are routinely printed in greyscale and read by people with colour-vision
deficiency. Colours are the dataviz reference categorical slots validated
all-pairs on a light surface (worst CVD dE 9.2, normal-vision dE 16.3).

  fidelity_rtt_cdf_ieee.pdf  per-second RTT CDFs: measured trace, real Linux BBR,
                             simulated stock, on identical replayed capacity
  replay_deltas_ieee.pdf     paired policy-minus-stock deltas on replayed real
                             capacity (the absolute-level row of the analysis
                             figure is dropped: at print size it was unreadable
                             and it carries no claim the delta row does not)
  tradeoff_plane_ieee.pdf    the policy positioned among the measured CCAs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ROOT = PACKAGE_ROOT.parent
TEXT_WIDTH, COLUMN_WIDTH = 7.16, 3.5
CITIES = ["Sydney", "Tokyo", "Mumbai", "Ohio", "London", "SaoPaulo"]
CITY = {"SaoPaulo": "São Paulo"}
INK, MUTED, GRID, ZERO = "#1a1a19", "#5b5a56", "#e3e2de", "#8a8986"
BLUE, ORANGE, AQUA, VIOLET, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#52514e"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": "#9c9b97", "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType: editable, accepted by IEEE PDF eXpress
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _style(axis, grid_axis="both"):
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _save(fig, out: Path, preview: Path | None):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview.with_name(out.stem + ".png"), dpi=220)
    plt.close(fig)
    print(f"saved -> {out}")


def fidelity_cdf(out_dir: Path, preview: Path | None) -> None:
    series = json.loads((ROOT / "reports" / "per_second_series.json").read_text())["series"]
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 3.55))
    sources = [("real_trace", "measured Starlink trace", NEUTRAL, (0, (4, 2)), 1.1),
               ("real_kernel", "real Linux BBR, replayed capacity", VIOLET, "-", 1.4),
               ("sim_stock", "simulated stock BBR", AQUA, (0, (6, 1.5, 1, 1.5)), 1.4)]
    for axis, city in zip(axes.ravel(), CITIES):
        values = {}
        for key, _label, _c, _ls, _lw in sources:
            got = [np.asarray(r[key]["rtt_ms"], float) for r in series
                   if r["location"] == city and key in r]
            if got:
                values[key] = np.concatenate(got)
        # Trim to the 99th percentile of the widest source: the measured trace's
        # extreme tail would otherwise compress every other curve into the axis.
        hi = max(np.percentile(v, 99) for v in values.values())
        lo = min(np.percentile(v, 0.5) for v in values.values())
        for key, label, colour, linestyle, width in sources:
            v = np.sort(values[key])
            axis.plot(v, np.arange(1, v.size + 1) / v.size, color=colour, linestyle=linestyle,
                      linewidth=width, label=label)
        axis.set_xlim(lo - 0.02 * (hi - lo), hi)
        axis.set_ylim(0, 1.02)
        axis.set_yticks([0, 0.5, 1.0])
        _style(axis)
        path = np.median(values["real_trace"])
        axis.set_title(f"{CITY.get(city, city)}  ({path:.0f} ms median)", loc="left", pad=2)
        model = np.percentile(values["sim_stock"], 90) - np.percentile(values["real_kernel"], 90)
        gap = np.percentile(values["real_kernel"], 90) - np.percentile(values["real_trace"], 90)
        axis.text(0.98, 0.05, f"model $\\Delta$p90 {model:+.1f} ms\nemulation $\\Delta$p90 {gap:+.1f} ms",
                  transform=axis.transAxes, ha="right", va="bottom", fontsize=6.3, color=INK,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRID, linewidth=0.5))
    for axis in axes[1]:
        axis.set_xlabel("RTT (ms)")
    for axis in axes[:, 0]:
        axis.set_ylabel("CDF")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.035), handlelength=2.6, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.6, w_pad=0.8)
    _save(fig, out_dir / "fidelity_rtt_cdf_ieee.pdf", preview)


def replay_deltas(out_dir: Path, preview: Path | None) -> None:
    rows = []
    for core in ("quantum", "classical"):
        rows += json.loads((ROOT / "reports" / "replay" / f"replay_v14_bbr_{core}.json").read_text())["rows"]
    panels = [("throughput_delta_vs_stock_pct", r"$\Delta$ throughput (%)"),
              ("rtt_p90_delta_vs_stock_ms", r"$\Delta$ RTT p90 (ms)"),
              ("retransmits_delta_vs_stock_per_s", r"$\Delta$ retransmissions (s$^{-1}$)")]
    arms = [("quantum", "QA2C", BLUE, None), ("classical", "A2C (matched)", ORANGE, "////")]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.05))
    width = 0.36
    for axis, (field, label) in zip(axes, panels):
        for offset, (core, name, colour, hatch) in enumerate(arms):
            data = [[r[field] for r in rows if r["location"] == c and r["core"] == core] for c in CITIES]
            positions = np.arange(len(CITIES)) + (offset - 0.5) * width
            box = axis.boxplot(data, positions=positions, widths=width * 0.84, patch_artist=True,
                               showfliers=False, whis=(5, 95),
                               medianprops=dict(color=INK, linewidth=0.9),
                               whiskerprops=dict(color=colour, linewidth=0.7),
                               capprops=dict(color=colour, linewidth=0.7),
                               boxprops=dict(linewidth=0.5))
            for patch in box["boxes"]:
                patch.set_facecolor(colour)
                patch.set_alpha(0.85)
                patch.set_edgecolor("white")
                if hatch:
                    patch.set_hatch(hatch)
        axis.axhline(0, color=ZERO, linewidth=0.8, zorder=1)
        axis.set_ylabel(label)
        axis.set_xticks(range(len(CITIES)))
        axis.set_xticklabels([CITY.get(c, c) for c in CITIES], rotation=35, ha="right")
        _style(axis, grid_axis="y")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c, alpha=0.85, hatch=h, edgecolor="white", label=n)
                        for _k, n, c, h in arms],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.0)
    _save(fig, out_dir / "replay_deltas_ieee.pdf", preview)


def tradeoff_plane(out_dir: Path, preview: Path | None) -> None:
    """Both evidence classes aggregated identically: per-city delta vs stock BBR,
    then mean +/- 1 s.e. across the six cities. (The analysis figure used the
    median replay row for the policy; the mean is used here so the two classes
    share one statistic. The mean is lower than the median because of Mumbai.)"""
    from qbbr.data.catalog import build_catalog
    from qbbr.scripts.make_tradeoff_plane_figure import _cca_stats, REAL_CCAS, BASELINE
    catalog = build_catalog(PACKAGE_ROOT / "data" / "raw")
    base = {c: _cca_stats(catalog, c, BASELINE) for c in CITIES}

    def summary(dthr, drtt):
        n = np.sqrt(len(dthr))
        return np.mean(dthr), np.mean(drtt), np.std(dthr) / n, np.std(drtt) / n

    measured = {}
    for cca in REAL_CCAS:
        dthr, drtt = [], []
        for city in CITIES:
            t, r = _cca_stats(catalog, city, cca)
            bt, br = base[city]
            if np.isfinite(t) and np.isfinite(bt) and bt > 0 and np.isfinite(r) and np.isfinite(br):
                dthr.append(100.0 * (t / bt - 1.0))
                drtt.append(r - br)
        measured[cca] = summary(dthr, drtt)
    policy = {}
    for core in ("quantum", "classical"):
        rows = json.loads((ROOT / "reports" / "replay" / f"replay_v14_bbr_{core}.json").read_text())["rows"]
        policy[core] = summary(
            [np.mean([r["throughput_delta_vs_stock_pct"] for r in rows if r["location"] == c]) for c in CITIES],
            [np.mean([r["rtt_p90_delta_vs_stock_ms"] for r in rows if r["location"] == c]) for c in CITIES])
    for name, v in list(measured.items()) + list(policy.items()):
        print(f"  tradeoff {name:<10} dthr {v[0]:+7.2f}% +/-{v[2]:.2f}  drtt {v[1]:+6.2f} ms +/-{v[3]:.2f}")

    names = {"leocc": "LEOCC", "pcc": "PCC", "bbr2": "BBRv2", "hybla": "Hybla",
             "cubic": "CUBIC", "vegas": "Vegas", "ccp": "Copa"}
    # Label anchors in data coordinates with leader lines: Hybla, CUBIC, Vegas
    # and Copa sit within ~20 % / ~6 ms of each other and collide otherwise.
    label_at = {"leocc": (-12, 22, "center"), "pcc": (-60, 18, "center"), "bbr2": (-33, 47, "center"),
                "hybla": (-70, -9, "left"), "cubic": (-86, -5, "center"),
                "vegas": (-102, -12, "left"), "ccp": (-90, -37, "left")}
    fig, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.6))
    axis.axhline(0, color=ZERO, linewidth=0.7, zorder=1)
    axis.axvline(0, color=ZERO, linewidth=0.7, zorder=1)
    for cca, (x, y, ex, ey) in measured.items():
        axis.errorbar(x, y, xerr=ex, yerr=ey, fmt="o", ms=3.4, color=NEUTRAL,
                      ecolor="#b9b8b3", elinewidth=0.6, capsize=1.5, zorder=3)
        tx, ty, ha = label_at[cca]
        axis.annotate(names[cca], (x, y), xytext=(tx, ty), textcoords="data", fontsize=6.5,
                      color=INK, ha=ha, va="center",
                      arrowprops=dict(arrowstyle="-", color="#b9b8b3", linewidth=0.5,
                                      shrinkA=1, shrinkB=2.5))
    x, y, ex, ey = policy["classical"]
    axis.errorbar(x, y, xerr=ex, yerr=ey, fmt="D", ms=4.2, mfc="white", mec=ORANGE, mew=1.0,
                  ecolor=ORANGE, elinewidth=0.6, capsize=1.5, zorder=5, label="A2C (matched)")
    x, y, ex, ey = policy["quantum"]
    axis.errorbar(x, y, xerr=ex, yerr=ey, fmt="*", ms=6.5, color=BLUE, mec="white", mew=0.3,
                  ecolor=BLUE, elinewidth=0.6, capsize=1.5, zorder=6, label="QA2C")
    axis.errorbar([], [], fmt="o", ms=3.4, color=NEUTRAL, label="measured CCA")
    handles, labels = axis.get_legend_handles_labels()
    order = [labels.index(k) for k in ("QA2C", "A2C (matched)", "measured CCA")]
    axis.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
                frameon=False, handletextpad=0.3, borderaxespad=0.2)
    axis.text(0.99, 0.02, "better than stock:\nmore throughput, lower RTT", transform=axis.transAxes,
              ha="right", va="bottom", fontsize=6, color=MUTED, style="italic")
    axis.set_xlabel(r"$\Delta$ throughput vs. stock BBR (%)")
    axis.set_ylabel(r"$\Delta$ RTT p90 vs. stock BBR (ms)")
    axis.set_xlim(-106, 8)
    axis.set_ylim(-44, 58)
    _style(axis)
    fig.tight_layout()
    _save(fig, out_dir / "tradeoff_plane_ieee.pdf", preview)


def seed_freeze(out_dir: Path, preview: Path | None) -> None:
    """Why seeds froze under v11: the deployed argmax leaves stock only when the
    stock-vs-best-non-stock logit margin changes sign, and that margin drifted
    at a rate set by the path's reward SNR. Colour is the arm, line style the
    path, as in the other figures."""
    cases = [("freeze_quantum_Sydney_downlink_1.json", "Sydney, QA2C seed 1", BLUE, "-"),
             ("freeze_quantum_London_downlink_1.json", "London, QA2C seed 1", BLUE, (0, (4, 1.6))),
             ("freeze_classical_London_downlink_4.json", "London, A2C seed 4", ORANGE, (0, (4, 1.6)))]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 2.75), sharex=True,
                                      gridspec_kw=dict(height_ratios=[1.6, 1.0]))
    top.axhspan(0, 0.75, color="#f1f0ec", zorder=0)
    top.text(29.6, 0.66, "margin $>0$: argmax deploys stock", ha="right", va="top", fontsize=6.3,
             color=MUTED, style="italic")
    for name, label, colour, style in cases:
        rows = next(iter(json.loads((ROOT / "reports" / name).read_text()).values()))
        episode = np.arange(1, len(rows) + 1)
        margin = np.array([r["stock_logit_margin"] for r in rows])
        greedy = np.array([r["greedy_nonstock_frac"] for r in rows])
        top.plot(episode, margin, color=colour, linestyle=style, linewidth=1.3, label=label)
        bottom.plot(episode, greedy, color=colour, linestyle=style, linewidth=1.3)
        if name.startswith("freeze_quantum_Sydney"):
            cross = int(np.argmax(margin < 0))
            top.annotate(f"sign change, episode {cross + 1}", (episode[cross], margin[cross]),
                         xytext=(13.5, -0.62), fontsize=6.3, color=INK, ha="left",
                         arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5, shrinkB=1.5))
    top.axhline(0, color=ZERO, linewidth=0.8, zorder=1)
    top.set_ylim(-1.45, 0.75)
    top.set_ylabel("stock logit margin")
    top.legend(loc="lower left", frameon=False, handlelength=2.4, borderaxespad=0.2)
    bottom.set_ylim(-0.04, 1.0)
    bottom.set_yticks([0, 0.5, 1.0])
    bottom.set_ylabel("greedy non-stock\nshare")
    bottom.set_xlabel("training episode")
    bottom.set_xlim(1, 30)
    for axis in (top, bottom):
        _style(axis)
    fig.tight_layout(h_pad=0.4)
    fig.align_ylabels((top, bottom))
    _save(fig, out_dir / "seed_freeze_ieee.pdf", preview)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--preview-dir", type=Path, default=None,
                        help="Also write PNG previews here (for inspection only).")
    parser.add_argument("--only", nargs="+", choices=None, default=None,
                        help="Subset of figures to draw (default: all).")
    args = parser.parse_args()
    preview = args.preview_dir / "x.png" if args.preview_dir else None
    for name in args.only or list(FIGURES):
        FIGURES[name](args.out_dir, preview)
    return 0


def snr_gate(out_dir: Path, preview: Path | None) -> None:
    """The measurement that chose the reward. Uniform-random actions, so it
    characterises the environment rather than a policy. Top: best pairwise
    action separation on London downlink by reward form (the hardest downlink
    cell). Bottom: the adopted reward on all twelve cells. All runs share the v14
    config and calibration, seed 0, 12 episodes (reports/snr_v14/)."""
    src = ROOT / "reports" / "snr_v14"
    load = lambda name: json.loads((src / f"{name}.json").read_text())
    rewards = [("London_downlink_throughput_only", "absolute throughput"),
               ("London_downlink_legacy_alpha_fair", "absolute multi-objective"),
               ("London_downlink_difference_delta0.25", r"difference, $\delta=0.25$"),
               ("London_downlink_difference_delta0.5", r"difference, $\delta=0.5$ (used)"),
               ("London_downlink_difference_delta1.0", r"difference, $\delta=1$"),
               ("London_downlink_difference_delta2.0", r"difference, $\delta=2$"),
               ("London_downlink_difference_delta4.0", r"difference, $\delta=4$")]
    gate = 2.0
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.45),
                                      gridspec_kw=dict(height_ratios=[1.0, 1.0]))
    y = np.arange(len(rewards))[::-1]
    for row, (name, label) in zip(y, rewards):
        d = load(name)
        sigma, r2, passed = d["best_separation_sigma"], d["r2_state"], d["passed"]
        colour = VIOLET if "difference" in name else NEUTRAL
        top.barh(row, sigma, height=0.62, color=colour if passed else "white", edgecolor=colour,
                 hatch=None if passed else "////", linewidth=0.7, alpha=0.9 if passed else 1.0)
        top.text(sigma + 0.3, row, f"{sigma:.1f}$\\sigma$,  $R^2$ {r2:.2f}", va="center",
                 fontsize=6.3, color=INK)
    top.set_yticks(y)
    top.set_yticklabels([label for _n, label in rewards])
    top.axvline(gate, color=INK, linewidth=0.7, linestyle=(0, (2, 1.5)))
    top.set_xlabel("best action separation, London downlink ($\\sigma$)")
    _style(top, grid_axis="x")

    cells = [(c, d) for c in CITIES for d in ("downlink", "uplink")]
    rows = {c: i for i, c in enumerate(CITIES[::-1])}
    for direction, marker, face in (("downlink", "o", VIOLET), ("uplink", "s", "white")):
        xs, ys = [], []
        for city in CITIES:
            d = load(f"{city}_{direction}_difference_delta0.5")
            xs.append(d["best_separation_sigma"])
            ys.append(rows[city] + (0.14 if direction == "downlink" else -0.14))
        bottom.scatter(xs, ys, marker=marker, s=16, facecolor=face, edgecolor=VIOLET,
                       linewidth=0.8, zorder=3, label=direction)
    bottom.axvline(gate, color=INK, linewidth=0.7, linestyle=(0, (2, 1.5)))
    bottom.set_yticks(list(rows.values()))
    bottom.set_yticklabels([CITY.get(c, c) for c in rows])
    bottom.set_xlim(0, None)
    bottom.set_xlabel(r"best action separation, difference reward $\delta=0.5$ ($\sigma$)")
    bottom.text(gate + 0.35, len(CITIES) - 0.55, "gate ($2\\sigma$)", fontsize=6.3, color=INK, va="center")
    bottom.legend(loc="center left", bbox_to_anchor=(0.12, 0.62), frameon=False,
                  handletextpad=0.2, borderaxespad=0.1)
    best = {load(f"{c}_{d}_difference_delta0.5")["pooled_best_action"] for c, d in cells}
    constant = sum(not load(f"{c}_{d}_difference_delta0.5")["state_dependent_features"] for c, d in cells)
    note = ("pooled best action:\ngain 1.25 in all 12 cells" if best == {4}
            else f"pooled best actions: {sorted(best)}")
    bottom.text(0.12, 0.24, f"{note};\nno stratum changes it in {constant} of 12",
                transform=bottom.transAxes, fontsize=6.3, color=MUTED, style="italic", va="center")
    _style(bottom, grid_axis="x")
    top.set_xlim(0, max(bottom.get_xlim()[1], top.get_xlim()[1]) * 1.12)
    bottom.set_xlim(top.get_xlim())
    fig.tight_layout(h_pad=0.8)
    _save(fig, out_dir / "snr_gate_ieee.pdf", preview)


def policy_boxes(out_dir: Path, preview: Path | None) -> None:
    """Absolute per-second throughput and RTT, every source side by side on the
    same replayed capacity: the measured Starlink trace (data/raw, stock BBR),
    real Linux BBR on the testbed, and the simulator's stock, QA2C and A2C.

    Read the policy against SIMULATED stock (same model, same forcing); the
    measured and testbed boxes show how far that model sits from reality, which
    is the scale any policy effect has to beat. Policies are the deployed seed 0
    (the paired multi-seed deltas are in replay_deltas_ieee.pdf)."""
    series = json.loads((ROOT / "reports" / "per_second_series.json").read_text())["series"]
    calibration = json.loads((PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants_v14.json").read_text())
    sources = [("real_trace", "measured Starlink trace", NEUTRAL, None),
               ("real_kernel", "real Linux BBR (testbed)", VIOLET, "//////"),
               ("sim_stock", "simulated stock BBR", AQUA, None),
               ("sim_quantum", "simulated QA2C", BLUE, None),
               ("sim_classical", "simulated A2C", ORANGE, None)]
    rtt_cap = 32.0
    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH, 3.75), sharex=True)
    width = 0.16
    for axis, metric in zip(axes, ("throughput_mbps", "rtt_ms")):
        for offset, (key, _label, colour, hatch) in enumerate(sources):
            data = []
            for city in CITIES:
                v = np.concatenate([np.asarray(r[key][metric], float) for r in series
                                    if r["location"] == city and key in r])
                if metric == "rtt_ms":
                    # RTT above the path's propagation floor: raw RTT spans 30-400 ms
                    # across paths, which flattens every box; the floor is physics,
                    # the remainder is the queueing a controller influences.
                    v = np.clip(v - calibration[city]["downlink"]["RTT_min_ms"], 0.0, None)
                data.append(v)
            positions = np.arange(len(CITIES)) + (offset - 2) * width
            box = axis.boxplot(data, positions=positions, widths=width * 0.82, patch_artist=True,
                               showfliers=False, whis=(5, 95),
                               medianprops=dict(color="white", linewidth=1.0),
                               whiskerprops=dict(color=colour, linewidth=0.7),
                               capprops=dict(color=colour, linewidth=0.7),
                               boxprops=dict(linewidth=0.0))
            for patch in box["boxes"]:
                patch.set_facecolor(colour)
                patch.set_edgecolor("white")
                if hatch:
                    patch.set_hatch(hatch)
            if metric == "rtt_ms":
                for x, v in zip(positions, data):
                    top = np.percentile(v, 95)
                    if top > rtt_cap:
                        axis.annotate(f"{top:.0f}", (x, rtt_cap), xytext=(0, 1.5), textcoords="offset points",
                                      ha="center", va="bottom", fontsize=5.8, color=colour,
                                      annotation_clip=False)
        _style(axis, grid_axis="y")
    axes[0].set_ylabel("throughput (Mbps)")
    axes[1].set_ylabel("RTT above floor (ms)")
    axes[1].set_ylim(-1, rtt_cap)
    axes[1].set_xticks(range(len(CITIES)))
    axes[1].set_xticklabels([CITY.get(c, c) for c in CITIES])
    axes[1].set_xlim(-0.55, len(CITIES) - 0.45)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c, hatch=h, edgecolor="white", label=n) for _k, n, c, h in sources],
               loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.035),
               handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=0.5)
    _save(fig, out_dir / "policy_boxes_ieee.pdf", preview)


def tier1_cells(out_dir: Path, preview: Path | None) -> None:
    """Tier-1 hold-out result for every location-direction cell and both arms.

    Left: throughput delta vs stock -- each training seed as a dot, the median
    as a bar with its 95% bootstrap interval. Right: RTT p90 delta, seeds as
    dots, and the cell's derived RTT budget as a black tick. The RTT criterion
    requires EVERY seed to sit at or left of the tick, which is why it, not
    throughput, decides qualification. Qualified cells are shaded."""
    report = json.loads((ROOT / "reports" / "tier1_v14_report.json").read_text())
    cells = [(c, d) for d in ("downlink", "uplink") for c in CITIES]
    row_of = {cell: i for i, cell in enumerate(cells)}
    n = len(cells)
    y_of = lambda cell: n - 1 - row_of[cell] - (0.5 if cell[1] == "uplink" else 0.0)
    arms = [("quantum", "QA2C", BLUE, "o", +0.16), ("classical", "A2C (matched)", ORANGE, "s", -0.16)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.3), sharey=True,
                                      gridspec_kw=dict(width_ratios=[1.0, 1.15], wspace=0.06))
    assessment = {(a["location"], a["direction"], a["core"]): a for a in report["selection_assessment"]}
    for cell in cells:
        y = y_of(cell)
        if all(assessment[cell + (core,)]["qualified_simulator_proxy_result"] for core, *_ in arms):
            for axis in (left, right):
                axis.axhspan(y - 0.45, y + 0.45, color="#e7f3ec", zorder=0, linewidth=0)
        limit = assessment[cell + ("quantum",)]["resolved_limits"]["max_rtt_p90_delta_vs_stock_ms"]
        right.plot([limit, limit], [y - 0.38, y + 0.38], color=INK, linewidth=1.3, zorder=4,
                   solid_capstyle="butt")
        for core, _name, colour, marker, dy in arms:
            a = assessment[cell + (core,)]
            recs = [r for r in report["records"]
                    if (r["location"], r["direction"], r["core"]) == cell + (core,)]
            thr = [r["evaluation"]["throughput_delta_vs_stock_pct"] for r in recs]
            rtt = [r["evaluation"]["rtt_p90_delta_vs_stock_ms"] for r in recs]
            ci = a["bootstrap_median_throughput_delta_vs_stock_pct"]
            left.plot([ci["lower"], ci["upper"]], [y + dy] * 2, color=colour, linewidth=1.0, zorder=2)
            left.scatter(thr, [y + dy] * len(thr), s=6, marker=marker, facecolor="white",
                         edgecolor=colour, linewidth=0.6, zorder=3)
            left.scatter([ci["point_estimate"]], [y + dy], s=20, marker=marker, color=colour,
                         edgecolor="white", linewidth=0.4, zorder=5)
            right.scatter(rtt, [y + dy] * len(rtt), s=6, marker=marker, facecolor="white",
                          edgecolor=colour, linewidth=0.6, zorder=3)
            right.scatter([np.median(rtt)], [y + dy], s=20, marker=marker, color=colour,
                          edgecolor="white", linewidth=0.4, zorder=5)
    ticks = [y_of(cell) for cell in cells]
    left.set_yticks(ticks)
    left.set_yticklabels([CITY.get(c, c) for c, _d in cells])
    for axis in (left, right):
        axis.axvline(0, color=ZERO, linewidth=0.7, zorder=1)
        axis.axhline(y_of(("Sydney", "uplink")) + 0.75, color="#b9b8b3", linewidth=0.6)
        _style(axis, grid_axis="x")
        axis.set_ylim(-0.8, n - 0.1)
    left.text(-0.02, y_of(("Sydney", "downlink")) + 0.75, "downlink", transform=left.get_yaxis_transform(),
              ha="right", va="center", fontsize=7, fontweight="bold", color=INK)
    left.text(-0.02, y_of(("Sydney", "uplink")) + 0.45, "uplink", transform=left.get_yaxis_transform(),
              ha="right", va="center", fontsize=7, fontweight="bold", color=INK)
    left.set_xlabel(r"$\Delta$ throughput vs. stock (%)")
    right.set_xlabel(r"$\Delta$ RTT p90 vs. stock (ms)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=c, marker=m, linestyle="-", linewidth=1.0, markersize=4,
                      markeredgecolor="white", label=f"{name}: median, 95% CI")
               for _k, name, c, m, _dy in arms]
    handles += [Line2D([], [], color=MUTED, marker="o", linestyle="", markersize=3, markerfacecolor="white",
                       label="training seed"),
                Line2D([], [], color=INK, marker="|", linestyle="", markersize=7, markeredgewidth=1.3,
                       label="derived RTT budget"),
                Patch(facecolor="#e7f3ec", label="all gates pass")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.53, 1.03),
               handletextpad=0.4, columnspacing=1.2)
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.12, top=0.9)
    _save(fig, out_dir / "tier1_cells_ieee.pdf", preview)


def learning_curves(out_dir: Path, preview: Path | None) -> None:
    """RQ3 at training time: episode return of QA2C and the parameter-matched
    A2C on every cell. Lines are the mean over five training seeds, bands +/-1
    s.d. Each panel prints the paired (same seed) difference QA2C - A2C over the
    last five episodes with a t-based 95% interval, so 'indistinguishable' is a
    number rather than an impression. Returns are sums of the difference reward
    under sampled (exploratory) actions, so their scale differs by path."""
    from scipy import stats
    report = json.loads((ROOT / "reports" / "tier1_v14_report.json").read_text())
    arms = [("quantum", "QA2C", BLUE, "-"), ("classical", "A2C (matched)", ORANGE, (0, (3.5, 1.5)))]
    fig, axes = plt.subplots(2, len(CITIES), figsize=(TEXT_WIDTH, 2.75), sharex=True)
    for row, direction in enumerate(("downlink", "uplink")):
        for col, city in enumerate(CITIES):
            axis = axes[row, col]
            last = {}
            for core, name, colour, style in arms:
                recs = sorted((r for r in report["records"]
                               if (r["location"], r["direction"], r["core"]) == (city, direction, core)),
                              key=lambda r: int(r["seed"]))
                curves = np.array([r["training"]["episode_rewards"] for r in recs], float)
                episode = np.arange(1, curves.shape[1] + 1)
                mean, sd = curves.mean(0), curves.std(0)
                axis.fill_between(episode, mean - sd, mean + sd, color=colour, alpha=0.15, linewidth=0)
                axis.plot(episode, mean, color=colour, linestyle=style, linewidth=1.0, label=name)
                last[core] = curves[:, -5:].mean(1)
            diff = last["quantum"] - last["classical"]
            half = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
            axis.text(0.97, 0.05, f"{diff.mean():+.1f} [{diff.mean() - half:+.1f}, {diff.mean() + half:+.1f}]",
                      transform=axis.transAxes, ha="right", va="bottom", fontsize=5.8, color=INK, zorder=6,
                      bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
            _style(axis)
            axis.tick_params(labelsize=6, pad=1.5)
            axis.set_xticks([1, 15, 30])
            if row == 0:
                axis.set_title(CITY.get(city, city), pad=2)
            if col == 0:
                axis.set_ylabel(f"{direction}\nepisode return", fontsize=7)
            if row == 1:
                axis.set_xlabel("episode", fontsize=7, labelpad=1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels + [], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.5, w_pad=0.3)
    _save(fig, out_dir / "learning_curves_ieee.pdf", preview)


def retrained_heldout(out_dir: Path, preview: Path | None) -> None:
    """Pre-registered retraining on real capacity (bbr runs 1-7), evaluated on
    held-out runs 8-10. Each marker is one training seed's median over the three
    held-out traces; the bar is the primary statistic (median over seeds). The
    dashed line is the registered success threshold, which applies to Sydney
    only (its measured transport-model throughput error); Tokyo's own error,
    7.0%, is off this axis."""
    import glob
    rows = []
    for path in sorted(glob.glob(str(ROOT / "reports" / "train_on_traces" / "*_seed*.json"))):
        rows += json.loads(Path(path).read_text())["rows"]
    groups = [(city, core) for city in ("Sydney", "Tokyo") for core in ("quantum", "classical")]
    style = {"quantum": (BLUE, "o", "QA2C"), "classical": (ORANGE, "s", "A2C")}
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 2.2))
    fields = [("heldout_median_throughput_delta_pct", r"held-out $\Delta$ throughput (%)"),
              ("heldout_median_rtt_p90_delta_ms", r"held-out $\Delta$ RTT p90 (ms)")]
    rng = np.random.default_rng(0)
    positions = [0, 1, 2.5, 3.5]
    for axis, (field, label) in zip(axes, fields):
        axis.axhline(0, color=ZERO, linewidth=0.7, zorder=1)
        for x, (city, core) in zip(positions, groups):
            colour, marker, _name = style[core]
            values = np.array([r[field] for r in rows if r["location"] == city and r["core"] == core])
            jitter = rng.uniform(-0.18, 0.18, values.size)
            axis.scatter(x + jitter, values, s=11, marker=marker, facecolor="white", edgecolor=colour,
                         linewidth=0.8, zorder=3)
            axis.plot([x - 0.3, x + 0.3], [np.median(values)] * 2, color=colour, linewidth=1.8, zorder=4,
                      solid_capstyle="butt")
        if field.startswith("heldout_median_throughput"):
            axis.plot([-0.45, 1.45], [1.5, 1.5], color=INK, linewidth=0.8, linestyle=(0, (3, 1.5)), zorder=2)
            axis.text(-0.45, 1.62, "success threshold", fontsize=6, color=INK, va="bottom")
        axis.set_xticks(positions)
        axis.set_xticklabels(["QA2C", "A2C", "QA2C", "A2C"], fontsize=6.5)
        for centre, city in ((0.5, "Sydney"), (3.0, "Tokyo")):
            axis.text(centre, -0.13, city, transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=7)
        axis.set_xlim(-0.6, 4.1)
        axis.set_ylabel(label)
        _style(axis, grid_axis="y")
    axes[0].set_ylim(-4.6, 2.3)
    fig.tight_layout(w_pad=1.0)
    fig.subplots_adjust(bottom=0.17)
    _save(fig, out_dir / "retrained_heldout_ieee.pdf", preview)


def sixcity_boxes(out_dir: Path, preview: Path | None) -> None:
    """Six-city comparison in the layout the literature uses: one box per arm per
    city, for throughput, retransmissions and RTT.

    Arms are the measured Starlink BBR runs from the dataset, real Linux BBR on
    the testbed, and the three simulated arms -- all driven by capacity replayed
    from those same measured runs, so the measured box is the thing the model is
    trying to reproduce rather than a competitor. Stock is seed-independent
    (5 traces per city); the policies add 5 seeds on top (25 values). The fourth
    panel is measured-only: neither the fluid model nor the testbed reports a
    congestion window the way the dataset's TCP_INFO logs do.
    """
    import pandas as pd
    from qbbr.data.catalog import build_catalog, FileRecord
    from qbbr.data.loader import load_trace

    rows = {core: json.loads((ROOT / "reports" / "replay" / f"replay_v14_bbr_{core}.json").read_text())["rows"]
            for core in ("quantum", "classical")}
    catalog = build_catalog(PACKAGE_ROOT / "data" / "raw")

    kernel = {c: {"thr": [], "rtx": [], "rtt": [], "cwnd": []} for c in CITIES}
    for row in json.loads((ROOT / "reports" / "kernel_testbed_6city_clean.json").read_text())["rows"]:
        k, city = row["kernel"], row["location"]
        if city not in kernel or not k.get("emulated_rtt_valid", False):
            continue
        kernel[city]["thr"].append(k["throughput_mbps_mean"])
        kernel[city]["rtt"].append(k["rtt_p90_ms"])
        kernel[city]["rtx"].append(k["retransmits_total"] / max(k["intervals"], 1))

    measured = {c: {"thr": [], "rtx": [], "rtt": [], "cwnd": []} for c in CITIES}
    selection = catalog[(catalog["direction"] == "downlink") & (catalog["cca"] == "bbr")
                        & (catalog["category"].str.contains("sequential"))]
    for _, row in selection.iterrows():
        if row["location"] not in measured:
            continue
        trace = load_trace(FileRecord(path=Path(row["path"]), category=row["category"],
                                      direction=row["direction"], location=row["location"],
                                      cca=row["cca"], run=int(row["run"])))
        frame = trace.intervals[trace.intervals.get("omitted") != True]  # noqa: E712
        num = lambda col: pd.to_numeric(frame[col], errors="coerce").dropna()
        m = measured[row["location"]]
        m["thr"].append(num("bits_per_second").mean() / 1e6)
        m["rtx"].append(num("retransmits").mean())
        m["rtt"].append(float(np.percentile(num("rtt_ms"), 90)))
        m["cwnd"].append(num("snd_cwnd").mean() / 1e6)

    def sim(city, arm, field):
        if arm == "stock":  # identical across seeds: keep one value per trace
            seen = {}
            for r in rows["quantum"]:
                if r["location"] == city:
                    seen[r["run"]] = r["stock_sim"][field]
            return list(seen.values())
        core = "quantum" if arm == "QA2C" else "classical"
        return [r["agent_sim"][field] for r in rows[core] if r["location"] == city]

    arms = [("measured", "measured Starlink trace", NEUTRAL, None),
            ("kernel", "real Linux BBR (testbed)", VIOLET, "//////"),
            ("stock", "simulated stock BBR", AQUA, None),
            ("QA2C", "simulated QA2C", BLUE, None),
            ("A2C", "simulated A2C", ORANGE, None)]
    panels = [("throughput_mbps_mean", "thr", "throughput (Mbps)", None),
              ("retransmits_per_s_mean", "rtx", "retransmissions (s$^{-1}$)", None),
              ("rtt_p90_ms", "rtt", "RTT p90 (ms)", None),
              (None, "cwnd", "congestion window (MB)", "measured traces only")]

    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 5.0))
    width = 0.16
    for axis, (field, key, label, only) in zip(axes.ravel(), panels):
        for offset, (arm, _name, colour, hatch) in enumerate(arms):
            if only and arm != "measured":
                continue
            data, positions = [], []
            for index, city in enumerate(CITIES):
                if arm == "measured":
                    values = measured[city][key]
                elif arm == "kernel":
                    values = kernel[city][key]
                else:
                    values = sim(city, arm, field)
                if values:
                    data.append(np.asarray(values, float))
                    positions.append(index + (offset - (len(arms) - 1) / 2) * width)
            box = axis.boxplot(data, positions=positions, widths=width * 0.84, patch_artist=True,
                               showfliers=False, whis=(5, 95),
                               medianprops=dict(color="white", linewidth=1.0),
                               whiskerprops=dict(color=colour, linewidth=0.7),
                               capprops=dict(color=colour, linewidth=0.7),
                               boxprops=dict(linewidth=0.0))
            for patch in box["boxes"]:
                patch.set_facecolor(colour)
                # Same-colour edge, not white: several arms have near-zero spread
                # (the simulator's retransmission rate barely moves), and a white
                # edge erases a box that thin.
                patch.set_edgecolor(colour)
                patch.set_linewidth(0.5)
                if hatch:
                    patch.set_hatch(hatch)
        if only:
            axis.text(0.02, 0.94, only, transform=axis.transAxes, ha="left", va="top",
                      fontsize=6.5, color=MUTED, style="italic")
        axis.set_ylabel(label)
        axis.set_xticks(range(len(CITIES)))
        axis.set_xticklabels([CITY.get(c, c) for c in CITIES], rotation=20, ha="right")
        axis.set_xlim(-0.55, len(CITIES) - 0.45)
        _style(axis, grid_axis="y")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c, hatch=h, edgecolor="white", label=n) for _k, n, c, h in arms],
               loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.025),
               handlelength=1.6, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.0, w_pad=1.6)
    _save(fig, out_dir / "sixcity_boxes_ieee.pdf", preview)


FIGURES = {"fidelity": fidelity_cdf, "replay": replay_deltas, "tradeoff": tradeoff_plane,
           "freeze": seed_freeze, "snr": snr_gate,
           "boxes": policy_boxes, "tier1": tier1_cells,
           "learning": learning_curves, "retrained": retrained_heldout,
           "sixcity": sixcity_boxes}


if __name__ == "__main__":
    raise SystemExit(main())
