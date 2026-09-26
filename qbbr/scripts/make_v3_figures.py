"""Figures, LaTeX table rows, and headline numbers for the final-v3 study (queue-delay reward)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = ROOT / "qbbr" / "data" / "calibrated" / "per_location_constants_v14.json"
CITIES = ["Sydney", "Tokyo", "Mumbai", "Ohio", "London", "SaoPaulo"]
PLOT_CITIES = ["Tokyo", "SaoPaulo", "Ohio", "London", "Mumbai", "Sydney"]
CITY_LABEL = {"SaoPaulo": "Sao Paulo"}
TEX_CITY = {"SaoPaulo": "S\\~ao Paulo"}
DIRECTIONS = ["downlink", "uplink"]
SEEDS = range(5)
CORES = [("a2c", "Classical A2C-BBR"), ("qa2c", "Hybrid QA2C-BBR"), ("qdqn", "Recurrent QDQN-BBR")]
COLOR = {"a2c": "#4f8fcf", "qa2c": "#f0a04b", "qdqn": "#9c6fb6", "stock": "#35a893"}
EDGE = "#4d4d4d"
# Simulator error against real Linux BBR (BBRv1) on identical replayed downlink capacity (Tier 3, stock only).
# outputs/ is not tracked, so tier3_model_error.py must be run first; main() fills this in.
TIER3_PATH = ROOT / "outputs" / "rq_study" / "final-v3" / "tier3_model_error.json"
MODEL_ERROR: dict = {}


def load_model_error(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} not found; run `python -m qbbr.scripts.tier3_model_error` first.")
    tier3 = json.loads(path.read_text())
    return {"thr": {c: round(v["thr_err_pct"], 1) for c, v in tier3.items()},
            "rtt": {c: round(v["rtt_err_ms"], 1) for c, v in tier3.items()}}


KEYS = {"thr": "throughput_delta_pct", "rtt": "rtt_p90_delta_ms", "rtx": "retransmits_delta_per_s"}
RNG = np.random.default_rng(20260920)


def boot(values) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    medians = [np.median(RNG.choice(x, len(x))) for _ in range(10000)]
    return float(np.median(x)), float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def load(study: Path) -> dict:
    """(stage, tier, location, direction, core, variant, seed) -> list of per-holdout rows."""
    out = {}
    for stage in ("1a_full", "1b_ablation", "1c_qdqn", "2_replay_train"):
        for path in (study / stage).glob("*/result.json"):
            r = json.loads(path.read_text())
            j = r["job"]
            for tier, key in ((1, "evaluation"), (2, "transfer_evaluation")):
                if r.get(key):
                    tier_ = 2 if stage == "2_replay_train" else tier
                    out[(stage, tier_, j["location"], j["direction"], j["core"], j["variant"], j["seed"])] = r[key]
    return out


def seed_means(data, stage, tier, loc, direction, core, variant="full", metric="thr"):
    rows = [data.get((stage, tier, loc, direction, core, variant, s)) for s in SEEDS]
    return [float(np.mean([e[KEYS[metric]] for e in r])) for r in rows if r]


def pairs(data, stage, tier, loc, direction, core, variant="full", metric="thr"):
    """Every (training seed, held-out seed[, trace]) paired delta, for box plots."""
    return [e[KEYS[metric]] for s in SEEDS
            for e in (data.get((stage, tier, loc, direction, core, variant, s)) or [])]


def gate(data, tier, loc, direction, core, budgets) -> bool:
    thr = seed_means(data, "1a_full", tier, loc, direction, core)
    rtt = float(np.median(seed_means(data, "1a_full", tier, loc, direction, core, metric="rtt")))
    rtx = float(np.median(seed_means(data, "1a_full", tier, loc, direction, core, metric="rtx")))
    median, low, _ = boot(thr)
    positive = np.mean(np.asarray(thr) > 0)
    return median >= 0 and low >= 0 and positive >= 0.70 and rtt <= budgets[loc][direction] and rtx <= 0.25


def style(ax, ylabel=None, title=None):
    ax.set_xticks(range(len(PLOT_CITIES)))
    ax.set_xticklabels([CITY_LABEL.get(c, c) for c in PLOT_CITIES], rotation=25)
    ax.grid(axis="y", color="#e4e4e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=12)


def boxes(ax, groups, width=0.36):
    offset = (len(groups) - 1) / 2
    for i, (core, values) in enumerate(groups):
        positions = [c + (i - offset) * (width + 0.02) for c in range(len(PLOT_CITIES))]
        ax.boxplot(values, positions=positions, widths=width, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=COLOR[core], edgecolor=EDGE, linewidth=0.8),
                   medianprops=dict(color="#1a1a19", linewidth=1.4),
                   whiskerprops=dict(color=EDGE, linewidth=0.8), capprops=dict(color=EDGE, linewidth=0.8))


def fig_tiers(data, out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5), sharey="row")
    columns = [(1, "downlink"), (2, "downlink"), (1, "uplink"), (2, "uplink")]
    for row, (metric, ylabel) in enumerate((("thr", "Throughput vs stock (%)"), ("rtt", "RTT p90 vs stock (ms)"))):
        for col, (tier, direction) in enumerate(columns):
            ax = axes[row, col]
            if tier == 2 and direction == "downlink":
                for c, city in enumerate(PLOT_CITIES):
                    err = abs(MODEL_ERROR[metric][city])
                    ax.add_patch(plt.Rectangle((c - 0.45, -err), 0.9, 2 * err, color="#e6e6e3", zorder=0))
            boxes(ax, [(core, [pairs(data, "1a_full", tier, city, direction, core, metric=metric)
                               for city in PLOT_CITIES]) for core in ("a2c", "qa2c")])
            ax.axhline(0, color=COLOR["stock"], linewidth=1.6, zorder=1)
            name = "synthetic capacity" if tier == 1 else "trace-derived capacity"
            style(ax, ylabel if col == 0 else None, f"Tier {tier}, {direction}: {name}" if row == 0 else None)
    axes[0, 0].set_ylim(-8, 11)
    axes[1, 0].set_ylim(-18, 18)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[c], edgecolor=EDGE) for c in ("a2c", "qa2c")]
    handles += [plt.Line2D([0], [0], color=COLOR["stock"], linewidth=1.6), plt.Rectangle((0, 0), 1, 1, facecolor="#e6e6e3")]
    labels = ["Classical A2C-BBR", "Hybrid QA2C-BBR", "Simulated stock proxy (zero)", "Tier-3 discrepancy vs real Linux BBR (BBRv1, downlink)"]
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_ablation(data, out: Path) -> None:
    import matplotlib.pyplot as plt
    rows = [(core, direction, variant) for direction in DIRECTIONS for core in ("qa2c", "a2c")
            for variant in ("telemetry", "telemetry_queue")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, (metric, xlabel) in zip(axes, (("thr", "Full minus reduced state: throughput (pp)"),
                                           ("rtt", "Full minus reduced state: RTT p90 (ms)"))):
        for y, (core, direction, variant) in enumerate(rows):
            diffs = [f - r for loc in CITIES for f, r in zip(
                seed_means(data, "1a_full", 1, loc, direction, core, "full", metric),
                seed_means(data, "1b_ablation", 1, loc, direction, core, variant, metric))]
            jitter = RNG.uniform(-0.18, 0.18, len(diffs))
            ax.scatter(diffs, y + jitter, s=12, color=COLOR[core], alpha=0.55, linewidths=0)
            median, low, high = boot(diffs)
            ax.plot([low, high], [y, y], color="#1a1a19", linewidth=2)
            ax.plot([median], [y], marker="|", markersize=14, color="#1a1a19")
        ax.axvline(0, color=COLOR["stock"], linewidth=1.4)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#e4e4e0", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    labels = [f"{'QA2C' if c == 'qa2c' else 'A2C'} {d} vs {'telemetry' if v == 'telemetry' else 'telemetry+queue'}"
              for c, d, v in rows]
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_operating_points(data, out: Path) -> dict:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    summary = {}
    for ax, direction in zip(axes, DIRECTIONS):
        for core, label in CORES:
            stage = "1c_qdqn" if core == "qdqn" else "1a_full"
            thr = [v for loc in CITIES for v in seed_means(data, stage, 1, loc, direction, core)]
            rtt = [v for loc in CITIES for v in seed_means(data, stage, 1, loc, direction, core, metric="rtt")]
            t, r = boot(thr), boot(rtt)
            summary[(core, direction)] = dict(thr=t, rtt=r)
            ax.errorbar(r[0], t[0], xerr=[[r[0] - r[1]], [r[2] - r[0]]], yerr=[[t[0] - t[1]], [t[2] - t[0]]],
                        fmt="o", markersize=9, color=COLOR[core], ecolor=COLOR[core], capsize=3, label=label)
        ax.plot([0], [0], marker="s", markersize=9, color=COLOR["stock"], label="Stock BBR-v3")
        ax.axhline(0, color="#cccccc", linewidth=0.8)
        ax.axvline(0, color="#cccccc", linewidth=0.8)
        ax.set_title(f"Tier 1, {direction}", fontsize=12)
        ax.set_xlabel("RTT p90 vs stock (ms)")
        ax.grid(color="#eeeeea", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Throughput vs stock (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return summary


def fmt(value, digits=2):
    return f"${value:+.{digits}f}$"


def core_table(data, budgets) -> str:
    lines = []
    for direction in DIRECTIONS:
        for loc in CITIES:
            cells = []
            for tier in (1, 2):
                for core in ("qa2c", "a2c"):
                    thr = float(np.median(seed_means(data, "1a_full", tier, loc, direction, core)))
                    rtt = float(np.median(seed_means(data, "1a_full", tier, loc, direction, core, metric="rtt")))
                    mark = "$^\\ast$" if tier == 1 and gate(data, 1, loc, direction, core, budgets) else ""
                    cells.append(f"{fmt(thr)}{mark} & {fmt(rtt)}")
            err = (f"{fmt(MODEL_ERROR['thr'][loc], 1)} & {fmt(MODEL_ERROR['rtt'][loc], 1)}"
                   if direction == "downlink" else "-- & --")
            name = f"{TEX_CITY.get(loc, loc)} ({'DL' if direction == 'downlink' else 'UL'})"
            lines.append(f"{name} & {cells[0]} & {cells[1]} && {cells[2]} & {cells[3]} && {err} \\\\")
        lines.append("\\midrule")
    for direction in DIRECTIONS:
        cells = []
        for tier in (1, 2):
            for core in ("qa2c", "a2c"):
                thr = [v for loc in CITIES for v in seed_means(data, "1a_full", tier, loc, direction, core)]
                rtt = [v for loc in CITIES for v in seed_means(data, "1a_full", tier, loc, direction, core, metric="rtt")]
                cells.append(f"{fmt(boot(thr)[0])} & {fmt(boot(rtt)[0])}")
        lines.append(f"Pooled ({'DL' if direction == 'downlink' else 'UL'}) & {cells[0]} & {cells[1]} && "
                     f"{cells[2]} & {cells[3]} && & \\\\")
    return "\n".join(lines)


def replay_table(data) -> str:
    lines = []
    for loc in CITIES:
        row = []
        for core in ("qa2c", "a2c"):
            thr = seed_means(data, "2_replay_train", 2, loc, "downlink", core)
            rtt = seed_means(data, "2_replay_train", 2, loc, "downlink", core, metric="rtt")
            row.append(f"{fmt(float(np.median(thr)))} & {int(np.sum(np.asarray(thr) > 0))}/5 & {fmt(float(np.median(rtt)))}")
        lines.append(f"{TEX_CITY.get(loc, loc)} & {row[0]} & {row[1]} \\\\")
    return "\n".join(lines)


def headline(data, budgets, points) -> dict:
    h = {}
    for tier in (1, 2):
        for direction in DIRECTIONS:
            for core in ("qa2c", "a2c"):
                thr = [v for loc in CITIES for v in seed_means(data, "1a_full", tier, loc, direction, core)]
                rtt = [v for loc in CITIES for v in seed_means(data, "1a_full", tier, loc, direction, core, metric="rtt")]
                h[f"t{tier}_{direction}_{core}"] = dict(thr=boot(thr), rtt=boot(rtt), positive=f"{sum(np.asarray(thr) > 0)}/{len(thr)}",
                    gates=sum(gate(data, tier, loc, direction, core, budgets) for loc in CITIES))
            diff = [a - b for loc in CITIES for a, b in zip(seed_means(data, "1a_full", tier, loc, direction, "qa2c"),
                                                            seed_means(data, "1a_full", tier, loc, direction, "a2c"))]
            diff_rtt = [a - b for loc in CITIES for a, b in zip(seed_means(data, "1a_full", tier, loc, direction, "qa2c", metric="rtt"),
                                                                seed_means(data, "1a_full", tier, loc, direction, "a2c", metric="rtt"))]
            h[f"t{tier}_{direction}_qa2c_minus_a2c"] = dict(thr=boot(diff), rtt=boot(diff_rtt), positive=f"{sum(np.asarray(diff) > 0)}/{len(diff)}")
    for direction in DIRECTIONS:
        for other in ("qa2c", "a2c"):
            diff = [a - b for loc in CITIES for a, b in zip(seed_means(data, "1c_qdqn", 1, loc, direction, "qdqn"),
                                                            seed_means(data, "1a_full", 1, loc, direction, other))]
            h[f"qdqn_minus_{other}_{direction}"] = dict(thr=boot(diff), positive=f"{sum(np.asarray(diff) > 0)}/{len(diff)}")
        h[f"qdqn_{direction}"] = {k: v for k, v in points[("qdqn", direction)].items()}
        for core in ("qa2c", "a2c"):
            for variant in ("telemetry", "telemetry_queue"):
                for metric in ("thr", "rtt"):
                    diffs = [f - r for loc in CITIES for f, r in zip(
                        seed_means(data, "1a_full", 1, loc, direction, core, "full", metric),
                        seed_means(data, "1b_ablation", 1, loc, direction, core, variant, metric))]
                    h[f"abl_{direction}_{core}_{variant}_{metric}"] = dict(
                        diff=boot(diffs), identical=int(np.sum(np.abs(diffs) < 1e-9)), n=len(diffs))
    for core in ("qa2c", "a2c"):
        thr = [v for loc in CITIES for v in seed_means(data, "2_replay_train", 2, loc, "downlink", core)]
        rtt = [v for loc in CITIES for v in seed_means(data, "2_replay_train", 2, loc, "downlink", core, metric="rtt")]
        h[f"replay_train_{core}"] = dict(thr=boot(thr), rtt=boot(rtt), positive=f"{sum(np.asarray(thr) > 0)}/{len(thr)}",
                                         exact_zero=int(np.sum(np.abs(thr) < 1e-9)))
    rtx = {f"{d}_{t}": float(np.median([v for loc in CITIES for v in seed_means(data, "1a_full", t, loc, d, "qa2c", metric="rtx")]))
           for d in DIRECTIONS for t in (1, 2)}
    h["rtx_qa2c_pooled"] = rtx
    h["rtx_sydney_downlink"] = {t: float(np.median(seed_means(data, "1a_full", t, "Sydney", "downlink", "qa2c", metric="rtx"))) for t in (1, 2)}
    return h


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "outputs" / "rq_study" / "final-v3")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    MODEL_ERROR.update(load_model_error(TIER3_PATH))
    calibration = json.loads(CALIBRATION.read_text())
    budgets = {loc: {d: calibration[loc][d]["stock_rtt_p90_run_half_iqr_ms"] for d in DIRECTIONS} for loc in CITIES}
    data = load(args.study)
    fig_tiers(data, args.figures / "v3_tier_deltas.png")
    fig_ablation(data, args.figures / "v3_ablation.png")
    points = fig_operating_points(data, args.figures / "v3_operating_points.png")
    (args.study / "table_core.tex").write_text(core_table(data, budgets) + "\n")
    (args.study / "table_replay_train.tex").write_text(replay_table(data) + "\n")
    (args.study / "headline.json").write_text(json.dumps(headline(data, budgets, points), indent=1))
    print("ok")


if __name__ == "__main__":
    main()
