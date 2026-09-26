from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION = ROOT / "qbbr" / "data" / "calibrated" / "per_location_constants_v14.json"
CITIES = ["Tokyo", "SaoPaulo", "Ohio", "London", "Mumbai", "Sydney"]
CITY_LABEL = {"SaoPaulo": "Sao Paulo"}
DIRECTION = "downlink"  # set from --direction; a study tree may hold both directions
ARMS = [("a2c", "Classical A2C-BBR"), ("qa2c", "Hybrid QA2C-BBR"), ("stock", "Simulated stock proxy")]
COLOR = {"a2c": "#4f8fcf", "qa2c": "#f0a04b", "stock": "#35a893"}
EDGE = "#4d4d4d"
PANELS = [("throughput_mbps", "Mbps", "Throughput"),
          ("retransmits", "count per second", "Retransmissions"),
          ("rtt_ms", "ms", "RTT"),
          ("queue_ms", "ms", "Queueing delay (RTT - propagation RTT)"),
          ("rttvar_ms", "ms", "RTT Variance")]


def per_second(rows: list[dict], base_rtt_ms: float) -> dict[str, list[float]]:
    bins: dict[int, dict[str, float]] = {}
    elapsed, srtt, rttvar = 0.0, None, None
    for row in rows:
        rtt = row["rtt_ms"]
        if srtt is None:
            srtt, rttvar = rtt, rtt / 2.0
        else:
            rttvar = 0.75 * rttvar + 0.25 * abs(srtt - rtt)
            srtt = 0.875 * srtt + 0.125 * rtt
        dt = row["duration_s"]
        cursor, end = elapsed, elapsed + dt
        while cursor < end - 1e-9:
            second = int(np.floor(cursor))
            slice_end = min(end, second + 1.0)
            span = slice_end - cursor
            b = bins.setdefault(second, dict(bytes=0.0, rtx=0.0, rtt=0.0, var=0.0, time=0.0))
            b["bytes"] += row["delivered_bytes"] * span / dt
            b["rtx"] += row["retransmits"] * span / dt
            b["rtt"] += rtt * span
            b["var"] += rttvar * span
            b["time"] += span
            cursor = slice_end
        elapsed = end
    out = {key: [] for key, _, _ in PANELS}
    for second in sorted(bins):
        b = bins[second]
        if b["time"] < 0.5:  # ragged final partial second
            continue
        out["throughput_mbps"].append(b["bytes"] * 8.0 / b["time"] / 1e6)
        out["retransmits"].append(b["rtx"] / b["time"])
        out["rtt_ms"].append(b["rtt"] / b["time"])
        out["rttvar_ms"].append(b["var"] / b["time"])
        out["queue_ms"].append(max(b["rtt"] / b["time"] - base_rtt_ms, 0.0))
    return out


def _check(replayed: dict, stored: dict, what: str) -> None:
    for key in ("throughput_mbps", "rtt_p90_ms"):
        if not np.isclose(replayed[key], stored[key], rtol=1e-6, atol=1e-6):
            raise ValueError(f"Replay of {what} does not reproduce {key}: "
                             f"{replayed[key]} vs stored {stored[key]}")


def replay_job(args: tuple[str, str]) -> dict:
    import torch
    from qbbr.study.runner import Stock, build_agent, make_env, rollout

    result_path, calibration_path = args
    result = json.loads(Path(result_path).read_text())
    study, job = result["identity"]["protocol"], result["job"]
    calibration = json.loads(Path(calibration_path).read_text())
    saved = torch.load(Path(result_path).with_name("checkpoint.pt"), map_location="cpu", weights_only=False)
    if saved["signature"] != result["signature"]:
        raise ValueError(f"Checkpoint and result disagree: {result_path}")
    agent = build_agent(study, job["core"], job["seed"])
    agent.load_training_state_dict(saved["agent"])
    series = {"job": job, "policy": {}, "stock": {}}
    for stored in result["evaluation"]:
        seed = stored["holdout_seed"]
        env = make_env(study, job, calibration)
        base = env.env.params.rtt_rtp_s * 1000.0
        learned = rollout(agent, env, seed, job["core"],
                          study["qdqn"]["history_window"], keep_intervals=True)
        _check(learned, stored["policy"], f"{job} seed {seed}")
        series["policy"][seed] = per_second(learned["intervals"], base)
        # Stock depends only on (location, direction, seed); replayed per job
        # so each worker is independent, deduplicated when pooled.
        stock = rollout(Stock(), make_env(study, job, calibration), seed, keep_intervals=True)
        _check(stock, stored["stock"], f"stock {job['location']} seed {seed}")
        series["stock"][seed] = per_second(stock["intervals"], base)
    return series


def collect(study_dir: Path, calibration: Path, workers: int) -> dict:
    cache = study_dir / f"per_second_series_{DIRECTION}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    paths = sorted(study_dir.glob(f"*__{DIRECTION}__*/result.json"))
    with Pool(workers) as pool:
        replays = pool.map(replay_job, [(str(p), str(calibration)) for p in paths], chunksize=1)
    data: dict = {city: {arm: {} for arm, _ in ARMS} for city in CITIES}
    for replay in replays:
        job = replay["job"]
        data[job["location"]][job["core"]][str(job["seed"])] = replay["policy"]
        data[job["location"]]["stock"]["shared"] = replay["stock"]
    cache.write_text(json.dumps(data))
    return json.loads(cache.read_text())  # seed keys as strings, as a cached run sees them


def pooled(data: dict, city: str, arm: str, key: str) -> np.ndarray:
    return np.concatenate([np.asarray(series[key]) for by_seed in data[city][arm].values()
                           for series in by_seed.values()])


def paired_gains(study_dir: Path, city: str, arm: str) -> np.ndarray:
    """Throughput gain over stock on the same forcing, one value per (training, holdout) seed."""
    return np.asarray([e["throughput_delta_pct"]
                       for p in sorted(study_dir.glob(f"*__{city}__{DIRECTION}__{arm}__*/result.json"))
                       for e in json.loads(p.read_text())["evaluation"]])


def _style(ax, title: str, unit: str) -> None:
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(unit)
    ax.set_xticks(range(len(CITIES)))
    ax.set_xticklabels([CITY_LABEL.get(c, c) for c in CITIES], rotation=25)
    ax.grid(axis="y", color="#e4e4e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _boxes(ax, groups: list[tuple[str, list[np.ndarray]]], width: float = 0.26, gap: float = 0.02) -> None:
    offset = (len(groups) - 1) / 2
    for i, (arm, values) in enumerate(groups):
        positions = [c + (i - offset) * (width + gap) for c in range(len(CITIES))]
        ax.boxplot(values, positions=positions, widths=width, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=COLOR[arm], edgecolor=EDGE, linewidth=0.8),
                   medianprops=dict(color="#1a1a19", linewidth=1.4),
                   whiskerprops=dict(color=EDGE, linewidth=0.8),
                   capprops=dict(color=EDGE, linewidth=0.8))


def box_grid(data: dict, study_dir: Path, out: Path, note: str) -> None:
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (key, unit, title) in zip(axes.flat, PANELS):
        _boxes(ax, [(arm, [pooled(data, city, arm, key) for city in CITIES]) for arm, _ in ARMS])
        _style(ax, title, unit)
    ax = axes.flat[-1]
    _boxes(ax, [(arm, [paired_gains(study_dir, city, arm) for city in CITIES]) for arm in ("a2c", "qa2c")],
           width=0.36)
    ax.axhline(0, color=COLOR["stock"], linewidth=1.6, label="stock BBR-v3 (reference)")
    _style(ax, "Throughput gain over stock, paired per seed", "%")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[a], edgecolor=EDGE) for a, _ in ARMS]
    fig.suptitle(f"{'Download' if DIRECTION == 'downlink' else 'Upload'}, simulated Starlink paths (held-out seeds)", fontsize=15, y=0.985)
    fig.legend(handles, [label for _, label in ARMS], loc="upper center", ncol=3, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, 0.955))
    fig.text(0.5, 0.01, "\n".join(textwrap.wrap(note, 170)), ha="center", va="bottom", fontsize=9.5,
             color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.925))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def time_series(data: dict, out: Path, key: str, unit: str, title: str,
                train_seed: str = "0", holdout: str = "1000", seconds: int = 50) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, city in zip(axes.flat, ["Sydney", "Tokyo", "London", "SaoPaulo", "Ohio", "Mumbai"]):
        for arm, label in ARMS:
            by_seed = data[city][arm]["shared" if arm == "stock" else train_seed]
            y = by_seed[holdout][key][:seconds]
            # A2C and QA2C often coincide; A2C is drawn wide underneath and
            # QA2C dashed on top so both stay visible where they overlap.
            style = dict(a2c=dict(linewidth=3.0, zorder=1), qa2c=dict(linewidth=1.5, linestyle="--", zorder=3),
                         stock=dict(linewidth=1.5, zorder=2))[arm]
            ax.plot(range(1, len(y) + 1), y, color=COLOR[arm], label=label, **style)
        ax.set_title(CITY_LABEL.get(city, city))
        ax.set_xlabel("Test interval (s)")
        ax.set_ylabel(unit)
        ax.grid(color="#e4e4e0", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(f"{title}: first {seconds} held-out intervals "
                 f"(training seed {train_seed}, holdout seed {holdout}, simulator-proxy)", fontsize=14, y=0.985)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.945))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def summary_table(study_dir: Path, out: Path) -> list[dict]:
    """Paired policy-minus-stock deltas from result.json, per city and core."""
    rows = []
    for city in CITIES:
        for arm in ("a2c", "qa2c"):
            cells = [json.loads(p.read_text()) for p in sorted(study_dir.glob(f"*__{city}__{DIRECTION}__{arm}__*/result.json"))]
            per_seed = [dict(
                thr=np.mean([e["throughput_delta_pct"] for e in r["evaluation"]]),
                rtt=np.mean([e["rtt_p90_delta_ms"] for e in r["evaluation"]]),
                rtx=np.mean([e["retransmits_delta_per_s"] for e in r["evaluation"]]),
            ) for r in cells]
            rows.append(dict(
                city=city, core=arm, seeds=len(per_seed),
                throughput_delta_pct_median=float(np.median([s["thr"] for s in per_seed])),
                throughput_positive_seeds=int(sum(s["thr"] > 0 for s in per_seed)),
                rtt_p90_delta_ms_median=float(np.median([s["rtt"] for s in per_seed])),
                retransmits_delta_per_s_median=float(np.median([s["rtx"] for s in per_seed])),
            ))
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    parser.add_argument("--note", default="Simulator-proxy evidence (fluid model calibrated on measured Starlink "
                        "traces). Per-second samples, 5 training seeds x 10 held-out seeds per box; whiskers 1.5 IQR, "
                        "no outliers drawn. Bottom-right: one point per (training seed, holdout seed) pair, policy vs stock on the "
                        "same capacity forcing. Congestion and receiver windows are not modelled by the simulator.")
    parser.add_argument("--direction", choices=("downlink", "uplink"), default="downlink")
    args = parser.parse_args()
    global DIRECTION
    DIRECTION = args.direction

    data = collect(args.study, args.calibration, args.workers)
    tag = f"{args.study.name}_{DIRECTION}"
    args.figures.mkdir(exist_ok=True)
    box_grid(data, args.study, args.figures / f"comparison_boxes_{tag}.png", args.note)
    for key, unit, title in PANELS:
        time_series(data, args.figures / f"comparison_series_{key}_{tag}.png", key, unit, title)
    rows = summary_table(args.study, args.study / f"summary_by_city_{DIRECTION}.csv")
    print(f"{'city':<9} {'core':<5} {'thr %':>7} {'+seeds':>6} {'dRTTp90':>8} {'dRtx/s':>7}")
    for r in rows:
        print(f"{r['city']:<9} {r['core']:<5} {r['throughput_delta_pct_median']:>+7.2f} "
              f"{r['throughput_positive_seeds']:>4}/{r['seeds']} {r['rtt_p90_delta_ms_median']:>+8.2f} "
              f"{r['retransmits_delta_per_s_median']:>+7.3f}")


if __name__ == "__main__":
    main()
