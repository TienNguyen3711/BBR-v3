"""Per-second throughput and RTT, from every source, on identical capacity forcing.

Distributions (CDFs, box plots) need the underlying samples, not the summary
statistics the replay and testbed reports keep. This collects them.

  real trace   data/raw iperf3 intervals          native 1 s
  real kernel  kernel_testbed_logs iperf3 JSON    native 1 s
  simulator    FluidSimEnv on the same forcing    RESAMPLED to 1 s

THE RESAMPLING IS NOT OPTIONAL.  The simulator steps once per decision interval,
which is 2 x RTT_min: ~60 ms on Sydney but ~520 ms on London and ~780 ms on
SaoPaulo. iperf3 reports once per second. Comparing distributions at different
resolutions manufactures a difference: finer samples resolve more of the queue
transient, so the finer-grained source shows a wider spread and a longer tail
purely from how it was sampled. Every simulator step is therefore accumulated
into 1-second bins -- delivered bytes summed, RTT time-weighted -- before any
comparison, so all three sources describe the same thing at the same timescale.

The environment is built exactly as trace_replay.py builds it (same calibration,
dynamics overrides, phase gate, capacity trace and aligned phase offset), so
these samples are the ones behind that report's summary numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qbbr.data.catalog import build_catalog, FileRecord
from qbbr.data.loader import load_trace
from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.scripts.trace_replay import align_phase_offset, trace_forcing, _load_agent
from qbbr.scripts.run_native_qa2c_successor import STOCK_ACTION

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ROOT = PACKAGE_ROOT.parent


def _iperf_series(intervals) -> tuple[list[float], list[float]]:
    frame = intervals[intervals.get("omitted") != True]  # noqa: E712
    thr = pd.to_numeric(frame["bits_per_second"], errors="coerce") / 1e6
    rtt = pd.to_numeric(frame["rtt_ms"], errors="coerce")
    keep = thr.notna() & rtt.notna()
    return thr[keep].round(3).tolist(), rtt[keep].round(3).tolist()


def _kernel_series(path: Path) -> tuple[list[float], list[float]]:
    """Per-second wire throughput (RECEIVER) and RTT (SENDER) from a clean log.

    The sender's bits_per_second is its socket-WRITE rate: with a 64 MB send
    buffer it read a flat ~177 Mbps with 0-Mbps stalls while the verified wire
    rate was 122-196. Wire throughput therefore comes from the receiver's
    intervals (server_output_json); RTT, which only the sender measures, comes
    from the sender. The two series are truncated to a common length.
    """
    payload = json.loads(path.read_text())
    rtt = [s["rtt"] / 1000.0 for i in payload.get("intervals", [])
           for s in (i.get("streams") or [])[:1]
           if not s.get("omitted") and s.get("rtt") is not None]
    receiver = payload.get("server_output_json") or {}
    thr = [s["bits_per_second"] / 1e6 for i in receiver.get("intervals", [])
           for s in (i.get("streams") or [])[:1]
           if not s.get("omitted") and s.get("bits_per_second") is not None]
    if not receiver:
        raise ValueError(f"{path.name}: no receiver-side JSON -- not a clean testbed log")
    n = min(len(thr), len(rtt))
    return [round(x, 3) for x in thr[:n]], [round(x, 3) for x in rtt[:n]]

def _sim_series(env, action_fn) -> tuple[list[float], list[float]]:
    """Step the simulator and accumulate into 1-second bins (see module docstring)."""
    state, done = env.reset(seed=0), False
    bins: dict[int, dict[str, float]] = {}
    elapsed = 0.0
    while not done:
        allowed = env.allowed_action_indices()
        state, _r, done, info = env.step(action_fn(state, allowed))
        dt = float(info["t_dec_s"])
        # Attribute the step to the second in which it ends; for steps longer
        # than a second (long-RTT paths) split bytes and RTT-time pro rata.
        start, end = elapsed, elapsed + dt
        cursor = start
        while cursor < end - 1e-9:
            second = int(np.floor(cursor))
            slice_end = min(end, second + 1.0)
            share = (slice_end - cursor) / dt
            bucket = bins.setdefault(second, {"bytes": 0.0, "rtt_time": 0.0, "time": 0.0})
            bucket["bytes"] += info["delivered_bytes"] * share
            bucket["rtt_time"] += info["rtt_ms"] * (slice_end - cursor)
            bucket["time"] += slice_end - cursor
            cursor = slice_end
        elapsed = end
    thr, rtt = [], []
    for second in sorted(bins):
        bucket = bins[second]
        if bucket["time"] < 0.5:        # drop a ragged final partial second
            continue
        thr.append(round(bucket["bytes"] * 8.0 / bucket["time"] / 1e6, 3))
        rtt.append(round(bucket["rtt_time"] / bucket["time"], 3))
    return thr, rtt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "tier1_v14_fidelity.yaml")
    parser.add_argument("--calibration", type=Path,
                        default=PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants_v14.json")
    parser.add_argument("--ckpt-root", type=Path, default=ROOT / "outputs" / "checkpoints_v14")
    parser.add_argument("--kernel-logs", type=Path, default=ROOT / "reports" / "kernel_testbed_logs")
    parser.add_argument("--locations", nargs="+",
                        default=["Sydney", "Tokyo", "Mumbai", "Ohio", "London", "SaoPaulo"])
    parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=0,
                        help="One deployed policy per core; see the pre-registration on seed choice.")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "per_second_series.json")
    parser.add_argument("--refresh-kernel-only", action="store_true",
                        help="Replace only the real_kernel series in an existing --out file from "
                             "clean testbed logs. The simulator and real-trace series are deterministic "
                             "and are left untouched, so the simulator is not re-run.")
    args = parser.parse_args()

    if args.refresh_kernel_only:
        payload = json.loads(args.out.read_text())
        refreshed = missing = 0
        for record in payload["series"]:
            path = args.kernel_logs / f"bbr_{record['location']}__run{record['run']}__sim.json"
            if path.exists():
                record["real_kernel"] = dict(zip(("throughput_mbps", "rtt_ms"), _kernel_series(path)))
                refreshed += 1
            else:
                record.pop("real_kernel", None)   # never keep a stale (retracted) series
                missing += 1
        payload["kernel_source"] = ("clean testbed: symmetric RTT, receiver-side throughput, "
                                    "GSO off, time-bounded bfifo at the simulator clamp")
        args.out.write_text(json.dumps(payload))
        print(f"refreshed real_kernel for {refreshed} traces, removed {missing} stale")
        return 0

    cfg = yaml.safe_load(args.config.read_text())
    calibration = load_calibration(args.calibration)
    overrides = cfg["simulator"]["dynamics_overrides"]
    gate = bool(cfg["simulator"].get("probe_bw_phase_gate", False))
    catalog = build_catalog(PACKAGE_ROOT / "data" / "raw")

    series = []
    for location in args.locations:
        for run in args.runs:
            rows = catalog[(catalog["location"] == location) & (catalog["direction"] == "downlink")
                           & (catalog["category"].str.contains("sequential"))
                           & (catalog["cca"] == "bbr") & (catalog["run"] == run)]
            if rows.empty:
                continue
            row = rows.iloc[0]
            trace = load_trace(FileRecord(path=Path(row["path"]), category=row["category"],
                                          direction=row["direction"], location=row["location"],
                                          cca=row["cca"], run=int(row["run"])))
            forcing = trace_forcing(trace.intervals)
            offset = align_phase_offset(forcing["times_s"], forcing["capacity_bytes_s"])

            def make_env():
                return FluidSimEnv(
                    location, "downlink", calibration, episode_s=forcing["duration_s"],
                    reward_mode="throughput_only", risk_mode=cfg["simulator"]["risk_mode"],
                    dynamics_overrides=overrides, probe_bw_phase_gate=gate,
                    capacity_trace=(forcing["times_s"], forcing["capacity_bytes_s"]),
                    phase_offset_s=offset,
                )

            record = {"location": location, "run": run}
            record["real_trace"] = dict(zip(("throughput_mbps", "rtt_ms"), _iperf_series(trace.intervals)))
            kernel_path = args.kernel_logs / f"bbr_{location}__run{run}__sim.json"
            if kernel_path.exists():
                record["real_kernel"] = dict(zip(("throughput_mbps", "rtt_ms"), _kernel_series(kernel_path)))
            record["sim_stock"] = dict(zip(("throughput_mbps", "rtt_ms"),
                                           _sim_series(make_env(), lambda s, a: STOCK_ACTION)))
            probe = make_env()
            for core in ("quantum", "classical"):
                model = _load_agent(cfg, probe, args.ckpt_root, location, "downlink", args.seed, core)
                policy = lambda s, a, m=model: m.act(s, a, deterministic=True, deployment=True)[0]
                record[f"sim_{core}"] = dict(zip(("throughput_mbps", "rtt_ms"), _sim_series(make_env(), policy)))
            series.append(record)
            sizes = {k: len(v["rtt_ms"]) for k, v in record.items() if isinstance(v, dict)}
            print(f"  {location:<9} run{run}  samples {sizes}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "resolution_s": 1.0,
        "note": "Simulator steps accumulated into 1 s bins to match iperf3's reporting interval.",
        "series": series,
    }))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
