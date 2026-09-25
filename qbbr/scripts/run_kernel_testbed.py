"""Kernel-in-the-loop: real Linux TCP + real BBR over a replayed Starlink capacity trace."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from qbbr.data.catalog import build_catalog, FileRecord
from qbbr.data.loader import load_trace
from qbbr.env.calibration import load_calibration
from qbbr.scripts.trace_replay import trace_forcing

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
NETWORK = "qbbr-testbed-net"
BUFFERS = [
    "--sysctl", "net.ipv4.tcp_rmem=4096 131072 67108864",
    "--sysctl", "net.ipv4.tcp_wmem=4096 16384 67108864",
]
IMAGE = "qbbr-testbed"


def _docker(*args, check=True, capture=True):
    return subprocess.run(["docker", *args], check=check,
                          capture_output=capture, text=True)


def preflight(cca: str) -> None:
    """Fail loudly before spending a 300 s run on a kernel that lacks the CCA."""
    if shutil.which("docker") is None:
        sys.exit("docker not found.")
    probe = _docker("run", "--rm", "--privileged", IMAGE, "sh", "-c",
                    "modprobe tcp_bbr 2>/dev/null; "
                    "sysctl -n net.ipv4.tcp_available_congestion_control", check=False)
    available = (probe.stdout or "").split()
    print(f"[preflight] kernel offers: {' '.join(available) or '(nothing)'}")
    if cca not in available:
        sys.exit(f"[preflight] '{cca}' is not available in this kernel. Docker Desktop's "
                 f"LinuxKit kernel has no tcp_bbr; use a real Linux VM (colima).")


def write_schedule(forcing: dict, path: Path) -> None:
    times = np.asarray(forcing["times_s"], dtype=float)
    capacity = np.asarray(forcing["capacity_bytes_s"], dtype=float) * 8.0
    with path.open("w") as handle:
        for t, rate in zip(times - times[0], capacity):
            handle.write(f"{t:.2f} {rate:.0f}\n")


def verify_schedule_visible(schedule: Path) -> None:
    """Confirm the container can actually READ the schedule before running."""
    probe = _docker("run", "--rm", "-v", f"{schedule}:/schedule.txt:ro", IMAGE,
                    "sh", "-c", "wc -l < /schedule.txt 2>/dev/null", check=False)
    digits = [int(tok) for tok in (probe.stdout or "").split() if tok.isdigit()]
    lines = max(digits) if digits else 0
    if lines < 2:
        sys.exit(f"[schedule] container sees {lines} lines in {schedule}. "
                 f"It must live under a path the VM mounts (colima mounts $HOME only).")
    print(f"[schedule] container reads {lines} steps")


def bottleneck_limit_packets(median_capacity_bps: float, rtt_ms: float) -> int:
    """Size the bottleneck queue to ONE bandwidth-delay product."""
    bdp_bytes = median_capacity_bps / 8.0 * rtt_ms / 1000.0
    return max(100, int(round(bdp_bytes / 1448.0)))


def run_once(schedule: Path, rtt_ms: float, duration: int, cca: str, out_json: Path,
             limit_packets: int = 20000, max_queue_ms: float = 0.0) -> dict:
    _docker("network", "rm", NETWORK, check=False)
    _docker("network", "create", NETWORK)
    server = client = None
    try:
        _docker(
            "run", "-d", "--rm", "--cap-add=NET_ADMIN", "--network", NETWORK, *BUFFERS,
            "--name", "qbbr-rcv", IMAGE, "sh", "-c",
            "ethtool -K eth0 gro off >/dev/null 2>&1; "
            f"tc qdisc add dev eth0 root netem delay {rtt_ms:.1f}ms limit 100000 && exec iperf3 -s -J"
        )
        time.sleep(3)

        result = _docker(
            "run", "--rm", "--privileged", "--network", NETWORK, *BUFFERS,
            "-v", f"{schedule}:/schedule.txt:ro", "-e", f"RTT_MS={rtt_ms:.1f}",
            "-e", f"NETEM_LIMIT={limit_packets}", "-e", f"MAX_QUEUE_MS={max_queue_ms:.1f}",
            IMAGE, "sh", "-c",
            "modprobe tcp_bbr >/dev/null 2>&1; "
            f"sysctl -w net.ipv4.tcp_congestion_control={cca} >/dev/null 2>&1; "
            "/shape.sh > /shape.log 2>&1 & "
            "sleep 1; "
            f"iperf3 -c qbbr-rcv -t {duration} -l 131072 -P 1 -O 0 --json --get-server-output",
            check=False)
        payload = result.stdout
        brace = payload.find("{")
        payload = payload[brace:] if brace >= 0 else ""
        # No fallback: a run without sender-side stats is refused, not patched.
        try:
            parsed = json.loads(payload)
            if parsed.get("error"):
                raise ValueError(parsed["error"])
            stream = parsed["intervals"][min(3, len(parsed["intervals"]) - 1)]["streams"][0]
            if stream.get("sender") is not True:
                raise ValueError("JSON is not the sending side")
        except Exception as error:
            raise RuntimeError(f"no usable sender-side JSON: {error}; "
                               f"stderr: {result.stderr.strip()[:200]}") from error
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(payload)
        return parsed
    finally:
        _docker("rm", "-f", "qbbr-rcv", check=False)
        _docker("network", "rm", NETWORK, check=False)


def check_emulated_rtt(stats: dict, rtt_ms: float) -> bool:
    """Refuse a run whose emulated round trip is not the configured one."""
    floor = stats.get("rtt_p1_ms", float("nan"))
    ratio = floor / rtt_ms if rtt_ms else float("nan")
    ok = 0.85 <= ratio <= 1.6
    tag = "ok" if ok else "INVALID -- emulated RTT does not match the path"
    print(f"    [rtt-check] floor {floor:.1f} ms vs configured {rtt_ms:.1f} ms  ratio {ratio:.2f}  {tag}")
    return ok


def summarise(payload: dict) -> dict:
    """Throughput from the RECEIVER (wire rate); RTT and retransmits from the SENDER."""
    sender_rtt, sender_retx = [], []
    for interval in payload.get("intervals", []):
        streams = interval.get("streams") or []
        if streams and not streams[0].get("omitted"):
            if streams[0].get("rtt") is not None:
                sender_rtt.append(streams[0]["rtt"] / 1000.0)
            if streams[0].get("retransmits") is not None:
                sender_retx.append(streams[0]["retransmits"])
    receiver = payload.get("server_output_json") or {}
    wire = [i["streams"][0]["bits_per_second"] / 1e6 for i in receiver.get("intervals", [])
            if i.get("streams") and not i["streams"][0].get("omitted")]
    if not wire or not sender_rtt:
        return {}
    wire = np.asarray(wire, dtype=float)
    rtt = np.asarray(sender_rtt, dtype=float)
    return {
        "throughput_mbps_mean": float(wire.mean()),
        "throughput_per_second": [round(float(x), 3) for x in wire],
        "rtt_per_second": [round(float(x), 3) for x in rtt],
        "rtt_p1_ms": float(np.percentile(rtt, 1)),
        "rtt_median_ms": float(np.median(rtt)),
        "rtt_p90_ms": float(np.percentile(rtt, 90)),
        "rtt_p99_ms": float(np.percentile(rtt, 99)),
        "retransmits_total": float(np.sum(sender_retx)) if sender_retx else 0.0,
        "stall_seconds": int((wire < 1.0).sum()),
        "intervals": int(len(wire)),
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--locations", nargs="+", default=["Sydney"],
                        help="Runs are sequential on purpose: two shaped flows in one VM "
                             "contend for CPU and the netem path, which would inflate RTT "
                             "and depress throughput -- the very quantities being measured.")
    parser.add_argument("--direction", default="downlink", choices=("downlink", "uplink"))
    parser.add_argument("--buffers", nargs="+", default=["sim"],
                        help="Bottleneck queue depths to sweep. 'sim' = the simulator's "
                             "probe_max_queue_delay_ms clamp; '<x>bdp' = x times RTT_min of "
                             "queueing delay. Depth is an UNMEASURED property of the real "
                             "Starlink path, so it is swept rather than calibrated.")
    parser.add_argument("--cca", default="bbr")
    parser.add_argument("--trace-cca", default="bbr", help="Which recorded runs supply the schedule.")
    parser.add_argument("--runs", nargs="+", type=int, default=[1])
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--calibration", type=Path,
                        default=PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants_v14.json")
    parser.add_argument("--out", type=Path, default=PACKAGE_ROOT.parent / "reports" / "kernel_testbed.json")
    args = parser.parse_args()

    preflight(args.cca)
    catalog = build_catalog(PACKAGE_ROOT / "data" / "raw")
    all_calibration = load_calibration(args.calibration)

    results = []
    for location in args.locations:
      calibration = all_calibration[location][args.direction]
      rtt_ms = float(calibration["RTT_min_ms"])
      print(f"\n=== {location} {args.direction} · rtt_min {rtt_ms:.0f} ms ===")
      for run in args.runs:
          rows = catalog[(catalog["location"] == location) & (catalog["direction"] == args.direction)
                         & (catalog["category"].str.contains("sequential"))
                         & (catalog["cca"] == args.trace_cca) & (catalog["run"] == run)]
          if rows.empty:
              print(f"  run{run}: no such trace, skipped")
              continue
          row = rows.iloc[0]
          trace = load_trace(FileRecord(path=Path(row["path"]), category=row["category"],
                                        direction=row["direction"], location=row["location"],
                                        cca=row["cca"], run=int(row["run"])))
          forcing = trace_forcing(trace.intervals)

          # Under the repo (inside $HOME) so the colima VM can see it.
          schedule_dir = (args.out.parent / "kernel_testbed_logs" / "schedules").resolve()
          schedule_dir.mkdir(parents=True, exist_ok=True)
          # Absolute: docker bind mounts reject relative paths.
          schedule = (schedule_dir / f"{location}_{args.direction}_run{run}.txt").resolve()
          write_schedule(forcing, schedule)
          verify_schedule_visible(schedule)

          print(f"  run{run}: schedule {len(forcing['times_s'])} steps · "
                f"median {np.median(forcing['capacity_bytes_s'])*8/1e6:.0f} Mbps · rtt_min {rtt_ms:.0f} ms")
          out_json = args.out.parent / "kernel_testbed_logs" / f"{args.cca}_{location}__run{run}.json"
          limit = bottleneck_limit_packets(float(np.median(forcing["capacity_bytes_s"])) * 8.0, rtt_ms)
          print(f"    [buffer] 1 BDP = {limit} packets")
          for buffer in args.buffers:
            if buffer == "sim":
                max_queue_ms = float(calibration["probe_max_queue_delay_ms"])
            elif buffer.endswith("bdp"):
                max_queue_ms = float(buffer[:-3]) * rtt_ms
            else:
                raise SystemExit(f"unknown buffer spec {buffer!r}")
            print(f"    [buffer {buffer}] max queueing delay {max_queue_ms:.1f} ms", flush=True)
            buffer_json = out_json.with_name(out_json.stem + f"__{buffer}.json")
            payload = run_once(schedule, rtt_ms, args.duration, args.cca, buffer_json,
                               limit_packets=limit, max_queue_ms=max_queue_ms)
            stats = summarise(payload)
            if not stats:
                print(f"  run{run} [{buffer}]: no usable intervals — check {buffer_json}")
                continue
            stats["emulated_rtt_valid"] = check_emulated_rtt(stats, rtt_ms)
            print(f"  run{run} [{buffer}]: wire {stats['throughput_mbps_mean']:.1f} Mbps · "
                  f"rtt p50/p90 {stats['rtt_median_ms']:.0f}/{stats['rtt_p90_ms']:.0f} ms · "
                  f"stalls {stats['stall_seconds']} · retx {stats['retransmits_total']:.0f} | "
                  f"real trace {forcing['realised_mbps_mean']:.1f} Mbps · "
                  f"{forcing['realised_rtt_p90_ms']:.1f} ms", flush=True)
            results.append({
                "location": location, "direction": args.direction, "run": run,
                "cca": args.cca, "trace_cca": args.trace_cca,
                "buffer": buffer, "max_queue_ms": max_queue_ms,
                "kernel": stats,
                "trace_realised_mbps_mean": forcing["realised_mbps_mean"],
                "trace_realised_rtt_p90_ms": forcing["realised_rtt_p90_ms"],
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "evidence_tier": "kernel_in_the_loop_emulated_capacity",
        "caveat": "Real Linux TCP and real BBR over a tc-replayed capacity schedule. "
                  "Emulated capacity, not a satellite: no weather loss, no LEO jitter, "
                  "no ground-station queueing. Replaces the transport model, not the field trial.",
        "rows": results,
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
