"""Run an iperf3 campaign over Starlink, protocol-identical to qbbr/data/raw."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PACKAGE_ROOT / "data" / "field"

# Read back from qbbr/data/raw start.test_start -- do not "improve" these.
REFERENCE_PROTOCOL = {"blksize": 131072, "duration_s": 300, "streams": 1, "omit": 0}


def detect_congestion_control() -> str:
    """What the local sending stack will actually use, as a dataset-style tag."""
    system = platform.system()
    if system == "Linux":
        try:
            value = Path("/proc/sys/net/ipv4/tcp_congestion_control").read_text().strip()
            return value or "unknown"
        except OSError:
            return "unknown"
    if system == "Darwin":
        # XNU ships CUBIC, NewReno and LEDBAT ("background"); never BBR.
        return "cubic"
    return "unknown"


def check_environment(direction: str) -> str:
    if shutil.which("iperf3") is None:
        sys.exit("iperf3 not found. Install it first:  brew install iperf3   (or apt install iperf3)")
    cca = detect_congestion_control()
    print(f"[env] {platform.system()} {platform.release()} · sending stack uses '{cca}'")
    if cca != "bbr2":
        print(f"[env] NOTE: '{cca}' is not bbr2, the reference arm. These runs measure {cca} and")
        print(f"[env]       cannot stand in for bbr2. Runs will be labelled '{cca}' accordingly.")
    if direction == "downlink":
        print("[env] NOTE: downlink puts the SERVER on the sending side, so this client-side JSON")
        print("[env]       carries throughput but no rtt/retransmits. Capture the server's")
        print("[env]       `iperf3 -s -J --logfile ...` for sender-side statistics.")
    return cca


def run_once(server: str, port: int, direction: str, duration: int, out_path: Path) -> bool:
    command = [
        "iperf3", "-c", server, "-p", str(port), "-J",
        "-t", str(duration),
        "-l", str(REFERENCE_PROTOCOL["blksize"]),
        "-P", str(REFERENCE_PROTOCOL["streams"]),
        "-O", str(REFERENCE_PROTOCOL["omit"]),
    ]
    if direction == "downlink":
        command.append("-R")

    started = time.time()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.time() - started

    # iperf3 emits JSON even on a failed test; keep it either way so the
    # deviation is auditable rather than silently missing from the campaign.
    text = completed.stdout.strip()
    if not text:
        print(f"    FAILED after {elapsed:.0f}s: {completed.stderr.strip()[:160]}")
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        print(f"    FAILED: iperf3 produced unparseable output")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)

    error = payload.get("error")
    if error:
        print(f"    FAILED: {error}")
        return False
    end = payload.get("end", {})
    stream = (end.get("streams") or [{}])[0]
    side = stream.get("sender") if isinstance(stream.get("sender"), dict) else end.get("sum_received", {})
    mbps = float(side.get("bits_per_second", 0.0)) / 1e6
    print(f"    ok  {mbps:8.1f} Mbps  ({elapsed:.0f}s)  -> {out_path.name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", required=True, help="iperf3 server host or IP")
    parser.add_argument("--port", type=int, default=5201)
    parser.add_argument("--location", default="Sydney",
                        help="Label for these runs. Used in the filename and to pick the "
                             "reference city for comparison. Default Sydney.")
    parser.add_argument("--direction", choices=("downlink", "uplink"), default="uplink")
    parser.add_argument("--runs", type=int, default=10,
                        help="Reference dataset uses 10 runs per cell.")
    parser.add_argument("--duration", type=int, default=REFERENCE_PROTOCOL["duration_s"])
    parser.add_argument("--gap-s", type=float, default=60.0,
                        help="Idle time between runs, so one run's queue does not bleed into the next.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    args = parser.parse_args()

    cca = check_environment(args.direction)
    code = "REV" if args.direction == "downlink" else "FWD"
    total_min = args.runs * (args.duration + args.gap_s) / 60.0
    print(f"[plan] {args.runs} runs x {args.duration}s + {args.gap_s:.0f}s gap "
          f"= ~{total_min:.0f} min · {args.location} {args.direction} · cca={cca}")
    if args.dry_run:
        return 0

    target = args.out_dir / f"{args.direction}-sequential-logs" / args.location
    ok = 0
    for run in range(1, args.runs + 1):
        print(f"  run {run}/{args.runs}")
        name = f"{cca}_{args.location}__{code}_run{run}.json"
        if run_once(args.server, args.port, args.direction, args.duration, target / name):
            ok += 1
        if run < args.runs:
            time.sleep(args.gap_s)

    print(f"\n[done] {ok}/{args.runs} runs succeeded -> {target}")
    print(f"[next] python3 -m qbbr.scripts.compare_to_reference "
          f"--field {target} --location {args.location} --direction {args.direction}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
