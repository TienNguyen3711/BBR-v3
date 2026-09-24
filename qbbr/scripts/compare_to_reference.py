"""Compare a fresh Starlink campaign against the reference dataset.

WHAT THIS CAN AND CANNOT SETTLE.  A new campaign differs from qbbr/data/raw in
two ways at once -- a different link (different terminal, sky, time, load) and
possibly a different congestion control algorithm. One comparison cannot
separate them, so this script always reports TWO:

  vs --reference-cca   (default bbr2)  the arm you asked about
  vs the SAME cca as the field runs    the control

Read them together. If the field runs match the reference on their own
algorithm but differ from bbr2, the gap is the ALGORITHM. If they differ from
their own algorithm too, the gap is the LINK, and the bbr2 comparison cannot be
interpreted at all until that is explained.

bbr2 is the reference arm for this project. Real BBRv1 stalls to zero throughput
in 47-57% of one-second samples on five of six uplinks; bbr2 does not (2-7%), so
it is the only BBR variant in the dataset that behaves sanely on the uplink.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qbbr.data.catalog import build_catalog, FileRecord
from qbbr.data.loader import load_trace

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_ROOT = PACKAGE_ROOT / "data" / "raw"


def _run_means(paths) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-run mean throughput (Mbps) and mean RTT (ms); how many had RTT."""
    throughput, rtt, with_rtt = [], [], 0
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text())
        except Exception:
            continue
        rows = []
        for interval in payload.get("intervals", []):
            streams = interval.get("streams") or []
            if not streams:
                continue
            s = streams[0]
            if s.get("omitted"):
                continue
            rows.append((s.get("bits_per_second"), s.get("rtt")))
        if not rows:
            continue
        frame = pd.DataFrame(rows, columns=["bps", "rtt_us"])
        bps = pd.to_numeric(frame["bps"], errors="coerce").dropna()
        if bps.empty:
            continue
        throughput.append(bps.mean() / 1e6)
        r = pd.to_numeric(frame["rtt_us"], errors="coerce").dropna()
        if not r.empty:
            rtt.append(r.mean() / 1000.0)
            with_rtt += 1
    return np.asarray(throughput), np.asarray(rtt), with_rtt


def _reference_runs(location: str, direction: str, cca: str):
    catalog = build_catalog(REFERENCE_ROOT)
    rows = catalog[(catalog["location"] == location) & (catalog["direction"] == direction)
                   & (catalog["category"].str.contains("sequential")) & (catalog["cca"] == cca)]
    return [row["path"] for _, row in rows.iterrows()]


def bootstrap_difference(a: np.ndarray, b: np.ndarray, resamples: int = 10_000,
                         seed: int = 20260905) -> dict:
    """CI of mean(a) - mean(b). Same procedure and seed as the simulator gates."""
    rng = np.random.RandomState(seed)
    draws = (rng.choice(a, (resamples, len(a)), replace=True).mean(axis=1)
             - rng.choice(b, (resamples, len(b)), replace=True).mean(axis=1))
    return {"point": float(a.mean() - b.mean()),
            "lower": float(np.quantile(draws, 0.025)),
            "upper": float(np.quantile(draws, 0.975))}


def _report(label: str, field: np.ndarray, reference: np.ndarray) -> None:
    if reference.size < 2:
        print(f"\n  vs {label}: not enough reference runs")
        return
    ci = bootstrap_difference(field, reference)
    relative = ci["point"] / reference.mean() * 100.0
    separates = ci["lower"] > 0 or ci["upper"] < 0
    print(f"\n  vs {label}  (n={reference.size}, mean {reference.mean():.1f} Mbps, "
          f"CV {reference.std(ddof=1)/reference.mean():.3f})")
    print(f"     difference {ci['point']:+.1f} Mbps ({relative:+.1f}%)  "
          f"CI95 [{ci['lower']:+.1f}, {ci['upper']:+.1f}]")
    print(f"     {'DIFFERENT -- CI excludes zero' if separates else 'indistinguishable at 95%'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--field", type=Path, required=True,
                        help="Directory of the new campaign's JSON logs.")
    parser.add_argument("--location", default="Sydney",
                        help="Reference city to compare against. Default Sydney.")
    parser.add_argument("--direction", choices=("downlink", "uplink"), default="uplink")
    parser.add_argument("--reference-cca", default="bbr2",
                        help="Reference arm. Default bbr2, the project's reference.")
    parser.add_argument("--no-control", action="store_true",
                        help="Report only the bbr2 comparison. Off by default: without the "
                             "control, a gap cannot be attributed to link or to algorithm.")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    field_paths = sorted(args.field.glob("*.json"))
    if not field_paths:
        return int(bool(print(f"No JSON logs under {args.field}")))
    field_cca = field_paths[0].name.split("_")[0]
    throughput, rtt, with_rtt = _run_means(field_paths)
    if throughput.size < 2:
        return int(bool(print("Need at least two usable field runs.")))

    print(f"FIELD  {args.location} {args.direction} · cca={field_cca} · n={throughput.size}")
    print(f"  throughput  mean {throughput.mean():.1f} Mbps  sd {throughput.std(ddof=1):.1f}  "
          f"CV {throughput.std(ddof=1)/throughput.mean():.3f}")
    if with_rtt:
        print(f"  rtt         mean {rtt.mean():.1f} ms  (from {with_rtt}/{throughput.size} runs)")
    else:
        print("  rtt         NOT REPORTED -- this side is not the sender, or the OS")
        print("              does not expose tcp_info. Throughput comparison only.")

    print(f"\nREFERENCE  qbbr/data/raw · {args.location} {args.direction}")
    asked, _r, _n = _run_means(_reference_runs(args.location, args.direction, args.reference_cca))
    _report(f"{args.reference_cca}  [requested]", throughput, asked)

    if field_cca != args.reference_cca and not args.no_control:
        control, _r2, _n2 = _run_means(_reference_runs(args.location, args.direction, field_cca))
        _report(f"{field_cca}  [control -- same algorithm as the field runs]", throughput, control)
        print(f"\n  How to read this: the {args.reference_cca} comparison mixes link and algorithm.")
        print(f"  The {field_cca} comparison isolates the LINK, because the algorithm matches.")
    elif field_cca == args.reference_cca:
        print(f"\n  Field and reference use the same algorithm, so the comparison isolates the link.")
    else:
        print(f"\n  Control suppressed (--no-control): this number mixes link and algorithm.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "location": args.location, "direction": args.direction,
            "field_cca": field_cca, "reference_cca": args.reference_cca,
            "field_runs": int(throughput.size),
            "field_mean_mbps": float(throughput.mean()),
            "field_cv": float(throughput.std(ddof=1) / throughput.mean()),
            "field_rtt_reported": bool(with_rtt),
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
