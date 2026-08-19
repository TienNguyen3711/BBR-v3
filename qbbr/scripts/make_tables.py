from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_VALIDATION_CSV = PROJECT_ROOT / "outputs" / "simulator_validation.csv"
DEFAULT_ABLATION_GLOB = str(PROJECT_ROOT / "outputs" / "ablation" / "parallel_{core}_downlink_*.json")
DEFAULT_MAIN_TEX = PROJECT_ROOT / "main.tex"
OUT_DIR = PROJECT_ROOT / "outputs" / "tables"

# tab:fidelity's row order (matches main.tex; not alphabetical).
FIDELITY_LOCATION_ORDER = ["Sydney", "Ohio", "Tokyo", "London", "Mumbai", "SaoPaulo"]
FIDELITY_LABELS = {"SaoPaulo": "S\\~ao Paulo"}
QVC_LOCATION_ORDER = ["Sydney", "Ohio", "Tokyo", "London", "Mumbai", "SaoPaulo"]


def _latest(glob_pattern: str) -> Path:
    matches = sorted(glob.glob(glob_pattern))
    if not matches:
        raise FileNotFoundError(f"no files matching {glob_pattern}")
    return Path(matches[-1])


def make_tab_fidelity(validation_csv: Path) -> str:
    df = pd.read_csv(validation_csv)
    dl = df[df["direction"] == "downlink"].set_index("location")

    lines = []
    for location in FIDELITY_LOCATION_ORDER:
        row = dl.loc[location]
        label = FIDELITY_LABELS.get(location, location)
        lines.append(
            f"{label:11s} & {row['real_throughput_mbps_median']:.1f} & "
            f"{row['sim_throughput_mbps_median']:.1f} & {row['real_rtt_ms_median']:.1f}  & "
            f"{row['sim_rtt_ms_median']:.1f}  & {row['real_retransmits_per_s_median']:.1f} & "
            f"{row['sim_retransmits_per_s_median']:.1f}  \\\\"
        )
    return "\n".join(lines)


def make_tab_qvc(ablation_glob: str) -> str:
    from scipy import stats as scipy_stats

    by_core: dict[str, dict[str, list[float]]] = {}
    for core in ("classical", "quantum"):
        path = _latest(ablation_glob.format(core=core))
        data = json.loads(path.read_text())
        by_loc: dict[str, list[float]] = {}
        for r in data["results"]:
            by_loc.setdefault(r["location"], []).append(r["final_mean_reward"])
        by_core[core] = by_loc

    lines = []
    for location in QVC_LOCATION_ORDER:
        c = sorted(by_core["classical"][location])
        q = sorted(by_core["quantum"][location])
        c_med, q_med = statistics.median(c), statistics.median(q)
        c_iqr = statistics.quantiles(c, n=4)[2] - statistics.quantiles(c, n=4)[0]
        q_iqr = statistics.quantiles(q, n=4)[2] - statistics.quantiles(q, n=4)[0]
        _u, p = scipy_stats.mannwhitneyu(c, q, alternative="two-sided")
        label = "S\\~ao Paulo" if location == "SaoPaulo" else location
        lines.append(
            f"{label:11s} & {c_med:.4f} ({c_iqr:.4f}) & {q_med:.4f} ({q_iqr:.4f}) & {p:.3f} \\\\"
        )
    return "\n".join(lines)


def extract_main_tex_table_body(main_tex: Path, label: str) -> str | None:
    text = main_tex.read_text()
    marker = f"\\label{{{label}}}"
    idx = text.find(marker)
    if idx == -1:
        return None
    midrule = text.find("\\midrule", idx)
    bottomrule = text.find("\\bottomrule", idx)
    if midrule == -1 or bottomrule == -1:
        return None
    return text[midrule + len("\\midrule"): bottomrule].strip()


def check(label: str, generated: str, main_tex: Path) -> None:
    current = extract_main_tex_table_body(main_tex, label)
    if current is None:
        print(f"  [{label}] WARNING: could not find this table in {main_tex.name} to compare against")
        return
    # Loose comparison: strip whitespace differences, since hand-formatting
    # (column alignment spaces) legitimately differs from the generator's.
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    if norm(current) == norm(generated):
        print(f"  [{label}] OK: matches {main_tex.name}")
    else:
        print(f"  [{label}] DRIFT: {main_tex.name}'s current table body differs from the generated one "
              f"-- recalibration/retraining likely happened since the last manual paste")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validation-csv", type=Path, default=DEFAULT_VALIDATION_CSV)
    parser.add_argument("--ablation-glob", default=DEFAULT_ABLATION_GLOB)
    parser.add_argument("--main-tex", type=Path, default=DEFAULT_MAIN_TEX)
    parser.add_argument("--check", action="store_true", help="also diff against main.tex's current tables")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "tab:fidelity": make_tab_fidelity(args.validation_csv),
        "tab:qvc": make_tab_qvc(args.ablation_glob),
    }

    for label, body in tables.items():
        out_path = OUT_DIR / f"{label.replace(':', '_')}.tex"
        out_path.write_text(body + "\n")
        print(f"=== {label} -> {out_path.relative_to(PROJECT_ROOT)} ===")
        print(body)
        print()

    if args.check:
        print("--- drift check against main.tex ---")
        for label, body in tables.items():
            check(label, body, args.main_tex)


if __name__ == "__main__":
    main()
