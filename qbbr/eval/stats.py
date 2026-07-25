"""Statistical rigor: median/IQR summaries + Mann-Whitney U tests across runs.

Per main.tex's "Statistical Rigor and Ablations": each configuration-location
pair receives >=10 runs at varied times of day, reported as median and IQR
with Mann-Whitney U tests (p < 0.05).

Not yet implemented.
"""
from __future__ import annotations

import pandas as pd


def summarize_median_iqr(samples: pd.Series) -> dict[str, float]:
    raise NotImplementedError("Median/IQR summary not yet implemented.")


def mann_whitney_test(a: pd.Series, b: pd.Series) -> dict[str, float]:
    raise NotImplementedError("Mann-Whitney U test not yet implemented.")
