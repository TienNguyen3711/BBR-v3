"""Reward contract for the successor QRL--BBR brain."""

from __future__ import annotations

import pandas as pd


def compute_throughput_only_reward(telemetry: pd.DataFrame) -> pd.Series:
    """Return delivered throughput in Mbps at every decision point."""

    if "bits_per_second" not in telemetry:
        raise KeyError("throughput-only reward requires a bits_per_second column")
    return telemetry["bits_per_second"].astype(float) / 1e6
