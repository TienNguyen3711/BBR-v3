from __future__ import annotations

from pathlib import Path

import pytest

_DATASET_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"


@pytest.fixture(scope="session")
def dataset_root() -> Path:
    if not _DATASET_ROOT.is_dir():
        pytest.skip(f"dataset not found at {_DATASET_ROOT}")
    return _DATASET_ROOT


@pytest.fixture(scope="session")
def sample_trace(dataset_root):
    """One real, clean, high-throughput/low-RTT trace: stock BBR-v3, Sydney downlink."""
    from qbbr.data.catalog import build_catalog, iter_file_records
    from qbbr.data.loader import load_trace

    df = build_catalog(dataset_root)
    row = df[(df.cca == "bbr") & (df.location == "Sydney") & (df.direction == "downlink")].iloc[0]
    record = next(iter_file_records(df[df.index == row.name]))
    return load_trace(record)


@pytest.fixture(scope="session")
def sample_calibration(dataset_root):
    from qbbr.env.calibration import compute_calibration

    return compute_calibration(dataset_root)
