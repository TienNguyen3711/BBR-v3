from __future__ import annotations

from qbbr.data.catalog import build_catalog, iter_file_records


def test_catalog_size(dataset_root):
    df = build_catalog(dataset_root)
    assert len(df) == 2160


def test_catalog_balanced_by_cca(dataset_root):
    df = build_catalog(dataset_root)
    counts = df.groupby("cca").size()
    assert set(counts.index) == {
        "bbr", "bbr1", "bbr2", "ccp", "cubic", "hybla", "leocc", "pcc", "vegas",
    }
    assert (counts == 240).all()


def test_catalog_ten_runs_per_group(dataset_root):
    df = build_catalog(dataset_root)
    group_sizes = df.groupby(["category", "direction", "location", "cca"]).size()
    assert (group_sizes == 10).all()


def test_iter_file_records_roundtrip(dataset_root):
    df = build_catalog(dataset_root)
    records = list(iter_file_records(df.head(5)))
    assert len(records) == 5
    for record in records:
        assert record.path.exists()
