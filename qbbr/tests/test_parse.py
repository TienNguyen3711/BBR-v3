from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from qbbr.data.parse import parse_iperf3_file, parse_iperf3_json

# Concrete examples of each quirk category, located by scanning the corpus.
_CLEAN = "uplink-competitive-logs/Mumbai/pcc_Mumbai__FWD_run9.json"
_TRAILING_ONLY = "downlink-competitive-logs/Mumbai/leocc_Mumbai__REV_run2.json"
_LEADING_AND_TRAILING = "downlink-competitive-logs/Mumbai/bbr2_Mumbai__REV_run7.json"
_ERROR_ONLY = "uplink-competitive-logs/Mumbai/vegas_Mumbai__FWD_run5.json"


def test_clean_file(dataset_root):
    result = parse_iperf3_file(dataset_root / _CLEAN)
    d = result.diagnostics
    assert not d.had_leading_junk
    assert not d.had_trailing_junk
    assert not d.had_error_field
    assert 300 <= d.n_intervals <= 319


def test_trailing_junk_file(dataset_root):
    result = parse_iperf3_file(dataset_root / _TRAILING_ONLY)
    d = result.diagnostics
    assert not d.had_leading_junk
    assert d.had_trailing_junk
    assert "intervals" in result.raw


def test_leading_and_trailing_junk_file(dataset_root):
    result = parse_iperf3_file(dataset_root / _LEADING_AND_TRAILING)
    d = result.diagnostics
    assert d.had_leading_junk
    assert d.had_trailing_junk
    assert "intervals" in result.raw


def test_error_field_file(dataset_root):
    result = parse_iperf3_file(dataset_root / _ERROR_ONLY)
    d = result.diagnostics
    assert d.had_error_field
    assert d.error_message is not None


def test_no_json_object_raises():
    with pytest.raises(ValueError):
        parse_iperf3_json("not json at all")


@pytest.mark.slow
def test_full_corpus_parses_without_exception(dataset_root):
    files = glob.glob(str(dataset_root / "**" / "*.json"), recursive=True)
    assert len(files) == 2160

    leading = trailing = errfield = clean = 0
    for fp in files:
        text = Path(fp).read_text()
        result = parse_iperf3_json(text)
        d = result.diagnostics
        assert 300 <= d.n_intervals <= 319
        if d.had_leading_junk:
            leading += 1
        if d.had_trailing_junk:
            trailing += 1
        if d.had_error_field:
            errfield += 1
        if not (d.had_leading_junk or d.had_trailing_junk or d.had_error_field):
            clean += 1

    assert leading == 95
    assert trailing == 1080
    assert errfield == 158
    assert clean == 1017
