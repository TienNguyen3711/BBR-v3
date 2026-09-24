"""The RQ aggregation must pair arms at matched forcing, or the contrast it
reports is dominated by capacity noise rather than by the arm."""
from __future__ import annotations

import json
from statistics import median

import pytest

from qbbr.scripts.report_rq_study import by_cell, describe, load_rows, paired


def _row(core, variant, seed, holdout, throughput, location="Sydney"):
    return {"location": location, "direction": "downlink", "core": core, "variant": variant,
            "seed": seed, "holdout_seed": holdout, "action_counts": {"2": 1},
            "throughput_delta_pct": throughput, "rtt_p90_delta_ms": 0.0,
            "retransmits_delta_per_s": 0.0}


def test_pairing_recovers_a_constant_effect_hidden_by_forcing_noise():
    """Each holdout seed shifts BOTH arms by a large shared amount.

    The true arm effect is +2.0 everywhere. Unpaired arm medians differ by
    that too here, but the paired estimate must be exact rather than merely
    close, because the shared per-seed term cancels term by term.
    """
    forcing_noise = {1000: -40.0, 1001: +5.0, 1002: +90.0}
    rows = []
    for holdout, shift in forcing_noise.items():
        rows.append(_row("qa2c", "full", 0, holdout, shift + 2.0))
        rows.append(_row("qa2c", "telemetry", 0, holdout, shift))

    differences = paired(rows, "throughput_delta_pct",
                         {"core": "qa2c", "variant": "full"},
                         {"core": "qa2c", "variant": "telemetry"})
    assert differences == {("Sydney", 0): 2.0}


def test_pairing_ignores_holdout_seeds_only_one_arm_has():
    rows = [_row("qa2c", "full", 0, 1000, 10.0), _row("qa2c", "full", 0, 1001, 99.0),
            _row("qa2c", "telemetry", 0, 1000, 4.0)]
    differences = paired(rows, "throughput_delta_pct",
                         {"core": "qa2c", "variant": "full"},
                         {"core": "qa2c", "variant": "telemetry"})
    assert differences == {("Sydney", 0): 6.0}  # 99.0 has no counterpart and is dropped


def test_pairing_never_crosses_cores_or_locations():
    rows = [_row("qa2c", "full", 0, 1000, 10.0),
            _row("a2c", "telemetry", 0, 1000, 1.0),
            _row("qa2c", "telemetry", 0, 1000, 4.0, location="Tokyo")]
    differences = paired(rows, "throughput_delta_pct",
                         {"core": "qa2c", "variant": "full"},
                         {"core": "qa2c", "variant": "telemetry"})
    assert differences == {}  # the only qa2c/telemetry row is a different location


def test_cells_aggregate_holdout_seeds_within_a_training_seed():
    rows = [_row("qa2c", "full", 0, 1000, 1.0), _row("qa2c", "full", 0, 1001, 3.0),
            _row("qa2c", "full", 1, 1000, 10.0)]
    cells = by_cell(rows, core="qa2c", variant="full")
    assert cells[("Sydney", 0)]["throughput_delta_pct"] == 2.0  # median of 1 and 3
    assert cells[("Sydney", 1)]["throughput_delta_pct"] == 10.0


def test_describe_counts_sign_and_flags_an_interval_clear_of_zero():
    summary = describe({("a", 0): 1.0, ("b", 0): 2.0, ("c", 0): 3.0}, "demo")
    assert summary["n"] == 3 and summary["positive"] == "3/3"
    assert summary["median"] == 2.0 and summary["excludes_zero"] is True

    mixed = describe({("a", 0): -5.0, ("b", 0): 0.5, ("c", 0): 4.0}, "demo")
    assert mixed["positive"] == "2/3" and mixed["excludes_zero"] is False


def test_describe_reports_absence_rather_than_inventing_a_number():
    assert describe({}, "demo")["n"] == 0


def test_load_rows_counts_missing_jobs(tmp_path):
    identity = {"protocol": {"study_id": "t"}}
    (tmp_path / "manifest.shard-1of1.json").write_text(json.dumps(
        {"identity": identity, "jobs": [{"a": 1}, {"a": 2}], "status": "partial", "completed": []}))
    directory = tmp_path / "synthetic__Sydney__downlink__qa2c__full__0"
    directory.mkdir()
    (directory / "result.json").write_text(json.dumps({
        "job": {"forcing": "synthetic", "location": "Sydney", "direction": "downlink",
                "core": "qa2c", "variant": "full", "seed": 0},
        "evaluation": [{"holdout_seed": 1000, "policy": {"action_counts": {"2": 5}},
                        "throughput_delta_pct": 1.0, "rtt_p90_delta_ms": 0.0,
                        "retransmits_delta_per_s": 0.0}]}))

    rows, status = load_rows(tmp_path)
    assert len(rows) == 1
    assert status["declared_jobs"] == 2 and status["completed_jobs"] == 1
