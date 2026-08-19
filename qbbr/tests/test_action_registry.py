from __future__ import annotations

from pathlib import Path

import pytest

from qbbr.action.registry import (
    decode_flat_action,
    dimension_names,
    dimension_sizes,
    encode_flat_action,
    is_multihead,
    level_for_action,
    levels_for_action,
    levels_for_flat_action,
    load_action_space,
    n_actions,
    n_actions_multihead,
)

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_load_pacing_gain_action_space_matches_main_tex():
    action_space = load_action_space(_CONFIGS_DIR / "action_pacing_gain.yaml")
    assert action_space["action_space"] == "pacing_gain"
    assert action_space["levels"] == [0.75, 0.9, 1.0, 1.1, 1.25]
    assert action_space["clamp_state"] == "ProbeBW_CRUISE"


def test_n_actions_matches_level_count():
    action_space = load_action_space(_CONFIGS_DIR / "action_pacing_gain.yaml")
    assert n_actions(action_space) == 5


def test_level_for_action_roundtrip():
    action_space = load_action_space(_CONFIGS_DIR / "action_pacing_gain.yaml")
    for i, expected in enumerate([0.75, 0.9, 1.0, 1.1, 1.25]):
        assert level_for_action(action_space, i) == expected


def test_level_for_action_out_of_range_raises():
    action_space = load_action_space(_CONFIGS_DIR / "action_pacing_gain.yaml")
    with pytest.raises(IndexError):
        level_for_action(action_space, 5)


def test_load_action_space_missing_levels_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("action_space: pacing_gain\n")
    with pytest.raises(ValueError):
        load_action_space(bad)


def _multihead_space():
    return load_action_space(_CONFIGS_DIR / "action_multihead.yaml")


def test_load_multihead_action_space():
    action_space = _multihead_space()
    assert action_space["action_space"] == "multihead"
    assert is_multihead(action_space)
    assert dimension_names(action_space) == ["pacing_gain", "inflight_hi_mult", "inflight_lo_mult"]
    assert dimension_sizes(action_space) == [5, 5, 5]


def test_n_actions_dispatches_to_multihead():
    action_space = _multihead_space()
    assert n_actions(action_space) == 125
    assert n_actions_multihead(action_space) == 125


def test_encode_decode_flat_action_roundtrip():
    sizes = [5, 5, 5]
    for pacing_idx in range(5):
        for hi_idx in range(5):
            for lo_idx in range(5):
                flat = encode_flat_action([pacing_idx, hi_idx, lo_idx], sizes)
                assert decode_flat_action(flat, sizes) == [pacing_idx, hi_idx, lo_idx]


def test_encode_flat_action_covers_every_index_exactly_once():
    sizes = [5, 5, 5]
    seen = set()
    for pacing_idx in range(5):
        for hi_idx in range(5):
            for lo_idx in range(5):
                flat = encode_flat_action([pacing_idx, hi_idx, lo_idx], sizes)
                assert 0 <= flat < 125
                seen.add(flat)
    assert len(seen) == 125


def test_decode_flat_action_out_of_range_raises():
    with pytest.raises(IndexError):
        decode_flat_action(125, [5, 5, 5])


def test_encode_flat_action_out_of_range_index_raises():
    with pytest.raises(IndexError):
        encode_flat_action([5, 0, 0], [5, 5, 5])


def test_levels_for_flat_action_matches_config_defaults():
    action_space = _multihead_space()
    # pacing_gain is centered on the stock value (index 2). inflight_hi/lo
    # are expansion-only (see action_multihead.yaml's comment on the
    # symmetric-range exploit found during training), so their default,
    # matching fluid_sim.FluidParams, sits at index 0, not the center.
    flat = encode_flat_action([2, 0, 0], [5, 5, 5])
    levels = levels_for_flat_action(action_space, flat)
    assert levels == {"pacing_gain": 1.0, "inflight_hi_mult": 2.0, "inflight_lo_mult": 1.0}


def test_levels_for_flat_action_extremes():
    action_space = _multihead_space()
    flat_min = encode_flat_action([0, 0, 0], [5, 5, 5])
    flat_max = encode_flat_action([4, 4, 4], [5, 5, 5])
    assert levels_for_flat_action(action_space, flat_min) == {
        "pacing_gain": 0.75, "inflight_hi_mult": 2.0, "inflight_lo_mult": 1.0,
    }
    assert levels_for_flat_action(action_space, flat_max) == {
        "pacing_gain": 1.25, "inflight_hi_mult": 3.0, "inflight_lo_mult": 1.5,
    }


def test_levels_for_action_uniform_accessor_multihead():
    action_space = _multihead_space()
    flat = encode_flat_action([2, 0, 0], [5, 5, 5])
    assert levels_for_action(action_space, flat) == {
        "pacing_gain": 1.0, "inflight_hi_mult": 2.0, "inflight_lo_mult": 1.0,
    }


def test_levels_for_action_uniform_accessor_single_level():
    action_space = load_action_space(_CONFIGS_DIR / "action_pacing_gain.yaml")
    assert levels_for_action(action_space, 2) == {"pacing_gain": 1.0}
