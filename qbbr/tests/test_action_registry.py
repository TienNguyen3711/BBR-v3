from __future__ import annotations

from pathlib import Path

import pytest

from qbbr.action.registry import level_for_action, load_action_space, n_actions

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
