from __future__ import annotations

import pytest

from qbbr.action import bbr_hook


@pytest.fixture(autouse=True)
def _redirect_debugfs_paths(tmp_path, monkeypatch):
    """Point the module's assumed debugfs paths at real tmp files.

    This tests our own read/write contract logic, not the (nonexistent
    here) kernel side -- see bbr_hook's module docstring.
    """
    gain_path = tmp_path / "pacing_gain_override"
    cruise_path = tmp_path / "cruise_active"
    monkeypatch.setattr(bbr_hook, "_PACING_GAIN_OVERRIDE_PATH", gain_path)
    monkeypatch.setattr(bbr_hook, "_CRUISE_ACTIVE_PATH", cruise_path)
    cruise_path.write_text("0\n")
    yield gain_path, cruise_path


def test_set_pacing_gain_writes_the_value(_redirect_debugfs_paths):
    gain_path, _ = _redirect_debugfs_paths
    bbr_hook.set_pacing_gain(1.1)
    assert float(gain_path.read_text().strip()) == pytest.approx(1.1)


def test_clear_pacing_gain_override_writes_sentinel(_redirect_debugfs_paths):
    gain_path, _ = _redirect_debugfs_paths
    bbr_hook.set_pacing_gain(0.9)
    bbr_hook.clear_pacing_gain_override()
    assert float(gain_path.read_text().strip()) == bbr_hook._NO_OVERRIDE_SENTINEL


def test_set_pacing_gain_none_is_equivalent_to_clear(_redirect_debugfs_paths):
    gain_path, _ = _redirect_debugfs_paths
    bbr_hook.set_pacing_gain(None)
    assert float(gain_path.read_text().strip()) == bbr_hook._NO_OVERRIDE_SENTINEL


def test_in_probe_bw_cruise_reads_state_file(_redirect_debugfs_paths):
    _, cruise_path = _redirect_debugfs_paths
    cruise_path.write_text("1\n")
    assert bbr_hook.in_probe_bw_cruise() is True
    cruise_path.write_text("0\n")
    assert bbr_hook.in_probe_bw_cruise() is False
