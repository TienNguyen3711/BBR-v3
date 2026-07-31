from __future__ import annotations

import struct

import pytest

from qbbr.action import bbr_hook
from qbbr.env.testbed_env import (
    _TCP_INFO_FIELDS,
    _TCP_INFO_FORMAT,
    _TCP_INFO_SIZE,
    TestbedEnv,
    read_tcp_info,
)

_SAMPLE_CALIBRATION = {"B_max_mbps": 200.0, "RTT_min_ms": 50.0, "RTT_max_ms": 100.0}


@pytest.fixture(autouse=True)
def _redirect_debugfs_paths(tmp_path, monkeypatch):
    gain_path = tmp_path / "pacing_gain_override"
    cruise_path = tmp_path / "cruise_active"
    monkeypatch.setattr(bbr_hook, "_PACING_GAIN_OVERRIDE_PATH", gain_path)
    monkeypatch.setattr(bbr_hook, "_CRUISE_ACTIVE_PATH", cruise_path)
    cruise_path.write_text("0\n")


class _FakeSocket:
    def __init__(self, packed_bytes: bytes) -> None:
        self._packed_bytes = packed_bytes

    def getsockopt(self, level, optname, buflen):
        return self._packed_bytes[:buflen]


def _pack_tcp_info(**overrides) -> bytes:
    values = {name: 0 for name in _TCP_INFO_FIELDS}
    values.update(overrides)
    return struct.pack(_TCP_INFO_FORMAT, *(values[name] for name in _TCP_INFO_FIELDS))


def test_read_tcp_info_parses_known_fields_correctly():
    raw = _pack_tcp_info(rtt=50_000, snd_cwnd=10, total_retrans=3, retransmits=1)
    info = read_tcp_info(_FakeSocket(raw))
    assert info["rtt"] == 50_000
    assert info["snd_cwnd"] == 10
    assert info["total_retrans"] == 3
    assert info["retransmits"] == 1


def test_read_tcp_info_returns_every_documented_field():
    raw = _pack_tcp_info()
    info = read_tcp_info(_FakeSocket(raw))
    assert set(info.keys()) == set(_TCP_INFO_FIELDS)


def test_tcp_info_size_is_consistent_with_format_string():
    assert struct.calcsize(_TCP_INFO_FORMAT) == _TCP_INFO_SIZE


def test_testbed_env_matches_base_env_interface():
    raw = _pack_tcp_info(rtt=50_000, snd_cwnd=10)
    env = TestbedEnv(_FakeSocket(raw), _SAMPLE_CALIBRATION, episode_s=10.0)
    assert env.observation_dim == 6
    assert env.action_space_size == 5


def test_step_before_reset_raises():
    raw = _pack_tcp_info(rtt=50_000, snd_cwnd=10)
    env = TestbedEnv(_FakeSocket(raw), _SAMPLE_CALIBRATION, episode_s=10.0)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_reset_produces_a_valid_6dim_state():
    raw = _pack_tcp_info(rtt=50_000, snd_cwnd=10, snd_mss=1460)
    env = TestbedEnv(_FakeSocket(raw), _SAMPLE_CALIBRATION, episode_s=10.0)
    s0 = env.reset()
    assert s0.shape == (6,)
