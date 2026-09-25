"""Telemetry sources for the field agent, and the state vector built from them."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from qbbr.features.bbr_internals import compute_bhat_mbps
from qbbr.features.state_builder import compute_state_vector
from qbbr.risk.ptot import compute_risk_features

MSS_BYTES = 1448.0
_BHAT_WINDOW_S = 10.0


@dataclass(frozen=True)
class TcpSample:
    """One observation of the monitored flow, in the units the kernel reports."""

    now_s: float              # UTC epoch seconds -- see the module docstring
    delivered_bytes: float    # since the previous sample
    interval_s: float         # wall time since the previous sample
    rtt_ms: float             # smoothed RTT (tcpi_rtt / 1000)
    min_rtt_ms: float         # tcpi_min_rtt / 1000
    inflight_bytes: float     # tcpi_unacked * mss, or tcpi_bytes_in_flight
    delivery_rate_bps: float  # tcpi_delivery_rate * 8 -- BBR's own bw estimate
    retransmits: float        # since the previous sample


class TelemetrySource(Protocol):
    """Where TcpSamples come from. Implemented by the kernel reader and by replay."""

    def sample(self) -> TcpSample: ...


class FieldStateBuilder:
    """Turn a stream of TcpSamples into the canonical seven-feature state."""

    def __init__(self, calibration: dict[str, float], risk_mode: str = "closed_form_dynamic",
                 reconfig_cycle_s: float = 15.0, reconfig_mean_phase_s: float = 10.5) -> None:
        self.calibration = calibration
        self.risk_mode = risk_mode
        self.reconfig_cycle_s = reconfig_cycle_s
        self.reconfig_mean_phase_s = reconfig_mean_phase_s
        self._history: list[dict[str, float]] = []

    def _row(self, sample: TcpSample) -> dict[str, float]:
        bdp_bytes = max(sample.delivery_rate_bps / 8.0 * sample.min_rtt_ms / 1000.0, 1.0)
        queue_bytes = max(sample.inflight_bytes - bdp_bytes, 0.0)
        interval_s = max(sample.interval_s, 1e-6)
        return {
            # UTC seconds: compute_state_vector takes `t_start % cycle`, so the
            # modulo of epoch seconds is exactly the UTC-anchored phase.
            "t_start": sample.now_s,
            "b_hat_mbps": 0.0,                      # filled by _update_bhat
            "rtt_ms": sample.rtt_ms,
            "rtt_base_ms": sample.min_rtt_ms,
            "v_over_bdp": sample.inflight_bytes / bdp_bytes,
            "q_packets": queue_bytes / MSS_BYTES,
            "bits_per_second": sample.delivered_bytes * 8.0 / interval_s,
            "retransmits": sample.retransmits,
            "ecn_mark_fraction": 0.0,               # not read from tcp_info
        }

    def push(self, sample: TcpSample):
        """Append one sample and return the current seven-feature state vector."""
        row = self._row(sample)
        self._history.append(row)

        # b_hat over a 10 s window, matching FluidSimEnv._update_bhat.
        window = max(1, round(_BHAT_WINDOW_S / max(sample.interval_s, 1e-6)))
        series = pd.Series([r["bits_per_second"] for r in self._history])
        self._history[-1]["b_hat_mbps"] = float(compute_bhat_mbps(series, window=window).iloc[-1])

        frame = pd.DataFrame(self._history[-2:])
        risk = compute_risk_features(frame, mode=self.risk_mode)
        state = compute_state_vector(
            frame, risk, self.calibration,
            reconfig_cycle_s=self.reconfig_cycle_s,
            reconfig_mean_phase_s=self.reconfig_mean_phase_s,
        )
        from qbbr.env.fluid_env import _STATE_COLS
        return state.iloc[-1][_STATE_COLS].to_numpy(dtype=float)


class ReplayTelemetrySource:
    """Replays recorded TcpSamples."""

    def __init__(self, samples: list[TcpSample]) -> None:
        self._samples = list(samples)
        self._index = 0

    def sample(self) -> TcpSample:
        if self._index >= len(self._samples):
            raise StopIteration("replay exhausted")
        item = self._samples[self._index]
        self._index += 1
        return item


class LinuxTcpInfoSource:
    """Reads tcp_info for one established socket via getsockopt."""

    _TCP_INFO = 11

    def __init__(self, sock, mss_bytes: float = MSS_BYTES) -> None:
        self._sock = sock
        self._mss = mss_bytes
        self._previous: dict[str, float] | None = None

    def sample(self) -> TcpSample:
        from qbbr.field.tcp_info import read_tcp_info, require_measurement_fields
        info = read_tcp_info(self._sock)
        require_measurement_fields(info)
        now = time.monotonic()
        total_retrans = info['total_retrans']
        bytes_acked = info['bytes_acked']
        sample = {
            "rtt_ms": info['rtt'] / 1000.0,
            "min_rtt_ms": (info['min_rtt'] if info['min_rtt'] != 0xffffffff else info['rtt']) / 1000.0,
            "inflight_bytes": max(info['unacked'] - info['sacked'] - info['lost'] + info['retrans'], 0) * info['snd_mss'],
            "delivery_rate_bps": info['delivery_rate'] * 8.0,
            "bytes_acked": bytes_acked,
            "total_retrans": total_retrans,
            "now": now,
        }
        previous = self._previous or dict(sample, bytes_acked=bytes_acked,
                                          total_retrans=total_retrans, now=now - 1e-3)
        self._previous = sample
        return TcpSample(
            now_s=time.time(),
            delivered_bytes=max(sample["bytes_acked"] - previous["bytes_acked"], 0.0),
            interval_s=max(now - previous["now"], 1e-6),
            rtt_ms=sample["rtt_ms"],
            min_rtt_ms=sample["min_rtt_ms"] or sample["rtt_ms"],
            inflight_bytes=sample["inflight_bytes"],
            delivery_rate_bps=sample["delivery_rate_bps"],
            retransmits=(sample["total_retrans"] - previous["total_retrans"]) % (1 << 32),
        )
