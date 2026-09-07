"""Supervisor-locked throughput-only reward for native-action RL--BBR."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NativePerformanceReward:
    r"""\(R_t=x_t\), with \(x_t\) delivered throughput in Mbps.

    RTT, retransmission rate, queue and risk are passed to preserve the live
    adapter interface, but deliberately do not alter the learning signal.
    They remain safety guards and reporting metrics.
    """

    def throughput_utility(self, throughput_bps: float) -> float:
        return max(float(throughput_bps) / 1e6, 0.0)

    def __call__(
        self,
        throughput_bps: float,
        rtt_s: float,
        min_rtt_s: float,
        retransmission_rate: float,
        previous_retransmission_rate: float | None,
    ) -> float:
        del rtt_s, min_rtt_s, retransmission_rate, previous_retransmission_rate
        return self.throughput_utility(throughput_bps)
