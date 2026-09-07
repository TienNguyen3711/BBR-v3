from qbbr.reward.native_performance import NativePerformanceReward


def test_native_reward_is_delivered_throughput_only() -> None:
    reward = NativePerformanceReward()
    value = reward(
        throughput_bps=250_000_000,
        rtt_s=0.5,
        min_rtt_s=0.02,
        retransmission_rate=0.9,
        previous_retransmission_rate=0.0,
    )
    assert value == 250.0
