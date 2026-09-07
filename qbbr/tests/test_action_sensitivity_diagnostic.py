from qbbr.scripts.diagnose_action_sensitivity import _summarize


def test_action_sensitivity_flags_only_a_material_throughput_span() -> None:
    records = []
    for action, throughput in ((0, 99.0), (2, 100.0), (4, 101.0)):
        records.append(
            {
                "location": "Sydney", "direction": "downlink", "forced_action": action,
                "throughput_mbps_mean": throughput, "retransmits_per_s_mean": 1.0,
                "rtt_ms_mean": 20.0, "drawdown_indicator_mean": 0.1,
                "effective_pacing_gain_mean": 1.0, "action_opportunity_fraction": 0.5,
            }
        )
    row = _summarize(records, min_throughput_span_pct=0.5, min_improvement_pct=0.1)[0]
    assert row["throughput_sensitive"] is True
    assert row["throughput_improvable"] is True
    assert row["throughput_span_pct_across_fixed_actions"] == 2.0
