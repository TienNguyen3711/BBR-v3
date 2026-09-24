"""Difference (counterfactual) reward: the agent's contribution, not the weather."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

EPSILON = 1e-6
EPSILON_RTX = 1.0


def compute_difference_reward(
    row: Mapping[str, float],
    baseline: Mapping[str, Sequence[float]],
    index: int,
    delta: float = 1.0,
    beta: float = 0.5,
    eps: float = EPSILON,
    eps_rtx: float = EPSILON_RTX,
    delay_form: str = "total",
    queue_floor_ms: float = 5.0,
) -> float:
    """Reward for one decision, relative to stock on the same forcing."""
    i = min(int(index), len(baseline["bits_per_second"]) - 1)
    thr_a = float(row["bits_per_second"]) / 1e6
    thr_s = float(baseline["bits_per_second"][i]) / 1e6
    rtt_a = float(row["rtt_ms"])
    rtt_s = float(baseline["rtt_ms"][i])
    rtx_a = float(row["retransmits"])
    rtx_s = float(baseline["retransmits"][i])

    utility = math.log((thr_a + eps) / (thr_s + eps))
    if delay_form == "total":
        delay = delta * math.log((rtt_a + eps) / (rtt_s + eps))
    elif delay_form == "queue":
        # Agent and stock share the path, so one propagation RTT serves both.
        base = float(row["rtt_base_ms"])
        q_a, q_s = max(rtt_a - base, 0.0), max(rtt_s - base, 0.0)
        delay = delta * math.log((q_a + queue_floor_ms) / (q_s + queue_floor_ms))
    else:
        raise ValueError(f"Unknown delay_form {delay_form!r}")
    loss = beta * (rtx_a - rtx_s) / (rtx_s + eps_rtx)
    return utility - delay - loss


def stock_baseline(env: Any, seed: int | None, stock_action: int = 2) -> dict:
    """Per-decision stock telemetry for one seed, for use as the counterfactual."""
    env.reset(seed=seed)
    out: dict[str, list[float]] = {"bits_per_second": [], "rtt_ms": [], "retransmits": []}
    done = False
    while not done:
        allowed = env.allowed_action_indices()
        action = stock_action if stock_action in allowed else allowed[0]
        _state, _reward, done, info = env.step(action)
        out["bits_per_second"].append(info["delivered_bytes"] * 8.0 / info["t_dec_s"])
        out["rtt_ms"].append(info["rtt_ms"])
        out["retransmits"].append(info["retransmits"])
    return out
