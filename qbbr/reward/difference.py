"""Difference (counterfactual) reward: the agent's contribution, not the weather.

Measured on this simulator, the canonical seven-feature state explains only
~11% of the variance of the discounted return; a linear critic on the feature
the QNN critic actually sees explains 0.6%. The other ~89% is exogenous --
handover dips and capacity drift that the agent neither observes nor controls.
An absolute-throughput reward therefore hands the policy gradient a signal that
is roughly ten times more "weather" than "decision", which is why reward
scaling, entropy schedules, a 30x RTT penalty, n-step updates and a faster
critic all left the learned policy unchanged.

The environment is deterministic given its seed, so the fix is common random
numbers: run stock BBR over the *same* capacity forcing, cache its per-decision
telemetry, and reward the agent for the difference. The exogenous term cancels
by construction and the residual is exactly the decision's contribution.

    reward(t) = log(thr_a / thr_s)
                - delta * log(rtt_a / rtt_s)
                - beta  * (rtx_a - rtx_s) / (rtx_s + eps_rtx)

With delay_form="queue" the delay term compares queueing delay instead of
total RTT, q = rtt - rtt_base, over a fixed floor q0 (queue_floor_ms):

                - delta * log((q_a + q0) / (q_s + q0))

The total-RTT ratio is structurally weak on long paths: a 3 ms self-inflicted
queue is ~1% of London's 258 ms base RTT, so no delta <= 2 outweighs a ~4%
log-throughput gain (delta sweep, 2026-09-24). The queue form prices a
millisecond of queue the same on every path. q0 is needed because stock's
median queue in the simulator is 0 ms; it sets the scale at which extra queue
starts to cost the full log-ratio.

Caveat that must travel with any result: at decision t the agent occupies a
different state than stock did (different queue, different bandwidth estimate),
so the stock row is a counterfactual for the same *forcing*, not for the
agent's own state. That is what we want -- the divergence between the two
trajectories IS the agent's effect -- but it is not a per-state counterfactual.
"""

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
    """Reward for one decision, relative to stock on the same forcing.

    `index` is clamped to the baseline's length: the agent's trajectory can run
    a different number of decisions than stock if the episode boundary lands
    differently, and the tail is better compared against stock's last decision
    than dropped.
    """
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
    """Per-decision stock telemetry for one seed, for use as the counterfactual.

    `env` must be a twin of the learner's environment -- same calibration, same
    capacity trace, same phase offset -- whose reward mode is anything but
    `difference` (otherwise building the baseline would recurse). Deterministic
    given the seed, so a caller computes this once per seed and reuses it for
    every episode that replays that seed.
    """
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
