"""One-step counterfactual action-sensitivity diagnostic.

This module follows a stock-action trajectory and, at every decision point,
branches a copy of the environment once for every *admissible* action.  The
branches share precisely the same state, phase offset, and EMA history.  It
therefore measures local control authority, not a trainable policy or a
globally feasible oracle trajectory.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import numpy as np

from qbbr.action.registry import levels_for_action
from qbbr.env.fluid_env import FluidSimEnv


def stock_action_index(action_config: dict[str, Any], n_actions: int) -> int:
    """Return the config action that represents the unmodified BBR defaults.

    Pacing gain defaults to 1.0.  In multi-head configurations, every
    non-pacing dimension's first level is the documented safe stock value.
    """
    for action in range(n_actions):
        levels = levels_for_action(action_config, action)
        if not np.isclose(levels.get("pacing_gain", 1.0), 1.0):
            continue
        if all(name == "pacing_gain" or np.isclose(value, spec["levels"][0])
               for name, value in levels.items()
               for spec in [action_config.get("dimensions", {}).get(name, {"levels": [value]})]):
            return action
    raise ValueError("action configuration does not contain a stock/no-op action")


def _action_record(action: int, action_config: dict[str, Any]) -> dict[str, Any]:
    return {"action": action, "levels": levels_for_action(action_config, action)}


def action_sensitivity(
    location: str,
    direction: str,
    calibration: dict[str, dict[str, dict[str, float]]],
    action_config: dict[str, Any],
    *,
    seeds: Sequence[int] = (1001, 1002, 1003),
    episode_s: float = 300.0,
    risk_mode: str = "closed_form",
    throughput_retention: float = 0.95,
    stock_action: int | None = None,
) -> dict[str, Any]:
    """Measure action headroom along stock trajectories.

    ``one_step_oracle_bound`` sums the best feasible *branch* at each stock
    state.  Because selecting that action would alter later states, it is an
    optimistic local bound, deliberately not reported as an achievable policy
    episode result.
    """
    if not 0.0 < throughput_retention <= 1.0:
        raise ValueError("throughput_retention must be in (0, 1]")

    template = FluidSimEnv(location, direction, calibration, action_config=action_config,
                           risk_mode=risk_mode, episode_s=episode_s)
    n_actions = template.action_space_size
    stock_action = stock_action if stock_action is not None else stock_action_index(action_config, n_actions)
    if not 0 <= stock_action < n_actions:
        raise IndexError(f"stock_action {stock_action} out of range")

    totals = {
        action: {"decisions": 0, "retransmits": 0.0, "delivered_bytes": 0.0,
                 "delta_retransmits_vs_stock": 0.0, "delta_delivered_bytes_vs_stock": 0.0,
                 "rtx_reducing_decisions": 0, "feasible_rtx_reducing_decisions": 0}
        for action in range(n_actions)
    }
    baseline = {"decisions": 0, "retransmits": 0.0, "delivered_bytes": 0.0}
    oracle = {"decisions": 0, "retransmits": 0.0, "delivered_bytes": 0.0}

    for seed in seeds:
        env = FluidSimEnv(location, direction, calibration, action_config=action_config,
                           risk_mode=risk_mode, episode_s=episode_s)
        env.reset(seed=seed)
        done = False
        while not done:
            # Every branch starts from exactly the observed stock state.
            branches = {action: deepcopy(env).step(action)[3] for action in range(n_actions)}
            stock = branches[stock_action]
            min_delivered = throughput_retention * stock["delivered_bytes"]
            feasible = [info for info in branches.values() if info["delivered_bytes"] + 1e-9 >= min_delivered]
            best = min(feasible, key=lambda info: (info["retransmits"], -info["delivered_bytes"]))

            baseline["decisions"] += 1
            baseline["retransmits"] += stock["retransmits"]
            baseline["delivered_bytes"] += stock["delivered_bytes"]
            oracle["decisions"] += 1
            oracle["retransmits"] += best["retransmits"]
            oracle["delivered_bytes"] += best["delivered_bytes"]

            for action, info in branches.items():
                item = totals[action]
                d_rtx = info["retransmits"] - stock["retransmits"]
                d_delivered = info["delivered_bytes"] - stock["delivered_bytes"]
                item["decisions"] += 1
                item["retransmits"] += info["retransmits"]
                item["delivered_bytes"] += info["delivered_bytes"]
                item["delta_retransmits_vs_stock"] += d_rtx
                item["delta_delivered_bytes_vs_stock"] += d_delivered
                item["rtx_reducing_decisions"] += int(d_rtx < -1e-9)
                item["feasible_rtx_reducing_decisions"] += int(
                    d_rtx < -1e-9 and info["delivered_bytes"] + 1e-9 >= min_delivered
                )

            # Advance the sole real trajectory only with stock BBR.
            _, _, done, _ = env.step(stock_action)

    for item in totals.values():
        n = item["decisions"]
        item["mean_delta_retransmits_per_decision"] = item["delta_retransmits_vs_stock"] / n
        item["mean_delivery_retention"] = item["delivered_bytes"] / baseline["delivered_bytes"]
        item["feasible_rtx_reduction_share"] = item["feasible_rtx_reducing_decisions"] / n

    oracle["retransmit_reduction_vs_stock"] = (
        (baseline["retransmits"] - oracle["retransmits"]) / baseline["retransmits"]
        if baseline["retransmits"] > 0 else 0.0
    )
    oracle["delivery_retention_vs_stock"] = oracle["delivered_bytes"] / baseline["delivered_bytes"]

    return {
        "meta": {
            "location": location, "direction": direction, "seeds": list(seeds),
            "episode_s": episode_s, "risk_mode": risk_mode,
            "throughput_retention_constraint": throughput_retention,
            "stock_action": _action_record(stock_action, action_config),
            "interpretation": "One-step branches on stock states; the oracle total is an optimistic local bound, not an executable policy result.",
        },
        "stock_trajectory": baseline,
        "one_step_oracle_bound": oracle,
        "actions": [{**_action_record(action, action_config), **totals[action]} for action in range(n_actions)],
    }
