from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch

from qbbr.agents.base_agent import BaseAgent
from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
from qbbr.agents.quantum.qa2c import QA2CAgent
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.eval.stats import summarize_median_iqr
from qbbr.train.loop import train


_RISK_MODE_BY_FLAG = {False: "stub_constant", True: "closed_form"}


def ablation_grid(base_config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    grid = base_config["ablation"]
    delta, beta = grid["delta_beta_center"]

    for alpha, n_layers, reupload, risk_features, core in itertools.product(
        grid["alpha"], grid["n_layers"], grid["data_reuploading"], grid["risk_features"], grid["core"]
    ):
        if core == "classical" and reupload:
            continue
        yield {
            "alpha": alpha,
            "delta": delta,
            "beta": beta,
            "n_layers": n_layers,
            "data_reuploading": reupload,
            "risk_features": risk_features,
            "core": core,
        }


def _build_agent(point: dict[str, Any], base_config: dict[str, Any]) -> BaseAgent:
    gamma = base_config.get("gamma", 0.99)
    lr = base_config.get("learning_rate", 1e-3)
    if point["core"] == "quantum":
        return QA2CAgent(n_layers=point["n_layers"], gamma=gamma, lr=lr, reupload=point["data_reuploading"])
    return MLPA2CAgent(n_layers=point["n_layers"], gamma=gamma, lr=lr)


def run_ablation_point(
    point: dict[str, Any],
    base_config: dict[str, Any],
    location: str,
    direction: str,
    calibration: dict[str, Any],
    n_episodes: int,
    n_runs: int = 10,
    base_seed: int = 0,
) -> dict[str, Any]:
    
    samples = []
    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        # Seeded before agent construction, same reasoning as
        # qbbr/scripts/train.py: train()'s own seeding is too late to cover
        # the agent's random weight initialization.
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = FluidSimEnv(
            location,
            direction,
            calibration,
            risk_mode=_RISK_MODE_BY_FLAG[point["risk_features"]],
            episode_s=base_config.get("episode", {}).get("duration_s", 300.0),
            reward_kwargs={"alpha": point["alpha"], "delta": point["delta"], "beta": point["beta"]},
        )
        agent = _build_agent(point, base_config)

        result = train(agent, env, n_episodes, config={"seed": seed})
        samples.append(result["final_mean_reward"])

    return {"config": point, "samples": samples, "summary": summarize_median_iqr(samples)}


def run_ablation(
    base_config: dict[str, Any],
    location: str,
    direction: str,
    calibration: dict[str, Any],
    n_episodes: int,
    n_runs: int = 10,
    base_seed: int = 0,
    on_point: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run every point in ablation_grid(base_config); see run_ablation_point.

    on_point(index, total, result) is called after each grid point
    finishes, for progress reporting on what is, at full scale, a
    multi-hour job.
    """
    grid = list(ablation_grid(base_config))
    results = []
    for i, point in enumerate(grid):
        result = run_ablation_point(
            point, base_config, location, direction, calibration, n_episodes, n_runs, base_seed
        )
        results.append(result)
        if on_point is not None:
            on_point(i, len(grid), result)
    return results


def save_ablation_results(
    results: list[dict[str, Any]], path: str | Path, meta: dict[str, Any] | None = None
) -> None:
    """Persist run_ablation()'s output to JSON, with optional run metadata

    (location, direction, n_runs, n_episodes, episode_s, ...) alongside it.
    """
    payload = {"meta": meta or {}, "results": results}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_ablation_results(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
