from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_action_space(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open() as f:
        spec = yaml.safe_load(f)

    if "action_space" not in spec or not ("levels" in spec or "dimensions" in spec):
        raise ValueError(f"{config_path}: action config must define action_space and (levels or dimensions)")
    return spec


def n_actions(action_space: dict[str, Any]) -> int:
    if "dimensions" in action_space:
        return n_actions_multihead(action_space)
    return len(action_space["levels"])


def level_for_action(action_space: dict[str, Any], action_index: int) -> float:
    levels = action_space["levels"]
    if not (0 <= action_index < len(levels)):
        raise IndexError(f"action_index {action_index} out of range for {len(levels)} levels")
    return levels[action_index]


def is_multihead(action_space: dict[str, Any]) -> bool:
    return "dimensions" in action_space


def dimension_names(action_space: dict[str, Any]) -> list[str]:
    return list(action_space["dimensions"].keys())


def dimension_sizes(action_space: dict[str, Any]) -> list[int]:
    return [len(d["levels"]) for d in action_space["dimensions"].values()]


def n_actions_multihead(action_space: dict[str, Any]) -> int:
    total = 1
    for size in dimension_sizes(action_space):
        total *= size
    return total


def encode_flat_action(indices: list[int], sizes: list[int]) -> int:
    flat = 0
    for idx, size in zip(indices, sizes):
        if not (0 <= idx < size):
            raise IndexError(f"index {idx} out of range for size {size}")
        flat = flat * size + idx
    return flat


def decode_flat_action(flat: int, sizes: list[int]) -> list[int]:
    total = 1
    for size in sizes:
        total *= size
    if not (0 <= flat < total):
        raise IndexError(f"flat action {flat} out of range for sizes {sizes} (total {total})")
    indices = [0] * len(sizes)
    for i in range(len(sizes) - 1, -1, -1):
        flat, indices[i] = divmod(flat, sizes[i])
    return indices


def levels_for_flat_action(action_space: dict[str, Any], flat: int) -> dict[str, float]:
    dims = action_space["dimensions"]
    names = list(dims.keys())
    sizes = [len(dims[name]["levels"]) for name in names]
    indices = decode_flat_action(flat, sizes)
    return {name: dims[name]["levels"][idx] for name, idx in zip(names, indices)}


def levels_for_action(action_space: dict[str, Any], action_index: int) -> dict[str, float]:
    if is_multihead(action_space):
        return levels_for_flat_action(action_space, action_index)
    return {action_space["action_space"]: level_for_action(action_space, action_index)}
