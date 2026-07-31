from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_action_space(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open() as f:
        spec = yaml.safe_load(f)

    if "action_space" not in spec or "levels" not in spec:
        raise ValueError(f"{config_path}: action config must define action_space and levels")
    return spec


def n_actions(action_space: dict[str, Any]) -> int:
    return len(action_space["levels"])


def level_for_action(action_space: dict[str, Any], action_index: int) -> float:
    levels = action_space["levels"]
    if not (0 <= action_index < len(levels)):
        raise IndexError(f"action_index {action_index} out of range for {len(levels)} levels")
    return levels[action_index]
