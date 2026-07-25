"""Config-driven discrete action-space registry.

Reads a configs/action_*.yaml file (e.g. action_pacing_gain.yaml, the MVP
action space) and returns the corresponding discrete action levels plus the
qbbr.action.bbr_hook entry point used to apply a chosen action, so
qbbr.env and qbbr.agents never hard-code the action space.

Not yet implemented.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_action_space(config_path: str | Path) -> dict[str, Any]:
    """Parse a configs/action_*.yaml file into {"levels": [...], "hook": ..., ...}."""
    raise NotImplementedError("Action-space registry not yet implemented; see configs/action_*.yaml.")
