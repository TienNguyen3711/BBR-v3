"""Load and validate the inventory of BBR-native actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import ActionSpec, ContractError


# Per-action level grids would create a new action set.  A single fixed value
# is allowed through ``fixed_parameters`` only after comparison with the
# canonical BBR action configuration.
FORBIDDEN_ACTION_SET_KEYS = {"levels", "dimensions"}


def action_specs_from_mapping(mapping: Mapping[str, Any]) -> tuple[ActionSpec, ...]:
    raw_actions = mapping.get("actions", ())
    if not raw_actions:
        raise ContractError("Native-action inventory has no actions.")
    specs: list[ActionSpec] = []
    for raw in raw_actions:
        forbidden = FORBIDDEN_ACTION_SET_KEYS.intersection(raw)
        if forbidden:
            fields = ", ".join(sorted(forbidden))
            raise ContractError(
                f"{raw.get('action_id', '<unknown>')} declares an implicit action-set field {fields}. "
                "Use one fixed_parameters mapping per existing BBR choice instead."
            )
        specs.append(
            ActionSpec(
                action_id=str(raw["action_id"]),
                native_semantic=str(raw["native_semantic"]),
                allowed_bbr_states=tuple(raw["allowed_bbr_states"]),
                fixed_parameters=tuple(
                    sorted((str(name), float(value)) for name, value in raw.get("fixed_parameters", {}).items())
                ),
                kernel_capability=raw.get("kernel_capability"),
                enabled=bool(raw.get("enabled", False)),
                description=str(raw.get("description", "")),
            )
        )
    if len({spec.action_id for spec in specs}) != len(specs):
        raise ContractError("Native-action inventory contains duplicate action_id values.")
    return tuple(specs)


def fixed_parameter_choices_from_config(path: str | Path) -> tuple[tuple[tuple[str, float], ...], ...]:
    """Flatten the existing fixed action config without adding any choices."""

    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if "dimensions" in config:
        from itertools import product

        names = tuple(config["dimensions"].keys())
        levels = [config["dimensions"][name]["levels"] for name in names]
        return tuple(
            tuple(sorted((name, float(value)) for name, value in zip(names, values)))
            for values in product(*levels)
        )
    name = str(config["action_space"])
    return tuple(((name, float(value)),) for value in config["levels"])


def validate_fixed_action_set(
    actions: tuple[ActionSpec, ...], config_path: str | Path
) -> None:
    """Ensure the contract has exactly, and only, the existing BBR choices."""

    expected = fixed_parameter_choices_from_config(config_path)
    actual = tuple(action.fixed_parameters for action in actions)
    if actual != expected:
        raise ContractError(
            "The native QRL action set differs from the fixed BBR configuration; "
            "changing its size or values is not permitted."
        )


def load_native_action_inventory(path: str | Path) -> tuple[ActionSpec, ...]:
    with Path(path).open(encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle)
    if not isinstance(mapping, Mapping):
        raise ContractError("Native-action inventory must be a mapping.")
    return action_specs_from_mapping(mapping)
