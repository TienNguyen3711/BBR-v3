
from __future__ import annotations

import json
from pathlib import Path

import yaml

from qbbr.control.contracts import mdp_contract_from_mapping
from qbbr.control.native_actions import load_native_action_inventory, validate_fixed_action_set


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "configs"
    with (directory / "native_rl_bbr_contract.yaml").open(encoding="utf-8") as handle:
        contract = mdp_contract_from_mapping(yaml.safe_load(handle))
    inventory = load_native_action_inventory(directory / "native_bbr_actions.yaml")
    validate_fixed_action_set(contract.action_space, directory / "action_pacing_gain.yaml")
    enabled_non_stock = [
        action.action_id
        for action in contract.action_space
        if action.enabled and action.action_id != contract.safety.stock_fallback_action
    ]
    print(
        json.dumps(
            {
                "contract_id": contract.contract_id,
                "observation_dim": len(contract.observation_space),
                "native_action_count": len(contract.action_space),
                "inventory_matches_contract": {
                    action.action_id for action in inventory
                } == {action.action_id for action in contract.action_space},
                "enabled_non_stock_actions": enabled_non_stock,
                "field_training_ready": bool(enabled_non_stock),
                "blocker": None if enabled_non_stock else "audit and enable a real BBR-v3 native interface",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
