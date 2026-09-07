from __future__ import annotations

import json
from pathlib import Path

import yaml

from qbbr.control.contracts import mdp_contract_from_mapping
from qbbr.control.native_actions import load_native_action_inventory, validate_fixed_action_set


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "configs"
    with (config_dir / "native_rl_bbr_contract.yaml").open(encoding="utf-8") as handle:
        contract = mdp_contract_from_mapping(yaml.safe_load(handle))
    inventory = load_native_action_inventory(config_dir / "native_bbr_actions.yaml")
    validate_fixed_action_set(contract.action_space, config_dir / "action_pacing_gain.yaml")
    validate_fixed_action_set(inventory, config_dir / "action_pacing_gain.yaml")
    inventory_ids = {action.action_id for action in inventory}
    contract_ids = {action.action_id for action in contract.action_space}
    if contract_ids != inventory_ids:
        raise SystemExit(
            "Contract and inventory action IDs differ: "
            f"contract={sorted(contract_ids)}, inventory={sorted(inventory_ids)}"
        )
    enabled = [action.action_id for action in contract.action_space if action.enabled]
    print(
        json.dumps(
            {
                "contract_id": contract.contract_id,
                "valid": True,
                "enabled_actions": enabled,
                "disabled_actions": [
                    action.action_id for action in contract.action_space if not action.enabled
                ],
                "note": "The five fixed actions match action_pacing_gain.yaml; they require an audited kernel interface before use.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
