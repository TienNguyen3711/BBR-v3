"""Fail-closed selection guard for an RL policy above BBR-v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .contracts import ActionSpec, MDPContract, require_observations


@dataclass(frozen=True)
class ActionDecision:
    requested_action: str
    applied_action: str
    used_stock_fallback: bool
    reason: str


def enforce_action(
    contract: MDPContract,
    requested_action: str,
    bbr_state: str,
    observed: Mapping[str, object],
    loss_rate: float | None = None,
) -> ActionDecision:
    """Permit only declared/observable, state-valid actions; otherwise stock BBR."""

    fallback = contract.safety.stock_fallback_action
    valid_observations, missing = require_observations(contract, observed)
    if contract.safety.fail_closed_on_missing_telemetry and not valid_observations:
        return ActionDecision(requested_action, fallback, True, f"missing telemetry: {', '.join(missing)}")
    if (
        contract.safety.max_loss_rate is not None
        and loss_rate is not None
        and loss_rate > contract.safety.max_loss_rate
    ):
        return ActionDecision(requested_action, fallback, True, "loss safety threshold exceeded")
    by_id = {action.action_id: action for action in contract.action_space}
    action: ActionSpec | None = by_id.get(requested_action)
    if action is None:
        return ActionDecision(requested_action, fallback, True, "undeclared action")
    if not action.enabled:
        return ActionDecision(requested_action, fallback, True, "action not kernel-enabled")
    if contract.safety.only_declared_states and bbr_state not in action.allowed_bbr_states:
        return ActionDecision(requested_action, fallback, True, "action invalid for current BBR state")
    return ActionDecision(requested_action, requested_action, False, "accepted")
