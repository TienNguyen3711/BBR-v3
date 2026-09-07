"""Machine-checkable contracts for a native-action BBR-v3 experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


class ContractError(ValueError):
    """Raised when the declared experiment/control contract is inconsistent."""


DEFAULT_TERMINAL_CONDITION = (
    "episode ends on completed transfer, explicit fault, or duration limit"
)


@dataclass(frozen=True)
class ObservationSpec:
    name: str
    source: str
    units: str
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class ActionSpec:
    """One BBR semantic that the RL layer may select.

    A numeric parameter is valid only when it is a member of the declared,
    fixed BBR action set.  The brain selects an existing choice; it never
    synthesises a new gain or inflight value.
    """

    action_id: str
    native_semantic: str
    allowed_bbr_states: tuple[str, ...]
    fixed_parameters: tuple[tuple[str, float], ...] = ()
    kernel_capability: str | None = None
    enabled: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ContractError("An action requires a non-empty action_id.")
        if not self.native_semantic:
            raise ContractError(f"Action {self.action_id!r} requires native_semantic.")
        if not self.allowed_bbr_states:
            raise ContractError(f"Action {self.action_id!r} requires allowed_bbr_states.")
        parameter_names = [name for name, _value in self.fixed_parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ContractError(f"Action {self.action_id!r} repeats a fixed parameter name.")
        # Stock preservation changes nothing; it is available without a custom
        # kernel hook.  Any other enabled action must expose its kernel path.
        if self.enabled and self.action_id != "stock_preserve" and not self.kernel_capability:
            raise ContractError(
                f"Enabled action {self.action_id!r} has no kernel_capability; "
                "it cannot yet be claimed as a testbed action."
            )

    @property
    def parameters(self) -> dict[str, float]:
        return dict(self.fixed_parameters)


@dataclass(frozen=True)
class RewardSpec:
    reward_id: str
    throughput_utility: str
    rtt_penalty: bool
    retransmission_penalty: bool
    ecn_penalty: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.throughput_utility:
            raise ContractError("Reward must specify its throughput utility.")


@dataclass(frozen=True)
class SafetySpec:
    stock_fallback_action: str = "stock_preserve"
    only_declared_states: bool = True
    fail_closed_on_missing_telemetry: bool = True
    max_loss_rate: float | None = None


@dataclass(frozen=True)
class MDPContract:
    contract_id: str
    observation_space: tuple[ObservationSpec, ...]
    action_space: tuple[ActionSpec, ...]
    reward: RewardSpec
    safety: SafetySpec = field(default_factory=SafetySpec)
    terminal_condition: str = DEFAULT_TERMINAL_CONDITION

    def __post_init__(self) -> None:
        if not self.contract_id:
            raise ContractError("MDP contract requires contract_id.")
        if not self.observation_space:
            raise ContractError("MDP contract requires observations.")
        if not self.action_space:
            raise ContractError("MDP contract requires actions.")
        names = [item.name for item in self.observation_space]
        if len(set(names)) != len(names):
            raise ContractError("Observation names must be unique.")
        action_ids = [item.action_id for item in self.action_space]
        if len(set(action_ids)) != len(action_ids):
            raise ContractError("Action identifiers must be unique.")
        if self.safety.stock_fallback_action not in action_ids:
            raise ContractError("Safety fallback action is not in action space.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunManifest:
    """Minimal provenance needed before a run may enter the active dataset."""

    run_id: str
    terminal_id: str
    geographic_location: str
    direction: str
    cca: str
    started_at_utc: str
    contract_id: str
    data_schema_version: str = "starlink-run-v1"

    def __post_init__(self) -> None:
        missing = [
            name for name, value in asdict(self).items()
            if value is None or (isinstance(value, str) and not value.strip())
        ]
        if missing:
            raise ContractError(f"Run manifest missing: {', '.join(missing)}")
        if self.direction not in {"uplink", "downlink"}:
            raise ContractError("direction must be 'uplink' or 'downlink'.")

    @classmethod
    def now(cls, **kwargs: str) -> "RunManifest":
        return cls(started_at_utc=datetime.now(timezone.utc).isoformat(), **kwargs)

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def mdp_contract_from_mapping(mapping: Mapping[str, Any]) -> MDPContract:
    """Parse a YAML/JSON mapping while retaining strict semantic validation."""

    observations = tuple(
        ObservationSpec(
            name=str(item["name"]),
            source=str(item["source"]),
            units=str(item.get("units", "unitless")),
            required=bool(item.get("required", True)),
            description=str(item.get("description", "")),
        )
        for item in mapping["observation_space"]
    )
    actions = tuple(
        ActionSpec(
            action_id=str(item["action_id"]),
            native_semantic=str(item["native_semantic"]),
            allowed_bbr_states=tuple(str(state) for state in item["allowed_bbr_states"]),
            fixed_parameters=tuple(
                sorted((str(name), float(value)) for name, value in item.get("fixed_parameters", {}).items())
            ),
            kernel_capability=item.get("kernel_capability"),
            enabled=bool(item.get("enabled", False)),
            description=str(item.get("description", "")),
        )
        for item in mapping["action_space"]
    )
    reward_data = mapping["reward"]
    safety_data = mapping.get("safety", {})
    return MDPContract(
        contract_id=str(mapping["contract_id"]),
        observation_space=observations,
        action_space=actions,
        reward=RewardSpec(
            reward_id=str(reward_data["reward_id"]),
            throughput_utility=str(reward_data["throughput_utility"]),
            rtt_penalty=bool(reward_data.get("rtt_penalty", False)),
            retransmission_penalty=bool(reward_data.get("retransmission_penalty", False)),
            ecn_penalty=bool(reward_data.get("ecn_penalty", False)),
            notes=str(reward_data.get("notes", "")),
        ),
        safety=SafetySpec(**safety_data),
        terminal_condition=str(mapping.get("terminal_condition", DEFAULT_TERMINAL_CONDITION)),
    )


def require_observations(
    contract: MDPContract, observed: Mapping[str, object]
) -> tuple[bool, tuple[str, ...]]:
    """Return whether a run has all required observations, without imputation."""

    missing = tuple(
        item.name
        for item in contract.observation_space
        if item.required and (item.name not in observed or observed[item.name] is None)
    )
    return not missing, missing
