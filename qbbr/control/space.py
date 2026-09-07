"""Bridge named BBR semantics to the discrete output of an RL policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .contracts import ActionSpec, ContractError, MDPContract, require_observations
from .safety import enforce_action


_BBR_STATE_CODES = {"STARTUP": 0.0, "DRAIN": 1.0, "PROBE_BW": 2.0, "PROBE_RTT": 3.0}


@dataclass(frozen=True)
class NativeActionSpace:
    """The fixed discrete action space presented to an RL core.

    The index is merely a representation of one declared BBR action.  It may
    carry a numeric gain/inflight value, but only one fixed in the source BBR
    action configuration.
    """

    contract: MDPContract

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(action.action_id for action in self.contract.action_space)

    @property
    def size(self) -> int:
        return len(self.action_ids)

    def index_for(self, action_id: str) -> int:
        try:
            return self.action_ids.index(action_id)
        except ValueError as exc:
            raise ContractError(f"Unknown native action {action_id!r}") from exc

    def action_for_index(self, index: int) -> str:
        try:
            return self.action_ids[index]
        except IndexError as exc:
            raise ContractError(f"Native action index {index} is outside [0, {self.size}).") from exc

    def spec_for_id(self, action_id: str) -> ActionSpec:
        for action in self.contract.action_space:
            if action.action_id == action_id:
                return action
        raise ContractError(f"Unknown native action {action_id!r}")

    def allowed_indices(
        self,
        bbr_state: str,
        observed: Mapping[str, object],
        loss_rate: float | None = None,
    ) -> tuple[int, ...]:
        """Return state-/telemetry-safe RL outputs; always retain stock fallback."""

        valid, _missing = require_observations(self.contract, observed)
        if self.contract.safety.fail_closed_on_missing_telemetry and not valid:
            return (self.index_for(self.contract.safety.stock_fallback_action),)
        allowed: list[int] = []
        for index, action_id in enumerate(self.action_ids):
            decision = enforce_action(self.contract, action_id, bbr_state, observed, loss_rate)
            if not decision.used_stock_fallback:
                allowed.append(index)
        fallback = self.index_for(self.contract.safety.stock_fallback_action)
        return tuple(allowed) if fallback in allowed else (fallback,)

    def masked_logits(self, logits: np.ndarray, allowed_indices: tuple[int, ...]) -> np.ndarray:
        """Mask unavailable semantics before policy sampling/training."""

        values = np.asarray(logits, dtype=float).copy()
        if values.ndim != 1 or len(values) != self.size:
            raise ContractError(f"Expected {self.size} action logits, received shape {values.shape}.")
        mask = np.ones(self.size, dtype=bool)
        mask[list(allowed_indices)] = False
        values[mask] = -np.inf
        return values


def encode_observation(contract: MDPContract, observed: Mapping[str, object]) -> np.ndarray:
    """Create the ordered MDP vector and reject missing required telemetry."""

    valid, missing = require_observations(contract, observed)
    if not valid:
        raise ContractError(f"Cannot encode observation; missing required fields: {', '.join(missing)}")
    encoded: list[float] = []
    for specification in contract.observation_space:
        value = observed.get(specification.name)
        if value is None:  # optional feature: retain an explicit sentinel.
            encoded.append(np.nan)
        elif specification.name == "bbr_state":
            if str(value) not in _BBR_STATE_CODES:
                raise ContractError(f"Unknown BBR state {value!r}.")
            encoded.append(_BBR_STATE_CODES[str(value)])
        else:
            try:
                encoded.append(float(value))
            except (TypeError, ValueError) as exc:
                raise ContractError(f"Observation {specification.name!r} is not numeric: {value!r}") from exc
    return np.asarray(encoded, dtype=np.float32)


@dataclass(frozen=True)
class NativeObservationNormalizer:
    """Train-split-only z-score + bounded angle encoding for the QNN input.

    Optional measurements are imputed only at inference by their training-set
    mean, after their absence has been retained in the field record.  Required
    telemetry is still rejected by ``encode_observation``.
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls, contract: MDPContract, observations: Iterable[Mapping[str, object]]
    ) -> "NativeObservationNormalizer":
        vectors = np.asarray([encode_observation(contract, item) for item in observations], dtype=float)
        if vectors.ndim != 2 or len(vectors) == 0:
            raise ContractError("Fit the native normalizer with at least one complete observation.")
        finite = np.isfinite(vectors)
        count = finite.sum(axis=0)
        mean = np.divide(
            np.where(finite, vectors, 0.0).sum(axis=0),
            count,
            out=np.zeros(vectors.shape[1]),
            where=count > 0,
        )
        squared_error = np.where(finite, (vectors - mean) ** 2, 0.0).sum(axis=0)
        scale = np.sqrt(
            np.divide(squared_error, count, out=np.ones(vectors.shape[1]), where=count > 0)
        )
        scale = np.where(scale > 0, scale, 1.0)
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, contract: MDPContract, observed: Mapping[str, object]) -> np.ndarray:
        vector = encode_observation(contract, observed)
        if vector.shape != self.mean.shape or self.mean.shape != self.scale.shape:
            raise ContractError("Normalizer dimension does not match the active MDP contract.")
        completed = np.where(np.isfinite(vector), vector, self.mean)
        # Bounded angles keep state magnitude valid for the QNN's rotation encoding.
        return (np.tanh((completed - self.mean) / self.scale) * np.pi).astype(np.float32)
