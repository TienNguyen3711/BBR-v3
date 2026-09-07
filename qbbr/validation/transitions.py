"""Quantitative validation of BBR phase-transition predictions from traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class TransitionFit:
    location: str
    direction: str
    predicted_transitions: int
    observed_transitions: int
    true_positives: int
    precision: float | None
    recall: float | None
    f1: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _transitions(states: list[str | None]) -> set[int]:
    return {
        index
        for index in range(1, len(states))
        if states[index] is not None and states[index - 1] is not None and states[index] != states[index - 1]
    }


def transition_fit_by_scenario(
    rows: Iterable[Mapping[str, object]],
    predicted_state_key: str = "predicted_bbr_state",
    observed_state_key: str = "bbr_state",
) -> list[TransitionFit]:
    """Compare model and observed state transitions per location/direction.

    Matching is indexed by decision interval.  This makes the required timing
    alignment visible; callers must not use illustrative single-city plots as
    a substitute for this all-scenario table.
    """

    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        location = str(row.get("geographic_location", "<missing>"))
        direction = str(row.get("direction", "<missing>"))
        grouped.setdefault((location, direction), []).append(row)
    fits: list[TransitionFit] = []
    for (location, direction), scenario_rows in sorted(grouped.items()):
        ordered = sorted(scenario_rows, key=lambda row: float(row.get("offset_s", 0.0)))
        predicted = _transitions([row.get(predicted_state_key) for row in ordered])
        observed = _transitions([row.get(observed_state_key) for row in ordered])
        true_positives = len(predicted & observed)
        precision = true_positives / len(predicted) if predicted else None
        recall = true_positives / len(observed) if observed else None
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
        fits.append(
            TransitionFit(
                location=location,
                direction=direction,
                predicted_transitions=len(predicted),
                observed_transitions=len(observed),
                true_positives=true_positives,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return fits
