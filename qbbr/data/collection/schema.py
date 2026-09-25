"""Dataset contract for field measurements; missing telemetry is explicit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping


REQUIRED_RUN_FIELDS = frozenset(
    {
        "run_id",
        "terminal_id",
        "geographic_location",
        "started_at_utc",
        "direction",
        "cca",
        "cloud_region",
        "path_fingerprint",
    }
)
PHYSICAL_TELEMETRY_FIELDS = frozenset(
    {"elevation_deg", "weather", "handover_event", "snr_db"}
)
PATH_STABILITY_FIELDS = frozenset({"traceroute", "mtr", "path_fingerprint"})


@dataclass(frozen=True)
class CollectionQualityReport:
    rows: int
    terminals: int
    locations: int
    missing_required: dict[str, int]
    physical_telemetry_coverage: dict[str, float]
    path_stability_coverage: dict[str, float]
    longitudinal_span_days: float | None = None
    distinct_days_by_terminal: dict[str, int] | None = None

    @property
    def is_minimally_ready(self) -> bool:
        return (
            self.rows > 0
            and self.terminals >= 2
            and self.locations >= 2
            and not self.missing_required
            and all(value == 1.0 for value in self.physical_telemetry_coverage.values())
            and all(value == 1.0 for value in self.path_stability_coverage.values())
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["is_minimally_ready"] = self.is_minimally_ready
        return result


def _coverage(rows: list[Mapping[str, Any]], fields: frozenset[str]) -> dict[str, float]:
    total = len(rows)
    if total == 0:
        return {field: 0.0 for field in sorted(fields)}
    return {
        field: sum(row.get(field) not in (None, "") for row in rows) / total
        for field in sorted(fields)
    }


def validate_collection_rows(rows: Iterable[Mapping[str, Any]]) -> CollectionQualityReport:
    """Assess readiness without silently filling unavailable ground truth."""

    materialized = list(rows)
    missing_required = {
        field: sum(row.get(field) in (None, "") for row in materialized)
        for field in REQUIRED_RUN_FIELDS
    }
    missing_required = {field: count for field, count in missing_required.items() if count}
    terminals = {str(row["terminal_id"]) for row in materialized if row.get("terminal_id")}
    locations = {
        str(row["geographic_location"])
        for row in materialized
        if row.get("geographic_location")
    }
    timestamps = []
    days_by_terminal: dict[str, set[str]] = {}
    for row in materialized:
        raw = row.get("started_at_utc")
        if not raw:
            continue
        try:
            value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            continue
        timestamps.append(value)
        terminal = str(row.get("terminal_id", ""))
        if terminal:
            days_by_terminal.setdefault(terminal, set()).add(value.date().isoformat())
    span_days = None
    if len(timestamps) >= 2:
        span_days = (max(timestamps) - min(timestamps)).total_seconds() / 86400.0
    distinct_days = {terminal: len(days) for terminal, days in days_by_terminal.items()}
    return CollectionQualityReport(
        rows=len(materialized),
        terminals=len(terminals),
        locations=len(locations),
        missing_required=missing_required,
        physical_telemetry_coverage=_coverage(materialized, PHYSICAL_TELEMETRY_FIELDS),
        path_stability_coverage=_coverage(materialized, PATH_STABILITY_FIELDS),
        longitudinal_span_days=span_days,
        distinct_days_by_terminal=distinct_days,
    )
