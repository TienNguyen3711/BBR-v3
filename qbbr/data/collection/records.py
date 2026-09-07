"""Write-once field-run records with provenance attached to every sample."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from qbbr.control.contracts import RunManifest


@dataclass(frozen=True)
class IntervalObservation:
    offset_s: float
    delivery_rate_bps: float | None
    rtt_s: float | None
    retransmission_rate: float | None
    bbr_state: str | None
    inflight_bytes: int | None = None
    cwnd_bytes: int | None = None
    elevation_deg: float | None = None
    weather: str | None = None
    handover_event: bool | None = None
    snr_db: float | None = None
    path_fingerprint: str | None = None
    traceroute: str | None = None
    mtr: str | None = None
    application_metrics: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FieldRun:
    manifest: RunManifest
    intervals: tuple[IntervalObservation, ...]
    packet_trace_path: str | None = None
    notes: str = ""

    def flattened_rows(self) -> list[dict[str, Any]]:
        """Produce rows for validation/analysis without losing manifest context."""

        base = self.manifest.as_dict()
        return [{**base, **interval.as_dict()} for interval in self.intervals]
