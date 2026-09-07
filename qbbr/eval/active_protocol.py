"""Validate that a planned evaluation covers the active study's commitments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ProtocolReadiness:
    valid: bool
    missing: tuple[str, ...]


def validate_active_protocol(protocol: Mapping[str, Any]) -> ProtocolReadiness:
    """Reject underspecified benchmark plans before results are produced."""

    missing: list[str] = []
    collection = protocol.get("collection", {})
    if collection.get("minimum_independent_terminals", 0) < 2:
        missing.append("at least two independent terminals")
    if collection.get("minimum_geographic_locations", 0) < 2:
        missing.append("at least two geographic locations")
    if not collection.get("longitudinal_capture"):
        missing.append("longitudinal capture duration")
    if not set(collection.get("required_physical_telemetry", ())).issuperset(
        {"elevation_deg", "weather", "handover_event", "snr_db"}
    ):
        missing.append("physical-layer telemetry")
    if not set(collection.get("required_path_evidence", ())).issuperset(
        {"path_fingerprint", "traceroute", "mtr"}
    ):
        missing.append("path-stability evidence")

    benchmark = protocol.get("benchmark", {})
    if not set(benchmark.get("required_ccas", ())).issuperset({"bbr-v3", "cubic", "hybla", "vegas"}):
        missing.append("required baseline CCAs")
    if not benchmark.get("true_mixed_bottleneck_required") or "true_mixed_bottleneck" not in benchmark.get("flow_modes", ()):
        missing.append("true mixed-flow bottleneck experiment")
    if not set(benchmark.get("application_profiles", ())).issuperset(
        {"video_streaming", "voip", "gaming"}
    ):
        missing.append("application-layer profiles")

    control = protocol.get("control", {})
    if not control.get("prohibit_action_set_expansion") or not control.get("fixed_bbr_action_config"):
        missing.append("fixed BBR action-set restriction")
    if not control.get("reward_contract"):
        missing.append("approved reward contract")
    return ProtocolReadiness(valid=not missing, missing=tuple(missing))
