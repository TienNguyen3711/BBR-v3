"""Robust parsing of raw iperf3 JSON logs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParseDiagnostics:
    had_leading_junk: bool
    had_trailing_junk: bool
    had_error_field: bool
    error_message: str | None
    n_intervals: int


@dataclass(frozen=True)
class ParsedTrace:
    raw: dict[str, Any]
    diagnostics: ParseDiagnostics


def parse_iperf3_json(text: str) -> ParsedTrace:
    """Parse the primary JSON object out of a raw iperf3 log's text content."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in file (no '{' present)")

    obj, end_idx = json.JSONDecoder().raw_decode(text, start)

    had_leading_junk = text[:start].strip() != ""
    had_trailing_junk = text[end_idx:].strip() != ""
    error_message = obj.get("error")
    diagnostics = ParseDiagnostics(
        had_leading_junk=had_leading_junk,
        had_trailing_junk=had_trailing_junk,
        had_error_field=error_message is not None,
        error_message=error_message,
        n_intervals=len(obj.get("intervals", [])),
    )
    return ParsedTrace(raw=obj, diagnostics=diagnostics)


def parse_iperf3_file(path: str | Path) -> ParsedTrace:
    text = Path(path).read_text()
    return parse_iperf3_json(text)
