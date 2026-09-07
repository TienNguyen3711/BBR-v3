from pathlib import Path

import yaml

from qbbr.eval.active_protocol import validate_active_protocol


def test_active_protocol_covers_required_study_axes() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "active_study_protocol.yaml"
    with path.open(encoding="utf-8") as handle:
        result = validate_active_protocol(yaml.safe_load(handle))
    assert result.valid is True


def test_protocol_rejects_simulator_only_fairness_plan() -> None:
    result = validate_active_protocol(
        {
            "collection": {"minimum_independent_terminals": 1},
            "benchmark": {"flow_modes": ["parallel"]},
            "control": {},
        }
    )
    assert result.valid is False
    assert "true mixed-flow bottleneck experiment" in result.missing
