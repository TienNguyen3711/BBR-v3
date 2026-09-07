from __future__ import annotations

import json
from pathlib import Path

import yaml

from qbbr.eval.active_protocol import validate_active_protocol


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "active_study_protocol.yaml"
    with path.open(encoding="utf-8") as handle:
        result = validate_active_protocol(yaml.safe_load(handle))
    print(json.dumps({"valid": result.valid, "missing": result.missing}, indent=2))
    if not result.valid:
        raise SystemExit("Active protocol is incomplete.")


if __name__ == "__main__":
    main()
