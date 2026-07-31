from __future__ import annotations

import csv
import json
import time
import uuid
from pathlib import Path
from typing import Any


def start_run(config: dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    snapshot = {"seed": config.get("seed"), "config": config}
    (run_dir / "config.json").write_text(json.dumps(snapshot, indent=2, default=str))
    return run_dir


def log_episode(run_dir: str | Path, episode: int, metrics: dict[str, float]) -> None:
    path = Path(run_dir) / "episodes.csv"
    row = {"episode": episode, **metrics}

    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
