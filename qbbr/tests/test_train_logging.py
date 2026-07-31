from __future__ import annotations

import json

from qbbr.train.logging import log_episode, start_run


def test_start_run_creates_dir_with_config_snapshot(tmp_path):
    config = {"seed": 7, "gamma": 0.99}
    run_dir = start_run(config, tmp_path)
    assert run_dir.is_dir()
    snapshot = json.loads((run_dir / "config.json").read_text())
    assert snapshot["seed"] == 7
    assert snapshot["config"] == config


def test_two_runs_in_same_second_do_not_collide(tmp_path):
    run_a = start_run({"seed": 1}, tmp_path)
    run_b = start_run({"seed": 2}, tmp_path)
    assert run_a != run_b
    assert run_a.is_dir() and run_b.is_dir()


def test_log_episode_appends_rows_with_header(tmp_path):
    run_dir = start_run({}, tmp_path)
    log_episode(run_dir, 0, {"episode_reward": 1.5, "loss": 0.2})
    log_episode(run_dir, 1, {"episode_reward": 2.5, "loss": 0.1})

    lines = (run_dir / "episodes.csv").read_text().strip().splitlines()
    assert lines[0] == "episode,episode_reward,loss"
    assert lines[1] == "0,1.5,0.2"
    assert lines[2] == "1,2.5,0.1"
