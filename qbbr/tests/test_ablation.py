from __future__ import annotations

import numpy as np

from qbbr.eval.ablation import (
    _point_dirname,
    ablation_grid,
    load_ablation_results,
    run_ablation,
    run_ablation_point,
    save_ablation_results,
)

_TOY_CONFIG = {
    "seed": 0,
    "gamma": 0.99,
    "learning_rate": 1e-3,
    "episode": {"duration_s": 0.2},
    "ablation": {
        "alpha": [0.5, 1],
        "delta_beta_center": [1.0, 0.5],
        "n_layers": [2],
        "data_reuploading": [False, True],
        "risk_features": [False, True],
        "core": ["quantum", "classical"],
    },
}


def test_ablation_grid_skips_classical_reupload_duplicates():
    grid = list(ablation_grid(_TOY_CONFIG))
    # 2 alpha x 1 n_layers x 2 risk x (quantum: 2 reupload + classical: 1 reupload=False) = 2*1*2*3 = 12
    assert len(grid) == 12
    assert not any(p["core"] == "classical" and p["data_reuploading"] for p in grid)


def test_ablation_grid_holds_delta_beta_fixed_at_center():
    for point in ablation_grid(_TOY_CONFIG):
        assert point["delta"] == 1.0
        assert point["beta"] == 0.5


def test_ablation_grid_covers_every_alpha_and_core():
    grid = list(ablation_grid(_TOY_CONFIG))
    assert {p["alpha"] for p in grid} == {0.5, 1}
    assert {p["core"] for p in grid} == {"quantum", "classical"}


def test_run_ablation_point_returns_config_samples_and_summary(sample_calibration):
    point = {
        "alpha": 1.0, "delta": 1.0, "beta": 0.5,
        "n_layers": 2, "data_reuploading": False, "risk_features": False, "core": "classical",
    }
    result = run_ablation_point(
        point, _TOY_CONFIG, "Sydney", "downlink", sample_calibration, n_episodes=2, n_runs=3
    )
    assert result["config"] == point
    assert len(result["samples"]) == 3
    assert np.isfinite(result["samples"]).all()
    assert result["summary"]["n"] == 3


def test_run_ablation_point_seed_makes_samples_reproducible(sample_calibration):
    point = {
        "alpha": 1.0, "delta": 1.0, "beta": 0.5,
        "n_layers": 2, "data_reuploading": False, "risk_features": False, "core": "classical",
    }
    kwargs = dict(
        point=point, base_config=_TOY_CONFIG, location="Sydney", direction="downlink",
        calibration=sample_calibration, n_episodes=2, n_runs=2, base_seed=7,
    )
    assert run_ablation_point(**kwargs)["samples"] == run_ablation_point(**kwargs)["samples"]


def test_run_ablation_point_saves_one_checkpoint_per_seed(tmp_path, sample_calibration):
    point = {
        "alpha": 1.0, "delta": 1.0, "beta": 0.5,
        "n_layers": 2, "data_reuploading": False, "risk_features": False, "core": "classical",
    }
    run_ablation_point(
        point, _TOY_CONFIG, "Sydney", "downlink", sample_calibration,
        n_episodes=2, n_runs=3, base_seed=5, checkpoint_dir=tmp_path,
    )
    saved = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert saved == ["seed5.pt", "seed6.pt", "seed7.pt"]


def test_run_ablation_point_saves_nothing_when_checkpoint_dir_is_none(tmp_path, sample_calibration):
    point = {
        "alpha": 1.0, "delta": 1.0, "beta": 0.5,
        "n_layers": 2, "data_reuploading": False, "risk_features": False, "core": "classical",
    }
    run_ablation_point(
        point, _TOY_CONFIG, "Sydney", "downlink", sample_calibration, n_episodes=2, n_runs=2,
    )
    assert list(tmp_path.iterdir()) == []


def test_run_ablation_uses_one_subdirectory_per_grid_point(tmp_path, sample_calibration):
    small_config = {**_TOY_CONFIG, "ablation": {**_TOY_CONFIG["ablation"], "alpha": [0.5, 1], "n_layers": [2],
                                                 "data_reuploading": [False], "risk_features": [False],
                                                 "core": ["classical"]}}
    run_ablation(
        small_config, "Sydney", "downlink", sample_calibration,
        n_episodes=2, n_runs=1, checkpoint_root=tmp_path,
    )
    grid = list(ablation_grid(small_config))
    expected_dirs = {_point_dirname(point) for point in grid}
    actual_dirs = {p.name for p in tmp_path.iterdir()}
    assert actual_dirs == expected_dirs
    for point_dir in tmp_path.iterdir():
        assert (point_dir / "seed0.pt").exists()


def test_run_ablation_calls_on_point_once_per_grid_point_and_covers_whole_grid(sample_calibration):
    tiny_config = {**_TOY_CONFIG, "ablation": {**_TOY_CONFIG["ablation"], "core": ["classical"]}}
    seen = []
    results = run_ablation(
        tiny_config, "Sydney", "downlink", sample_calibration,
        n_episodes=2, n_runs=2, on_point=lambda i, total, r: seen.append((i, total)),
    )
    expected_grid_size = len(list(ablation_grid(tiny_config)))
    assert len(results) == expected_grid_size
    assert seen == [(i, expected_grid_size) for i in range(expected_grid_size)]


def test_save_and_load_ablation_results_roundtrip(tmp_path, sample_calibration):
    tiny_config = {**_TOY_CONFIG, "ablation": {**_TOY_CONFIG["ablation"], "core": ["classical"], "alpha": [1]}}
    results = run_ablation(tiny_config, "Sydney", "downlink", sample_calibration, n_episodes=2, n_runs=2)

    out_path = tmp_path / "ablation_results.json"
    save_ablation_results(results, out_path, meta={"location": "Sydney", "direction": "downlink"})
    loaded = load_ablation_results(out_path)

    assert loaded["meta"] == {"location": "Sydney", "direction": "downlink"}
    assert len(loaded["results"]) == len(results)
    assert loaded["results"][0]["config"] == results[0]["config"]
    assert loaded["results"][0]["samples"] == results[0]["samples"]


def test_save_ablation_results_creates_parent_dirs(tmp_path):
    out_path = tmp_path / "nested" / "dir" / "results.json"
    save_ablation_results([], out_path)
    assert out_path.exists()
