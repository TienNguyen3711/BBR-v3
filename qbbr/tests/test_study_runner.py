"""Contract tests for the RQ study matrix: ablation, enumeration, and the
stock-rollout cache that makes the matrix affordable."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from qbbr.eval.state_ablation import FEATURE_SETS, PolicyObservationEnv, mask_features
from qbbr.study.protocol import jobs, load_protocol, validate_protocol

ROOT = Path(__file__).resolve().parents[2]
SCREEN = ROOT / "qbbr" / "configs" / "rq_study_screen.yaml"
CALIBRATION = ROOT / "qbbr" / "data" / "calibrated" / "per_location_constants_v14.json"


@pytest.fixture(scope="module")
def study():
    return load_protocol(SCREEN)


@pytest.fixture(scope="module")
def calibration():
    return json.loads(CALIBRATION.read_text())


# --------------------------------------------------------------- RQ2 ablation

def test_feature_sets_cover_the_declared_rq2_contrasts():
    assert FEATURE_SETS["full"] == ()
    assert FEATURE_SETS["telemetry"] == (3, 4, 5, 6)  # s1-s3 only
    assert FEATURE_SETS["telemetry_queue"] == (4, 5, 6)  # s1-s4
    for name, index in [("no_queue", 3), ("no_handover", 4), ("no_failure", 5), ("no_phase", 6)]:
        assert FEATURE_SETS[name] == (index,)


def test_mask_replaces_exactly_the_ablated_features():
    state = np.linspace(0.1, 0.7, 7, dtype=np.float32)
    masked = mask_features(state, "telemetry")
    assert np.allclose(masked[:3], state[:3])
    assert np.allclose(masked[3:], 0.5)
    assert np.allclose(mask_features(state, "full"), state)


def test_mask_rejects_malformed_observations():
    with pytest.raises(ValueError):
        mask_features(np.zeros(6), "full")
    with pytest.raises(ValueError):
        mask_features(np.full(7, np.nan), "full")
    with pytest.raises(ValueError):
        mask_features(np.zeros(7), "no_such_variant")


class _RecordingSelector:
    def __init__(self):
        self.seen = []

    def admissible_actions(self, logits, state, allowed):
        self.seen.append(np.asarray(state, dtype=float).copy())
        return tuple(allowed)


class _StubEnv:
    def __init__(self, state):
        self.state = np.asarray(state, dtype=np.float32)

    def reset(self, seed=None):
        return self.state

    def allowed_action_indices(self):
        return (0, 1, 2, 3, 4)

    def step(self, action):
        return self.state, 0.0, True, {"t_dec_s": 0.1}


def test_safety_mask_sees_raw_state_while_the_policy_sees_the_ablation():
    """The ablation is a policy-input restriction, not a change to safety.

    If the mask were computed on ablated features, every arm would get a
    different admissible set and the RQ2 contrast would confound the feature
    ablation with a safety change.
    """
    raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    selector = _RecordingSelector()
    env = PolicyObservationEnv(_StubEnv(raw), selector, "telemetry")

    observation = env.reset(seed=0)
    assert np.allclose(observation[3:], 0.5)  # policy input is ablated
    env.allowed_action_indices()
    assert np.allclose(selector.seen[-1], raw)  # safety input is not

    _, _, _, info = env.step(2)
    assert info["state_ablation"] == "telemetry"
    assert np.allclose(info["raw_state"], raw)


def test_step_refuses_an_action_outside_the_safety_mask():
    class _Restricted(_StubEnv):
        def allowed_action_indices(self):
            return (2,)

    env = PolicyObservationEnv(_Restricted(np.zeros(7)), _RecordingSelector(), "full")
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(4)


# ------------------------------------------------------------- enumeration

def test_qdqn_is_enumerated_only_at_the_full_state(study):
    enumerated = list(jobs(study))
    assert {job["variant"] for job in enumerated if job["core"] == "qdqn"} == {"full"}
    # qdqn is an estimator/history ablation, not a 126-parameter match, so it
    # must not multiply across the RQ2 variants.
    per_cell = len(study["variants"]) * 2 + 1
    assert len(enumerated) == (
        len(study["forcing"]) * len(study["locations"]) * len(study["directions"])
        * len(study["training_seeds"]) * per_cell
    )


def test_screen_protocol_is_the_difference_reward_contract(study):
    assert study["base"]["simulator"]["reward_mode"] == "difference"
    assert study["base"]["control"]["fixed_action_count"] == 5


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(training_seeds=[0, 1, 1]),
    lambda s: s.update(holdout_seeds=[0, 1000]),          # overlaps a training seed
    lambda s: s.update(variants=["full", "no_such_one"]),
    lambda s: s.update(cores=["qa2c", "random_forest"]),
    lambda s: s.update(episodes=0),
    lambda s: s["base"]["control"].update(fixed_action_count=6),
    lambda s: s["base"]["simulator"].update(reward_mode="throughput_only"),
])
def test_validate_protocol_rejects_contract_violations(study, mutate):
    broken = deepcopy(study)
    mutate(broken)
    with pytest.raises(ValueError):
        validate_protocol(broken)


# ------------------------------------------------------- stock rollout cache

# Wall-clock fields are measurements of this machine, not of the transport, so
# they are not reproducible between two runs of the same rollout. Everything
# else in a rollout is a deterministic function of the forcing and the seed.
_WALL_CLOCK_FIELDS = ("inference_p95_ms", "deadline_miss_fraction")


def _transport_only(rows):
    return [
        {
            key: ({name: number for name, number in value.items() if name not in _WALL_CLOCK_FIELDS}
                  if key in ("stock", "policy") else value)
            for key, value in row.items()
        }
        for row in rows
    ]


class _FixedAction:
    """Deterministic non-stock policy, so cached and uncached rows can differ."""

    def act(self, state, allowed=None, deterministic=True):
        allowed = tuple(allowed or (2,))
        return (max(allowed) if max(allowed) != 2 else 2), 0.0


def test_stock_cache_reproduces_the_uncached_evaluation(study, calibration):
    """The cache must be an optimisation only.

    Stock ignores the observation, so its rollout is shared across variants and
    cores. This asserts the shared rollout equals the independently computed
    one for two different variants -- if the simulator ever drew from a global
    RNG, or the mask reached the baseline, these would diverge.
    """
    from qbbr.study import runner

    scoped = deepcopy(study)
    scoped.update(duration_s=6.0, holdout_seeds=[1000, 1001])
    agent = _FixedAction()
    variants = ("full", "telemetry")

    def rows_for(variant, cache):
        job = dict(forcing="synthetic", location="Sydney", direction="downlink",
                   seed=0, core="qa2c", variant=variant)
        return runner.evaluate(scoped, job, calibration, agent, stock_cache=cache)

    uncached = {variant: rows_for(variant, None) for variant in variants}
    shared = {}
    cached = {variant: rows_for(variant, shared) for variant in variants}

    for variant in variants:
        assert _transport_only(uncached[variant]) == _transport_only(cached[variant])
    # The baseline is also identical ACROSS variants, which is the property
    # that makes one cached rollout legitimate for all of them.
    assert ([row["stock"] for row in cached["full"]]
            == [row["stock"] for row in cached["telemetry"]])
    # One baseline per (location, direction, forcing, holdout seed), not one per job.
    assert len(shared) == len(scoped["holdout_seeds"])
    assert all(row["stock"]["throughput_mbps"] > 0 for row in cached["full"])


def test_cached_baseline_is_copied_not_aliased(study, calibration):
    """Rows are serialised per job; a shared mutable baseline would let one
    job's report be corrupted by another's."""
    from qbbr.study import runner

    scoped = deepcopy(study)
    scoped.update(duration_s=6.0, holdout_seeds=[1000])
    job = dict(forcing="synthetic", location="Sydney", direction="downlink",
               seed=0, core="qa2c", variant="full")
    cache = {}
    first = runner.evaluate(scoped, job, calibration, _FixedAction(), stock_cache=cache)
    first[0]["stock"]["throughput_mbps"] = -1.0
    second = runner.evaluate(scoped, job, calibration, _FixedAction(), stock_cache=cache)
    assert second[0]["stock"]["throughput_mbps"] > 0


# ------------------------------------------------------------------ driver

def test_preflight_writes_a_plan_without_training(tmp_path, monkeypatch):
    from qbbr.scripts import run_rq_study_matrix as driver

    monkeypatch.setattr("sys.argv", [
        "run_rq_study_matrix", "--config", str(SCREEN), "--out", str(tmp_path),
        "--locations", "Sydney", "--training-seeds", "0", "--cores", "qa2c",
        "--variants", "full", "telemetry",
    ])
    driver.main()

    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["execution_mode"] == "preflight_only"
    assert plan["cost"]["jobs"] == 2
    assert plan["cost"]["total_wall_s_without_stock_cache"] > plan["cost"]["total_wall_s"]
    # Cost is driven by decisions, so the estimate must vary with the path's RTT.
    assert plan["decisions_per_episode"]["Sydney"] > 4000
    assert not (tmp_path / "manifest.json").exists()  # nothing was trained


def test_shards_partition_the_matrix_exactly(study):
    from qbbr.scripts.run_rq_study_matrix import shard_filter

    count = 4
    partitions = [[job for job in jobs(study) if shard_filter(study, index, count)(job)]
                  for index in range(count)]
    flattened = [tuple(sorted(job.items())) for part in partitions for job in part]
    assert len(flattened) == len(set(flattened)) == len(list(jobs(study)))
    assert max(map(len, partitions)) - min(map(len, partitions)) <= 1


def test_execute_rejects_replay_forcing_without_a_dataset(study, tmp_path):
    from qbbr.study.runner import execute

    scoped = deepcopy(study)
    scoped["forcing"] = ["replay"]
    with pytest.raises(ValueError, match="Replay requires"):
        execute(scoped, CALIBRATION, None, tmp_path)


def test_parallel_speedup_curve_is_measured_not_assumed_linear():
    """Oversubscribing past the performance cores LOSES throughput.

    Dividing by the shard count would claim 12 shards beat 7; they measure
    slower in aggregate, so the estimate must come from the measured curve.
    """
    from qbbr.scripts.run_rq_study_matrix import RECOMMENDED_PARALLELISM, speedup

    assert speedup(1) == 1.0
    assert speedup(7) > speedup(12)  # the curve peaks and falls
    assert speedup(7) < 7  # never linear
    assert speedup(64) == speedup(12)  # held flat, not extrapolated upward
    assert speedup(RECOMMENDED_PARALLELISM) == max(speedup(n) for n in range(1, 33))
