# QBBR: constrained QRL brain for BBR-v3

QBBR evaluates a constrained quantum reinforcement-learning (QRL) decision
layer for BBR-v3 under a declared Starlink simulator proxy. The learner is a
**brain**, not a replacement transport controller: it can select or sequence
only existing BBR-v3 pacing-gain decisions and cannot add a sixth action,
change an inflight-control interface, or redefine BBR-v3 itself.

## Evidence boundary

The repository does **not** claim a deployed Starlink or kernel-level BBR-v3
improvement. The current experiments are simulator-proxy screens. Operational
claims require multi-terminal, longitudinal Starlink measurements; physical
telemetry; route-stability evidence; calibrated recovery dynamics; and an
audited BBR-v3 kernel adapter. Generated reports, raw traces, fitted
calibration constants, and checkpoints are intentionally excluded from Git.

## Active method

- **Primary QRL brain:** `NativeQA2CAgent`.
- **Matched classical baseline:** `NativeMLPA2CAgent`, constructed with the
  same full trainable-parameter budget as QA2C.
- **Alternative ablation only:** recurrent prioritized variational Double-DQN.
  It never replaces QA2C as the primary successor.
- **State:** seven normalised BBR/LEO features: delivery-rate estimate, RTT
  ratio, inflight/BDP, queue estimate, handover ETA, failure probability, and
  reconfiguration phase.
- **Action alphabet:** exactly five native BBR-v3 pacing gains:
  `{0.75, 0.90, 1.00, 1.10, 1.25}`.
- **Training reward:** delivered throughput only. RTT, retransmission, queue,
  risk, and handover are state, safety, calibration, or reporting signals;
  they are not reward terms.

The default Tier-1 protocol is
[`qbbr/configs/tier1_native_qa2c_successor_protocol.yaml`](qbbr/configs/tier1_native_qa2c_successor_protocol.yaml).
It is a 30-episode, five-training-seed simulator gate over London and Sydney,
in both directions. It uses a hard native-action/safety mask, a stock-BBR
initialisation prior, and state-conditioned admissibility for low and high
native gains. Its purpose is to test stability before any larger matrix is
approved.

## Repository layout

```text
qbbr/
  agents/       QA2C, matched Classical A2C, and recurrent-QDQN ablation
  configs/      machine-readable MDP, action, and experiment contracts
  control/      fixed native-action inventory, masks, and safety selection
  data/         collection schemas; raw data and calibrations are ignored
  env/          simulator proxy and guarded native-BBR environment interface
  eval/         hold-out evaluation, statistics, and success gates
  reward/       throughput-only primary reward; legacy reward code is separate
  train/        primary A2C and QDQN training loops
  validation/   transition, queue, and recovery validation utilities
  scripts/      preflight, training, diagnostic, and collection entry points
  tests/        executable contract and regression tests
docs/
  native_rl_bbr_migration.md  method, evidence, and migration boundary
```

## Setup

Requires Python 3.9 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pennylane==0.38.0` and `autoray==0.6.12` are pinned because this codebase
targets Python 3.9. For a reference quantum backend use
`QBBR_QUANTUM_DEVICE=default.qubit`; the configured fast backend is validated
separately before use.

## Verify

```bash
pytest -q
```

## Run the complete four-RQ study plan

Before launching any job, run the read-only contract audit:

```bash
python -m qbbr.scripts.check_four_rq_readiness --strict
```

The complete study has one machine-readable plan covering the existing RQ1
baseline-transfer jobs, RQ2 state ablations, RQ3 recurrent-QDQN ablation, RQ4
mixed-flow/application experiments, and the field evidence gate. A preflight
is read-only:

```bash
python -m qbbr.scripts.run_complete_rq_study
```

It writes `outputs/complete_rq_study_plan.json`. Simulator training remains
explicit and must be acknowledged separately through the component runners.
Field evidence is accepted only as JSONL records and is assessed without
imputing missing weather, SNR, handover, or route observations:

```bash
python -m qbbr.scripts.run_complete_rq_study --field-records path/to/field_runs.jsonl
python -m qbbr.scripts.assess_collection path/to/field_runs.jsonl --require-ready
```

The collection report now includes longitudinal span and distinct collection
days per terminal. These are reported independently from structural telemetry
coverage; the active study protocol applies the two-terminal, two-location,
multi-week gate before any field result can be labelled as evidence.

## Run the RQ1/RQ2/RQ3 matrix

`run_complete_rq_study` emits the plan; `run_rq_study_matrix` executes it. One
matrix answers three research questions, because they are contrasts within the
same set of runs rather than separate experiments:

- **RQ1** `full` variant, `qa2c` and `a2c` cores, synthetic forcing, plus the
  unchanged-policy transfer onto measured capacity when `--dataset` is given.
- **RQ2** the six ablated state variants against `full` at a matched core and
  seed. Ablation restricts the actor/critic input only: the safety mask still
  sees the raw seven features, identically in every arm.
- **RQ3** `qa2c` against `a2c` at the shared 126-parameter budget, and the
  recurrent `qdqn` core as the estimator ablation.

[`rq_complete_study.yaml`](qbbr/configs/rq_complete_study.yaml) declares the
full 1800-job design: about 800 single-core hours at the measured per-decision
timings. [`rq_study_screen.yaml`](qbbr/configs/rq_study_screen.yaml) is the
90-job screen that precedes it.

Cost is driven by DECISIONS, not by simulated seconds. The decision interval is
a function of the path's minimum RTT, so one 300 s episode is 4967 decisions on
Sydney (30 ms) and 615 on SaoPaulo (388 ms) -- the same protocol costs up to 8x
more on one city than another, and Sydney alone is half of the full matrix.
Scope a run with that in mind: adding the five non-Sydney downlinks costs about
as much as Sydney by itself. The preflight is read-only and prints the job
breakdown with wall-clock hours by location and by core:

```bash
python -m qbbr.scripts.run_rq_study_matrix
```

Training is explicit, and the matrix shards across processes. Each shard keeps
its own manifest and writes into one shared result tree:

```bash
python -m qbbr.scripts.run_rq_study_matrix \
  --execute --allow-simulator-proxy --shard 1/6
```

Pass `--dataset qbbr/data/raw` to add the RQ1 transfer evaluation, in which the
synthetic-trained policy is replayed unchanged on measured capacity. Every
scope flag (`--locations`, `--variants`, `--cores`, `--episodes`, ...) enters
the protocol digest, so a narrowed run has its own identity and cannot share an
output tree with the full matrix. Jobs resume exactly from their checkpoints;
a completed job is reused rather than retrained.

The recurrent-QDQN core runs here under the v14 difference reward. The separate
[`full_recurrent_qdqn_ablation_protocol.yaml`](qbbr/configs/full_recurrent_qdqn_ablation_protocol.yaml)
still declares the earlier throughput-only reward, under which every pilot
policy froze at stock; prefer this matrix for new QDQN work.

RQ4 has no counterpart entry point. `MultiFlowFluidEnv` predates the native
action, state, and reward contract that `qbbr/study/protocol.py` enforces, and
the application profiles are declared names with no traffic model behind them.

## Run the primary Tier-1 protocol

Preflight validates the frozen seven-state/five-action contract and does not
train:

```bash
python -m qbbr.scripts.run_native_qa2c_successor
```

Training is explicit because it creates simulator-proxy artefacts:

```bash
python -m qbbr.scripts.run_native_qa2c_successor \
  --execute --allow-simulator-proxy
```

The runner writes reports and resumable checkpoints below `outputs/`, which is
ignored by Git. It reports seed-level throughput relative to stock BBR-v3,
retransmissions, RTT, native-action shares, bootstrap intervals, policy
diagnostics, and pre-specified qualification gates.

## QDQN ablation

The recurrent-QDQN path shares the same action set, state contract, and
throughput-only reward, but it is an alternative ablation:

```bash
python -m qbbr.scripts.run_recurrent_qdqn_ablation
```

See [`docs/native_rl_bbr_migration.md`](docs/native_rl_bbr_migration.md) for
the active/legacy boundary, data requirements, and the field-evidence gate.

## Reproducibility and scope

The legacy RQ1--RQ4, alpha-fair, beta/gamma, and boundary-freeze workflows are
retained only for historical reproduction. They must not be mixed with the
active successor results or used to alter the frozen successor reward/action
contract. Use the active protocol and its machine-readable contracts for new
work.

## Sydney baseline/action-semantics audit

The separate [Sydney audit](docs/sydney_semantics_audit.md) compares saved v14
learners with simple masked policies under both historical dynamics and an
opt-in CRUISE-only fluid proxy with autonomous stock probing. It selects
throughput subject to explicit RTT-p90 and retransmission non-regression
constraints. This exploratory audit uses reused traces; it does not replace
the original protocols or establish kernel/field performance.

```bash
python -m qbbr.scripts.audit_sydney_semantics              # preflight
python -m qbbr.scripts.audit_sydney_semantics --execute    # separate report
```

The follow-up [kernel adapter audit](docs/kernel_adapter_audit.md) builds the
pinned BBR-v3 source and a separately named control module in an isolated
Linux guest. It checks actual command application and documents the remaining
simulator differences that must be resolved before retraining.

The [action/testbed alignment update](docs/action_testbed_alignment.md) adds an
opt-in kernel actuator contract and measured TCP telemetry with a watchdog
heartbeat. Use its dynamics overlay for new pilots; historical experiment
configurations retain their original behavior.
