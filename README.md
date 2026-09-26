# QBBR: constrained QRL brain for BBR-v3

QBBR evaluates a constrained quantum reinforcement-learning (QRL) decision
layer for BBR-v3 under a declared Starlink simulator proxy. The learner is a
**brain**, not a replacement transport controller: it can select or sequence
only existing BBR-v3 pacing-gain decisions and cannot add a sixth action,
change an inflight-control interface, or redefine BBR-v3 itself.

## Evidence boundary

The repository does **not** claim a deployed Starlink or kernel-level BBR-v3
improvement. All learned-policy results are simulator-proxy results; no learned
policy has been run on a real TCP stack or a live Starlink link. The only real
transport runs (Tier 3) use stock Linux BBR, which on the testbed kernel is
mainline BBRv1 rather than BBR-v3. Operational
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
- **Action alphabet:** five bounded nominal pacing gains around the stock
  gain: `{0.75, 0.90, 1.00, 1.10, 1.25}`. A kernel adapter applies them as the
  fixed-point gains 192, 230, 256, 282, and 320 over 256.
- **Training reward:** depends on the protocol. The successor protocol below
  uses delivered throughput only. The final-v3 study reported in the thesis
  uses a stock-relative difference reward with a queueing-delay penalty (see
  [Reproduce the final-v3 study](#reproduce-the-final-v3-study-thesis-results)).

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

RQ4 in the thesis is a stratified analysis of the RQ1 results by location and
direction. The multi-flow RQ4 of the complete study plan has no counterpart
entry point. `MultiFlowFluidEnv` predates the native
action, state, and reward contract that `qbbr/study/protocol.py` enforces, and
the application profiles are declared names with no traffic model behind them.

## Reproduce the final-v3 study (thesis results)

The thesis results come from `outputs/rq_study/final-v3/`, produced by the
matrix runner above with the default `rq_study_screen.yaml` config and the
following overrides: six locations, both directions, training seeds 0–4,
held-out seeds 1000–1009, 30 episodes, and the queue-delay difference reward

$R_t=\log(x_t/x^s_t)-\delta\log\frac{q_t+q_0}{q^s_t+q_0}-\beta\frac{\ell_t-\ell^s_t}{\ell^s_t+1}$,
with $\delta=0.16$, $q_0=5$ ms, $\beta=0.5$, and $q_t=RTT_t-RTT_{\mathrm{prop}}$.
Each term is scored against the simulated stock proxy (gain 1.00) on the same
capacity forcing.

```bash
COMMON="--execute --allow-simulator-proxy --directions downlink uplink \
  --training-seeds 0 1 2 3 4 --holdout-seeds 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 \
  --curve-every 30 --reward-delta 0.16 --reward-delay-form queue --reward-queue-floor-ms 5"
# RQ1 main study (QA2C and matched A2C), with Tier-2 transfer onto trace-derived capacity
python -m qbbr.scripts.run_rq_study_matrix $COMMON --cores qa2c a2c --variants full \
  --dataset qbbr/data/raw --out outputs/rq_study/final-v3/1a_full --shard 1/7
# RQ2 policy-input ablation
python -m qbbr.scripts.run_rq_study_matrix $COMMON --cores qa2c a2c \
  --variants telemetry telemetry_queue --out outputs/rq_study/final-v3/1b_ablation --shard 1/7
# RQ3 secondary comparison (recurrent QDQN, 205 parameters)
python -m qbbr.scripts.run_rq_study_matrix $COMMON --cores qdqn --variants full \
  --out outputs/rq_study/final-v3/1c_qdqn --shard 1/7
# Training on trace-derived downlink capacity (runs 1-7), tested on runs 8-10
python -m qbbr.scripts.run_rq_study_matrix $COMMON --directions downlink --forcing replay \
  --cores qa2c a2c --variants full --dataset qbbr/data/raw \
  --out outputs/rq_study/final-v3/2_replay_train --shard 1/7
```

Run each command for shards `1/7` to `7/7`. The later `--directions downlink`
overrides the one in `$COMMON`. The analysis scripts then read these outputs,
in this order:

```bash
python -m qbbr.scripts.rtt_tolerance          # per-path RTT pass tolerance from the measured runs
python -m qbbr.scripts.tier3_model_error      # Tier 3: simulated stock vs Linux BBRv1 (needs the testbed report)
python -m qbbr.scripts.eval_validation_gate   # supplementary validation gate on trace runs 1-7
python -m qbbr.scripts.make_v3_figures        # tables, headline.json, tier/ablation/operating-point figures
python -m qbbr.scripts.make_v3_extra_figures  # action shares and training curves
python -m qbbr.scripts.make_final_comparison_figures --direction downlink
```

`tier3_model_error` compares against `reports/kernel_testbed_6city_clean.json`,
produced by `run_kernel_testbed.py`. `outputs/`, `reports/`, the raw traces,
and the calibration constants are not tracked by Git.

**Summary of the final-v3 results** (simulator only; see the thesis for the
full tables and caveats):

- **Tier 1, synthetic capacity:** QA2C raised pooled median throughput by
  3.7% on downlinks and 3.9% on uplinks, but RTT p90 rose by 4.7 ms on uplinks.
  Only 3 of 12 location–direction cells met every pre-specified criterion.
- **Tier 2, trace-derived capacity:** throughput fell by 1.3% (downlink) and
  2.0% (uplink) while RTT p90 fell. No cell met every criterion, and training
  directly on trace-derived downlink capacity did not recover the gain. A
  supplementary validation gate rejected all 120 agents, so the gated layer
  falls back to stock.
- **RQ2:** adding the queue, handover, and reconfiguration features to the
  policy input produced no appreciable median change.
- **RQ3:** no difference between QA2C and the matched classical A2C was
  detected at 126 parameters.
- **Actuation:** in the main simulator, a gain above 1.00 starts a probing
  cycle and the stock proxy does not probe on its own, so the policy controls
  probing rather than the CRUISE gain alone. See the
  [Sydney audit](docs/sydney_semantics_audit.md) and the
  [kernel adapter audit](docs/kernel_adapter_audit.md).

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
