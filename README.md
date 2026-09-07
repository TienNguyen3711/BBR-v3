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
