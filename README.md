# qbbr — The Limits of a Single Lever

A pre-registered evaluation of learned `pacing_gain` adaptation for BBR-v3 over
Starlink LEO networks. Builds on the six-city Starlink measurement study of
De Silva, Pokhrel & Kua (IEEE TMC), which found BBR-v3 outperforms Cubic,
Vegas and Hybla but pays for it with a fixed, environment-agnostic probing
budget (~2% loss/RTT) whose retransmission cost varies 42x across locations
(r=-0.78 with path RTT). This project tests four pre-registered interventions
on one control loop — state, agent, reward, action — to see whether any of
them let a learned agent adapt that budget productively.

**Status (2026-08-19):** all four RQs complete at full statistical power
(10 seeds/location, 6 locations). Title, framing, and scope decisions are
pending supervisor sign-off — see `main.tex`'s title-block comment and
`outputs/supervisor_meeting_questions.md`.

**Branch `align/rq3-rq4-docs` (2026-08-31):** adds two finished-but-unrun code
paths — a `closed_form_dynamic` RQ3 risk model (atmospheric + ISL + handover
hazard) and exact full-agent parameter matching for RQ4 (`qbbr/agents/matching.py`,
`--rq4-full-match`) — plus this doc alignment. Reported RQ1–RQ4 numbers are
unchanged; see "Deferred: full-protocol runs" below.

## Findings, in one paragraph

`pacing_gain` alone (RQ1a, pre-registered) cuts retransmissions at 1 of 6
locations (London, −37.1%, p=0.024, throughput retained 99.5%) — direction
correct everywhere, magnitude/significance the binding constraint elsewhere.
Three candidate extensions — richer state via closed-form risk features
(RQ3), an α-fair reward embedded in the objective (RQ2), a parameter-matched
quantum policy core (RQ4) — are each null or practically negligible at full
power. A post-hoc boundary-freeze mechanism initially looked positive but
did not survive a decoy-phase control (p=0.42–0.82 across all six locations)
and is reported retracted. Full numbers: `outputs/rq_summary_onepager.md`.
Full writeup: `main.tex` / `main.pdf`.

## Repository layout

```
qbbr/
  action/     action-space registry (pacing_gain-only, multi-head, inflight)
  agents/
    classical/  MLP actor-critic (mlp_a2c.py)
    quantum/    7-qubit quantum actor-critic (qa2c.py, ansatz.py)
    matching.py exact full-agent parameter match for RQ4 (classical <-> quantum)
  configs/    action spaces + reward variants (YAML)
  data/       raw traces (gitignored) + per-location calibration constants
  env/        fluid-model simulator (fluid_sim.py, fluid_env.py = Scenario A,
              multi_flow_env.py = Scenario B / coexistence, competing_ccas.py,
              testbed_env.py = real-kernel guardrail path)
  eval/       Scenario A/B evaluation, metrics, ablation harness
  features/   state-vector construction (telemetry + risk features)
  reward/     alpha-fair reward
  risk/       closed-form risk model (atmospheric, ISL, handover)
  scripts/    training, evaluation, calibration, and diagnostic entry points
  train/      A2C training loop, rollout buffer
  tests/      pytest suite (248 tests)

main.tex / main.pdf   paper draft (source is gitignored; PDF is build output)
outputs/              generated reports, tables, and meeting-prep docs (gitignored)
figures/               figures used by main.tex
```

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Requires Python >=3.9. `pennylane==0.38.0` and `autoray==0.6.12` are pinned
together — see `requirements.txt` for why.

## Running tests

```
pytest qbbr/tests/ -q
```

## Reproducing the headline results

Each RQ has its own parallel evaluation script under `qbbr/scripts/`:

| RQ | Script | Notes |
|---|---|---|
| RQ1a/RQ1b | `eval_rq1_parallel.py` | pacing_gain-only vs. stock BBR-v3, both cores |
| RQ2 | `eval_rq2_parallel.py` | coexistence, alpha in {0, 1, inf} (alpha=2 excluded — see script comment) |
| RQ3 | `train_rq3_checkpoints.py` then `eval_rq3_parallel.py` | risk-on (`closed_form` / `closed_form_dynamic`) vs. risk-off (`stub_constant`) |
| RQ4 | via `train.py` + `evaluate.py`, `--core classical\|quantum` | single grid point run so far (alpha=1, L=2, no re-uploading) |
| boundary-freeze | `eval_boundary_freeze_perseed.py` | per-seed effect size/IQR, baseline vs. freeze extension |
| shifted-freeze control | `eval_shifted_freeze.py` | decoy-phase control for the boundary-freeze mechanism |

The RQ3, boundary-freeze and shifted-freeze scripts are post-hoc/exploratory
(added after RQ1a's results) and are committed on `main`. They rebuild their
own checkpoints/reports from scratch — ask before re-running RQ3 or the freeze
diagnostics, since each retrains checkpoints.

`closed_form` remains the archived atmospheric-only risk feature used by the
reported RQ3 runs. `closed_form_dynamic` is the replacement candidate for new
experiments: it composes the atmospheric and representative-ISL baselines with
a bounded time-to-handover hazard. It changes the risk-on distribution, so it
requires newly trained risk-on checkpoints and must not be compared directly
with archived `closed_form` checkpoints.

For a new Scenario-A RQ4 run, train the classical arm with
`--rq4-full-match` and use the same flag when loading that checkpoint for
evaluation. This enforces equality of **all** trainable actor and critic
parameters with the corresponding QA2C core; it is intentionally incompatible
with the archived RQ4 classical checkpoints.

### Deferred: full-protocol runs for the new code paths

The `closed_form_dynamic` RQ3 mode and the `--rq4-full-match` RQ4 baseline are
wired and tested but **not** run at the 10-seed protocol. Ready-to-run:

```
# RQ3 with the dynamic risk model (6 cities x 10 seeds) -- retrains both arms
python -m qbbr.scripts.train_rq3_checkpoints --risk-mode closed_form_dynamic \
    --out-root outputs/checkpoints_rq3_dynamic/risk_on
python -m qbbr.scripts.train_rq3_checkpoints --risk-mode stub_constant \
    --out-root outputs/checkpoints_rq3_dynamic/risk_off
python -m qbbr.scripts.eval_rq3_parallel \
    --risk-on-checkpoint-root outputs/checkpoints_rq3_dynamic/risk_on \
    --risk-off-checkpoint-root outputs/checkpoints_rq3_dynamic/risk_off \
    --risk-on-mode closed_form_dynamic --out outputs/rq3_dynamic_report.json

# RQ4 L=2/L=3 x re-uploading grid, full-agent-matched classical arm
python -m qbbr.scripts.run_ablation --config qbbr/configs/base.yaml \
    --location <city> --direction downlink --n-runs 10 --n-episodes 200 \
    --alpha 1 --n-layers 2,3 --reupload false,true --core quantum,classical \
    --checkpoint-root outputs/ablation_rq4_full
```

`train_rq3_checkpoints.py` and `eval_rq3_parallel.py` both take `--locations`
and `--seeds` / `--n-seeds` for smaller pilot subsets. The RQ4 `L=3` cells
cost ~7-8x compute/episode; completing that grid vs. reporting the current
point as a scoped limitation is a decision pending with the supervisor
(`outputs/supervisor_meeting_questions.md`, Q2).

`validate_simulator.py` and `validate_handover_cadence.py` reproduce the
trace-calibration and handover-cadence validation reported in `main.tex`
Sec. "Trace-Calibrated Fluid-Model Simulator".

Checkpoints under `outputs/checkpoints/` (both cores) that predate the
`actor_trunk`/`actor_heads` architecture split are in a legacy state-dict
format; `MLPA2CAgent.load()` / `QA2CAgent.load()` both detect and handle
this format automatically (see `qbbr/tests/test_mlp_a2c.py` and
`test_qa2c.py` for the regression tests), so no special handling is needed
to load them.

## Known limitations

- RQ4's ablation grid (circuit depth L, data re-uploading) has one point
  tested at full power; L=3 alone costs ~7-8x compute/episode. Completing
  the grid vs. reporting the current point as a scoped limitation is a
  decision pending with the supervisor.
- The boundary-freeze and shifted-freeze-control findings are both
  post-hoc/exploratory (designed after seeing RQ1a's results); a
  pre-registered confirmatory replication is future work.
- The simulator's RTT dispersion is narrower than the real traces
  (see `main.tex` Sec. "Trace-Calibrated Fluid-Model Simulator") — median
  comparisons are supported, tail/variance claims are not.
- Branch hygiene: the `agent/*` worktree branches and `feature/pipeline-completion`
  are fully merged into `main` and can be deleted. `feature/gamma5-sweep-uplink-eval`
  carries the gamma-sweep / uplink-eval / oracle work (referenced by `main.tex`)
  and is not yet merged to `main`. `align/rq3-rq4-docs` branches off it and adds
  the `closed_form_dynamic` RQ3 mode, RQ4 full-agent parameter matching, and this
  doc pass. All evaluation/diagnostic scripts are committed (no working-tree-only
  scripts remain).
