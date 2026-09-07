# Native-action RL-BBR migration

## Status boundary

The existing `qbbr.action`, simulator environments, and A2C/Q-A2C scripts are
the **legacy single-lever programme**. They are preserved for reproduction and
ablations, but must not be described as a deployed improvement to BBR-v3.
Their numeric pacing-gain/inflight controls define the candidate fixed BBR
decision set.  The active brain may select and sequence those existing values,
but may not add values, dimensions, or a new control function space.

The active programme starts with an executable contract in
`qbbr/configs/native_rl_bbr_contract.yaml` and an independently checked action
inventory in `qbbr/configs/native_bbr_actions.yaml`.

## Non-negotiable controls

- The active performance reward is supervisor-locked to delivered throughput
  only (`supervisor-locked-throughput-only-v1`). RTT, retransmission, queue,
  risk and handover observations remain state, safety and reporting signals;
  the archived alpha-fair reward belongs only to the legacy programme.
- An action may contain a numeric pacing-gain or `inflight_hi/lo` value when
  that value is an existing member of the frozen BBR action configuration.
  QRL selects/reorders these choices; it may not introduce another value or
  control dimension.
- `NativeActionSpace` maps the fixed inventory to one categorical RL head and
  masks unavailable semantics before sampling/training.  It therefore makes
  the MDP action restriction executable instead of a manuscript-only claim.
- Fit `NativeObservationNormalizer` on the training split only, then pass its
  transform to `NativeBBRControlEnv`; it converts z-scored telemetry into
  bounded angles for QNN encoding without leaking hold-out observations.
- An action remains disabled until its exact BBR-v3 kernel interface has been
  verified on the testbed. The guard falls back to stock BBR on missing
  telemetry, an undeclared action, an invalid state, or excessive loss.
- Field records are not complete unless they identify terminal, location,
  direction, CCA, cloud region, route fingerprint, and physical/path telemetry.
  Missing measurements remain missing; no model may fabricate them.

## Programme structure

| Namespace | Purpose | Evidence status |
| --- | --- | --- |
| `qbbr/experiments/legacy_single_lever` | Existing calibrated simulator and prior figures | historical/reproduction only |
| `qbbr/data/collection` | Multi-terminal longitudinal run schema and quality gate | ready for collector integration |
| `qbbr/validation` | Packet-trace queue diagnostics | ready; requires observed traces |
| `qbbr/control` | Fixed action contract, native inventory, fail-closed guard | ready; kernel actions deliberately disabled |
| `qbbr/agents/quantum/native_qa2c.py` + `qbbr/agents/classical/native_a2c.py` | Primary QA2C brain and exact matched Classical A2C baseline | primary simulator-proxy protocol |
| `qbbr/agents/quantum/recurrent_prioritized_ddqn.py` | GRU context + variational Double-DQN + prioritized replay | alternative ablation only; action-masked |
| `qbbr/experiments/native_rl_bbr` | New field-validated experiments | scaffolded; blocked on data and audited kernel interface |

## Simulator screen, not a deployment claim

`pilot_variational_ddqn.py` and `pilot_recurrent_prioritized_ddqn.py` are
small technical screens. They use the legacy `FluidSimEnv` only as a proxy:
its seven feature inputs, five fixed pacing-gain choices, STARTUP/reconfiguration
mask and EMA smoothing are retained. Both scripts set `reward_mode` to
`throughput_only`, write `evidence_tier: simulator_proxy_only`, and compare
against the existing stock 1.0 gain. A result must not be promoted to a BBR-v3
or Starlink claim until the collection and kernel gates above pass.

## Recurrent-QDQN ablation benchmark

`qbbr/configs/matched_recurrent_qrl_benchmark.yaml` pre-registers an
alternative recurrent-QDQN comparison. Both ablation arms use the same
recurrent history, throughput-only reward, five-action BBR mask, prioritized
replay, Double-DQN target, seed list and training budget. This benchmark is
not the QA2C/A2C primary evaluation.

Run the following first; it is read-only and does **not** train:

```bash
python -m qbbr.scripts.run_matched_qrl_benchmark
```

The runner starts training only when both `--execute` and
`--allow-simulator-proxy` are supplied. That explicit acknowledgement is
required because the resulting JSON remains simulator-proxy evidence.

## Full recurrent-QDQN ablation protocol

`full_recurrent_qdqn_ablation_protocol.yaml` is the canonical configuration
for the multi-hour recurrent-QDQN ablation. It preserves the same
throughput-only reward, seven-state MDP and five-action native BBR contract as
the primary path, but its result may only be reported as an alternative QRL
ablation. The historical `full_successor_qrl_bbr_protocol.yaml` name is a
compatibility alias and must not be used for new output names.

Use the following preflight first; it creates no checkpoint and performs no
training:

```bash
.venv/bin/python qbbr/scripts/run_recurrent_qdqn_ablation.py
```

Training requires both explicit flags:

```bash
.venv/bin/python qbbr/scripts/run_recurrent_qdqn_ablation.py \
  --execute --allow-simulator-proxy
```

The runner resumes a matching checkpoint only when its protocol ID and full
shared contract match. It reports the per-training-seed held-out throughput
delta, retransmission delta, action distribution, a deterministic bootstrap
95% confidence interval for median throughput delta, and a positive-seed
fraction. A result qualifies only when the bootstrap lower bound and median
are non-negative, at least 70% of independent training seeds improve over
stock, low gain 0.75 remains below 10%, and action-distribution stability
passes. These are selection/reporting gates; they do not add a non-throughput
reward component.

## Primary successor algorithm: native QA2C

The primary successor agent is `NativeQA2CAgent`, not recurrent QDQN. It is a
masked quantum actor--critic whose sampled and updated action is always one of
the five fixed native BBR choices admitted by the BBR-state mask. Its reward is
throughput only. Its canonical observation is the same seven derived features
used by `FluidSimEnv` (`s1_bhat` through `s7_reconfig_phase`); raw TCP/BBR and
Starlink telemetry are retained by the adapter to derive those features and
enforce the BBR-state safety mask. `NativeMLPA2CAgent` is its exact full-agent
parameter-matched Classical A2C counterpart (seven observations, five actions
and two quantum layers yield 126 trainable parameters per arm; the Classical
actor/critic hidden widths are 3/9). The primary Tier 1 config is
`tier1_native_qa2c_successor_protocol.yaml` and its runner is
`run_native_qa2c_successor.py`. Any full matrix remains deferred until the
Tier-1 gate passes and must inherit the active Tier-1 action-selection
contract rather than an earlier configuration revision.

Recurrent prioritized variational Double-DQN remains in the repository as an
explicit ablation/alternative QRL brain. Its prior simulator outputs do not
constitute primary-successor results and must not be substituted for the QA2C
evaluation.

### Compute-bounded Tier 1 gate

Run the default `tier1_native_qa2c_successor_protocol.yaml` before any larger
matrix. The active v5 Tier-1 gate evaluates QA2C and matched Classical A2C for
30 episodes per arm at London and Sydney, in both directions with five
independent training seeds: 40 jobs in total. It trains with a hard
native-action/safety mask, starts the actor with a stock-BBR prior, and admits
sub-1.0 native gains only under declared inflight/queue pressure. These are
state-conditioned restrictions on the frozen five-action inventory, not new
actions or reward terms. The full matrix may begin only if Tier 1 clears its
positive-seed, bootstrap, retransmission, and action-distribution criteria;
otherwise revise the simulator or training hypothesis rather than scaling an
inconclusive workload.

```bash
.venv/bin/python qbbr/scripts/run_native_qa2c_successor.py \
  --checkpoint-root outputs/tier1_native_qa2c_checkpoints \
  --out outputs/tier1_native_qa2c_report.json \
  --execute --allow-simulator-proxy
```

### Staged training gate

Training budget is increased in stages, not merely until an attractive result
appears. The first substantive screen is 30 episodes per arm and seed, with
ten disjoint deterministic hold-out environment seeds per trained policy. The
output requires: (1) non-negative median throughput delta against stock gain
1.0; (2) a non-negative bootstrap lower confidence bound; (3) improvement in
at least 70% of independent training seeds; (4) no systematic low-gain 0.75
selection; and (5) mean Jensen--Shannon divergence no greater than 0.20.
Retransmission and RTT are reported alongside the throughput objective and are
part of the promotion decision, but do not modify the throughput-only reward
or introduce an extra BBR action.

For the recurrent-QDQN ablation's current 20-second London/downlink proxy
screen, use:

```bash
.venv/bin/python qbbr/scripts/run_matched_qrl_benchmark.py \
  --locations London --seeds 0 1 2 --episodes 50 --duration-s 20 \
  --evaluation-episodes 5 --execute --allow-simulator-proxy \
  --out outputs/matched_qrl_london_3seed_50ep.json
```

Only an arm that passes all three checks is eligible for the 100-episode
screen. This is a model-selection rule for the simulator only, not a fairness,
field-performance, or Starlink result.

The matched configuration now also uses baseline-anchored Double-DQN selection.
The stock BBR action (native ID 2, gain 1.0) remains available whenever the
BBR mask permits it; a different existing gain is selected only if its Q-value
exceeds stock by the declared relative/absolute margin. This does not remove
gain 0.75, invent a sixth action, or add a loss/RTT penalty. A positive fixed
division of throughput reward by 100 Mbps is used only for numerical
conditioning during optimisation, preserving the reward ordering exactly.

For the current proxy screen, that Q-value test is also empirically certified:
after at least 16 replay observations for each candidate and stock action, the
candidate's observed mean (scaled) throughput reward must be non-inferior to
stock before it can be exploited. Exploration still samples every action
permitted by BBR. This guards against an overconfident value estimate selecting
the 0.75 gain; it is not a permanent mask or a new control decision.

## Mandatory action-effect gate

Before either benchmark arm may train, run
`qbbr.scripts.diagnose_action_sensitivity`. It force-selects each of the five
existing gains only when the BBR-state/reconfiguration mask permits it, then
compares its delivered throughput to stock gain 1.0. The gate is deliberately
stricter than observing any action response: at least one fixed action must
improve throughput over stock by the pre-registered margin. A degradation-only
response means the simulator cannot support the intended throughput-improvement
claim and must be revised or recalibrated before model training.

The successor dynamics are off by default (`FluidParams` defaults of `0.0`),
preserving legacy outputs. Three modelling parameters together make the
throughput-only reward select *state-aware* control rather than a constant
gain; each is a proxy until calibrated, not a fitted Starlink constant:

- `bandwidth_estimate_recovery_s` -- a lagged BBR delivery-bandwidth estimate
  after a handover capacity dip. It creates a bounded ProbeBW underfill window
  where a well-timed higher gain recovers throughput faster than stock. Use
  `fit_bandwidth_recovery` only with handover-labelled delivery-rate
  observations; it refuses insufficient evidence.
- `drain_throughput_penalty` -- sustained over-pacing that drives BBR into
  DRAIN (`i_dwn` high) also depresses delivered throughput (effective capacity
  is scaled by `1 - drain_throughput_penalty * i_dwn`), not just retransmits.
  Without this, delivered throughput was pinned at the raw bottleneck for any
  gain `>= 1`, so "always aggressive" tied "well-timed" on the reward and the
  policy collapsed to a seed-dependent constant. With it, a gain raised only
  near a handover beats every constant gain on throughput while keeping
  retransmits near stock.
- `probe_bw_phase_gate` (a `FluidSimEnv` argument, not a `FluidParams` field)
  -- non-1.0 native gains are offered only while BBR is genuinely in
  ProbeBW_CRUISE (`i_crs >= 0.5`), matching Table II's "PROBE_BW only" and
  stopping a gain override every decision interval regardless of BBR phase.

Until the recovery and drain constants are fitted,
`action_sensitivity_recovery_proxy.yaml` and the Tier-1
`dynamics_overrides` block are explicitly uncalibrated sensitivity proxies,
suitable for controller selection but not a Starlink claim. Verify with
`diagnose_action_sensitivity` (an s7-timed oracle must beat every constant
gain) and `replay_native_policy` (the trained policy must condition its high
gains on state, not pick one constant) before scaling the training budget.

## Immediate operational sequence

1. Make one manifest per run and collect terminal/location, weather/elevation,
   handover/SNR, traceroute/MTR, TCP_INFO and packet traces together.
2. Use `validate_collection_rows` to report coverage. Do not train the active
   agent on a dataset that fails its multi-terminal/path/telemetry gate.
3. Estimate service-time `c²_s`, exponential goodness-of-fit, and direct packet
   arrival rate with `queue_model_diagnostics`; replace no assumption merely by
   a default constant.
4. Audit the actual BBR-v3 kernel build and enable only action semantics whose
   tracepoint or sanctioned interface is demonstrated. Then train/evaluate
   against BBR-v3, Cubic, Hybla, Vegas and additional available baselines,
   including true shared-bottleneck mixed-flow experiments.
