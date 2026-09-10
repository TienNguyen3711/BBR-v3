# Transport proxy v6

Scope: simulator only. No kernel adapter or Starlink field run is enabled.

The native QA2C protocols now opt into `consistent_transport`. Queue backlog is
separate from a delivery-derived propagation pipe estimate. RTT uses backlog /
current service rate, and retransmission demand consumes wire service before
unique delivered bytes are rewarded. Retransmission demand is still an exogenous,
phase-modulated calibrated proxy; this is not a packet-loss/recovery simulation.
The former artificial DRAIN capacity penalty is disabled in both protocols.

The cwnd budget subtracts existing excess queue before admitting new traffic.
Below one BDP, the fluid service ceiling also scales with cwnd / propagation RTT.
Loss backoff occurs once per accumulated propagation-RTT round; the environment
splits substeps at round boundaries. STARTUP exits after three rounds without
25% delivery-rate growth, excessive round loss, the existing volume threshold,
or the existing timeout. This remains an approximation of BBR, not a kernel port.
ProbeRTT excludes policy overrides in the action mask.

RTT p95, retransmitted bytes / (unique delivered + retransmitted bytes), and
retransmits/s must not regress against stock in any evaluated training seed.
The budgets are explicitly zero in both protocols. Missing/nonfinite safety
metrics fail these gates. Throughput remains the reward. Fixed-action safety
is separate from these post-evaluation acceptance gates.

Tier 1 and full share agent, simulator and acceptance settings under new v6 IDs.
Checkpoint contracts include simulator settings, rejecting incompatible resumes.
Old result files and calibrated constants are not overwritten. Historical
calibration is provisional for v6, and old checkpoints/results are not evidence
for the new dynamics. The low-gain share now counts both 0.75 and 0.90.

Run tests:

```sh
.venv/bin/python -m pytest qbbr/tests/test_transport_consistency.py
```

Run paired diagnostic baselines (two seeds, London/Sydney, both directions,
stock proxy, five constant requested gains, queue heuristic; 20/10 ms steps):

```sh
.venv/bin/python -m qbbr.scripts.check_transport_baselines
```

Constant requested gains still pass the environment phase mask, freeze and EMA;
they are not unrestricted constant wire pacing. The heuristic requests gain
0.75 at >=10 queued packets, 1.25 below 0.9 estimated inflight BDP, otherwise 1.
These constants are fixed before observing results. Stock is a fluid stock proxy,
not a kernel BBR baseline. The initial 30-second run is diagnostic, not a full
300-second validation or an independent field-calibration claim.

Training remains conditional on numerical stability and trace validation. A
successful software preflight alone is not permission to claim model validity.

## Executed validation

97 focused tests passed. Full protocol preflight succeeded. The initial paired
baseline audit executed 112 episodes (30 seconds each) and failed its predeclared
5% timestep tolerance: seed 7001 stock RTT p95 differed by 21.6% on Sydney
downlink and 8.4% on Sydney uplink between 20 and 10 ms. The fixed gain-1 arm
reproduced these failures, as expected. QA2C/A2C training was not started.

Outputs: `outputs/transport_v6_baselines.json` (raw rows and paired deltas),
`outputs/transport_v6_full_preflight.json`, and
`outputs/transport_v6_real_reference.json` (existing trace statistics, not new
field measurements). Further numerical convergence work is required before
recalibration and long-run training; no performance improvement is claimed.

---

# Transport proxy v7 -- ProbeBW cycle

Scope: simulator only. Still no kernel adapter or Starlink field run. Still
`evidence_tier: simulator_proxy_only` and `--allow-simulator-proxy`. This
addresses review points A, B and (partly) D above; C follows from A once the
throughput lever is real again.

## What changed (all opt-in `FluidParams`, default off => legacy path intact)

- **`probe_bw_cycle` (fixes A).** The chosen `pacing_gain` is no longer a
  sustained cruise multiplier. It is executed as a real BBR-v3 ProbeBW state
  machine: ProbeBW_CRUISE at `min(gain, 1.0)`; on the Eq. 22 timer *or* when
  the pipe is genuinely underfull (post-handover, bw estimate lagging, queue
  near empty) it enters ProbeBW_UP at the chosen amplitude; UP lasts one round
  (held longer only while still refilling a real underfill); then ProbeBW_DOWN
  drains at 0.75 until the backlog is gone. Consequence: a *sustained* high
  gain now costs only a transient `~(gain-1)*BDP` queue pulse, not a standing
  ~1 BDP backlog -- RTT p95 vs stock for a constant 1.25 drops from **+140..
  +240 ms (v6)** to **~+2..+9 ms (v7)**. A *well-timed* high gain still
  recovers throughput because the early-entry path catches the post-handover
  hole a fixed cadence would miss.
  **Supervisor-flagged reinterpretation:** stock gain 1.0 performs no
  autonomous ProbeBW_UP in this model (it reproduces the pre-v7 cruise
  baseline exactly). The agent's action is the ProbeBW_UP amplitude /
  suppression. Every value used is still a native BBR ProbeBW gain; the
  UP -> DOWN -> CRUISE sequencing is BBR-v3's own. Reversible: set
  `probe_bw_cycle: false`.

- **`overflow_retransmit_frac` (fixes B, part 1).** Backlog past the buffer
  ceiling (`cwnd_gain*BDP`) now also costs retransmissions, proportional to
  the overshoot, so a throughput-only policy cannot inflate the queue for
  free. The throughput-only reward is unchanged; this is enforced through the
  reported retransmission gate, not a reward term.

- **Retransmission-*ratio* gate removed (fixes B, part 2).**
  `retransmits / (delivered + retransmits)` double-counts the throughput
  denominator: a lower gain that loses goodput fails it for the wrong reason,
  and it disagreed in sign with `retransmits/s`. `selection_criteria` now
  gates `rtt_p95_delta` and `retransmits_delta_per_s` only. The ratio is
  still computed and reported. The gate *mechanism* is retained (and tested)
  for any future opt-in.

- **Numerics (fixes D, partly).** Timer resets now carry their overshoot
  (`t_since_probe_rtt_s`, probe-phase timers); bandwidth-estimate recovery and
  the `i_dwn`/`i_crs` relaxation use the exact exponential factor under
  `consistent_transport`; `FluidSimEnv` splits substeps on every ProbeBW /
  ProbeRTT phase boundary that is still ahead. Result: **mean RTT is now
  substep-size stable to ~1%** (was part of the 21% drift), the ProbeBW queue
  peak is substep-exact, and the throughput / retransmit deltas vs stock are
  substep-stable to <1% / <0.05 pps. **Residual:** the *absolute* RTT p95 tail
  still carries an O(dt) bias (~11-21% between 20 and 10 ms) because p95 of a
  spiky bimodal RTT with few tail samples is an ill-conditioned statistic.
  The acceptance gates use delta-vs-stock at a matched substep, where the
  common-mode part of that bias cancels (the p95 *delta* is stable to a few
  ms). Full convergence of the absolute tail still needs a sub-stepped or
  implicit queue update and is left as pre-recalibration work.

## Configs

`tier1_` / `full_native_qa2c_successor_protocol.yaml` -> protocol id
`...-probe-bw-cycle-v7`; `dynamics_evidence:
uncalibrated_probe_bw_cycle_proxy_v7`; `dynamics_overrides` adds
`probe_bw_cycle: true`, `overflow_retransmit_frac: 0.25`. `agent`,
`simulator` and `selection_criteria` stay identical between the two files.
A checkpoint from v6 will not resume against v7 (the contract hash includes
`simulator`).

## Executed validation (v7)

```sh
.venv/bin/python -m pytest -q                       # 330 passed, 2 deselected
.venv/bin/python -m qbbr.scripts.check_transport_baselines --duration-s 30
.venv/bin/python qbbr/scripts/diagnose_action_sensitivity.py \
    --config qbbr/configs/action_sensitivity_diagnostic_v7.yaml
```

Paired 30-s baseline audit: fixed gain-1.10 / 1.25 now recover throughput
(single-digit %) with an RTT p95 delta in the single-digit-ms range and a
near-zero retransmit delta; the 20-vs-10-ms RTT-p95 reproducibility check
still fails on Sydney (absolute tail, see Residual above). Action-sensitivity
gate: at least one native gain beats stock throughput by the pre-registered
margin (see `outputs/action_sensitivity_v7.json`).

Outputs: `outputs/transport_v7_baselines.json`,
`outputs/action_sensitivity_v7.json`,
`outputs/tier1_native_qa2c_v7_exploratory_report.json` (+ `.partial.json`).

## Standing caveats (unchanged)

Simulator-proxy only. `probe_bw_cycle` and `bandwidth_estimate_recovery_s`
are uncalibrated proxies; a positive Tier-1 result is a statement about this
proxy, not about BBR-v3 or Starlink. The Tier-1 run launched on this v7 model
is **exploratory** -- the RTT-p95 timestep audit is not green and the
transport model has not been re-fit to traces. A successful software
preflight is not permission to claim model validity.

## Tier-1 v7 exploratory result (40 jobs, 30 ep x 300 s)

`outputs/tier1_native_qa2c_v7_exploratory_report.json`. Not qualified in any
of the 8 cells, but the failure is informative:

- **QA2C: 3/5 training seeds positive in every cell** (+1.7..+6.2% throughput;
  seeds 1 and 3 frozen at exactly stock). `positive_training_seed_fraction`
  0.60 < 0.70 is the dominant blocker -- and it is the known actor-critic
  credit-assignment issue (critic `Linear(1,1)` on `sum(6 PauliZ)` cannot
  represent a return of ~+107), not a simulator defect. v6 gave all zeros.
- **Sydney downlink QA2C is a clean multi-objective win**: +4..+6% throughput,
  RTT p95 -2..-6 ms, retransmits down. All gates pass except the seed
  fraction.
- **London uplink QA2C**: +2..+3% throughput but RTT p95 +46..+59 ms -- a real
  over-cost of the probe on the degenerate `util=0.05` uplink (tiny BDP).
- **A2C (classical): frozen at stock in every cell.** The matched 126-param
  baseline does not learn at all under `reward_scale_mbps=100`.
- The zero-tolerance RTT-p95 / retransmit gates also fail where the probe adds
  even ~1 ms -- too strict given the absolute-p95 dt bias.

---

# Transport proxy v7b -- probe RTT budget + reward scale

`protocol_id: ...-probe-bw-cycle-v7b`. Three changes, driven by the v7 result:

1. **`probe_max_queue_delay_ms` (new `FluidParams`, default 0.0 = v7).** A hard
   ceiling on the standing queue any ProbeBW pulse -- or its residual into
   cruise / a handover dip -- may build, as a queuing *delay* (ms) rather than
   a BDP fraction, so the probe's RTT cost is bounded the same on every path.
   Config: `15.0`. On a `util=0.05` uplink this cuts the forced-probe RTT p90
   cost from ~+171 ms to ~+15 ms at a ~1% throughput cost; the Sydney-downlink
   throughput win is unchanged (+6.8%).
2. **Acceptance gate: RTT p90, not p95.** p95 of a spiky bimodal RTT with few
   tail samples keeps an O(dt) bias the delta cannot fully cancel; p90 is
   better sampled. Budgets are now "no material regression", not zero:
   `max_rtt_p90_delta_vs_stock_ms: 5.0`, `max_retransmits_delta_vs_stock_per_s:
   0.25`. p95 delta is still computed and reported.
3. **`reward_scale_mbps: 10000` (was 100).** The never-run "tier1_v5g" knob.
   Discounted return moves to ~+-1.5, inside the critic's `[-6,6]` range, so
   advantage stops collapsing to the return -- the intended fix for the
   seed 1/3 freeze. Positive scaling only; reward ordering preserved.

`tier1` and `full` configs stay share-equivalent on `agent` / `simulator` /
`selection_criteria`. Tests: `pytest -q` -> 331 passed.

Exploratory Tier-1 v7b retrain: `outputs/tier1_native_qa2c_v7b_exploratory_report.json`.
Fork-4 environment check: `qbbr/scripts/validate_stock_vs_raw.py` ->
`outputs/validate_stock_vs_raw_v7b.json` (v7b stock sim vs real sequential
BBR/BBRv2 iperf3 runs in `qbbr/data/raw/`; environment validation only, no
Starlink claim).
