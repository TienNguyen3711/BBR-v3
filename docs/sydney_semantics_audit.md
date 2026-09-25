# Sydney downlink baseline and action-semantics audit

This is a new exploratory audit, separate from the v14 manuscript experiments
and the September 14 trace-retraining preregistration. It does not replace
their results, relax their success criteria, or claim a field improvement.

The subsequent [kernel adapter audit](kernel_adapter_audit.md) replaces the
uncompiled adapter design described below. The simulator results and their
limitations remain unchanged.

## Why this audit is needed

The historical `probe_bw_cycle` simulator maps a requested gain above 1.0
to an UP/DOWN cycle. A constant gain-1.0 stock baseline never starts that
cycle. The environment also applies an exponential moving average to the
five categorical gains, producing intermediate numerical gains. Its original
phase mask uses `i_crs`, a continuous indicator, rather than the explicit
`probe_phase` state.

In contrast, `kernel/bbr_qrl/tcp_bbr_qrl.patch` proposes an override in
ProbeBW_CRUISE only; native probing phases keep their stock behavior. This
patch is an uncompiled design, not an audited working adapter. A controller
that wins by changing probe amplitude in the simulator has not demonstrated
that the CRUISE-only interface can reproduce that result.

The BBR state-machine description includes autonomous probing and separate
CRUISE, REFILL, UP and DOWN phases:
<https://www.ietf.org/archive/id/draft-ietf-ccwg-bbr-06.html#section-5.3.3>.
This draft is background, not verification of the kernel commit pinned by
the local patch.

## Opt-in model

`FluidParams.native_cruise_override=True` requires `consistent_transport` and
`probe_bw_cycle`. It introduces:

- Autonomous CRUISE → REFILL → UP → DOWN → CRUISE cycling for **every** arm,
  including stock. Gain choice no longer starts a probe, and no ground-truth
  capacity/underfill condition triggers an early probe.
- Stock REFILL gain 1.0, UP gain 1.25, and DOWN gain 0.90. Phase durations
  remain a simplified fixed-round proxy; these are not a port of kernel BBR.
- Exact selected categorical gains, without EMA smoothing or blending with
  the legacy phase indicator. This does not emulate kernel fixed-point gain
  quantization, ACK-driven update timing, or stale userspace writes.
- Explicit phase masking and per-substep enforcement: learner gains affect
  CRUISE only. Existing STARTUP, ProbeRTT, reconfiguration and congestion
  guards remain. The continuous cruise indicator is still an additional
  restriction, not a replacement for the explicit phase.

The legacy mode is the default and remains reproducible. No original protocol,
checkpoint, calibration, manuscript result or kernel file is replaced.
The new mode is a **semantic sensitivity model**, not validated BBR-v3:
bandwidth recovery, queue bounds, soft congestion signals and imposed
background retransmissions still inherit fluid-model limitations. Its stock
fidelity must be measured afresh before training or deployment claims.
The comparison changes this bundle of semantics together; it does not
attribute the result to one individual change (probing, smoothing, masking,
or gain application).

## Comparison

The runner uses full Sydney downlink BBR trace durations and the existing v14
calibration and capacity-inference method, held fixed between semantic modes.
Candidates share the same hard state/safety selector:

| Candidate | Behavior inside the admissible set |
|---|---|
| stock | Gain 1.0; native mode still autonomously probes |
| highest_permitted | Highest admitted action, otherwise stock |
| gain110 | Gain 1.10 when admitted, otherwise stock |
| balanced110 | Gain 1.10 when admitted; 0.90 under admitted drain conditions; otherwise stock |
| balanced125 | Gain 1.25 when admitted; 0.90 under admitted drain conditions; otherwise stock |
| quantum | All five saved synthetic-trained v14 QA2C seeds, deterministic deployment |
| classical | All five saved synthetic-trained v14 A2C seeds, deterministic deployment |

The trace-trained Sydney +1.33% result is not the transferred agent: that
runner saved metrics but not weights. No substitute weights are fabricated.
This comparison measures transfer of existing v14 agents, not their ability
to learn anew under revised dynamics.

## Constrained selection

1. Use runs 6 and 7 for validation and evaluate all five trained seeds.
2. Reject a candidate family if **any** validation trace/seed has increased
   RTT p90 or retransmissions/s against its own semantic-mode stock baseline.
   A `1e-9` tolerance accommodates floating point equality only.
3. Among feasible candidates, maximize median throughput delta, aggregating
   traces within seed and then seeds. Do not select the best seed.
4. Choose stock if no feasible candidate improves throughput.
5. Write `selection.json` before evaluating runs 8, 9 and 10. Report every
   candidate, plus the frozen choice; never replace it using evaluation scores.

Also report the stricter criterion: positive throughput and strictly lower
RTT p90 and retransmissions for **every** pair. Non-regression and simultaneous
improvement are different outcomes. Scalar training reward is unused here;
this is constrained policy selection over existing candidates, not a claim
to have trained a constrained RL algorithm.

All these traces have already been analyzed; they are **reused evaluation
data**, not a fresh confirmatory holdout. Calibration uses all runs, phase
alignment uses each full trace, and capacity is inferred from achieved BBR
throughput. These limitations are retained to isolate action semantics and
are recorded in the result manifest. A future confirmatory experiment needs
new traces, training-only calibration, and causal phase estimates.

## Reproduction

From the repository root:

```sh
MPLCONFIGDIR=/tmp/qbbr-mpl XDG_CACHE_HOME=/tmp/qbbr-cache OMP_NUM_THREADS=1 \
  .venv/bin/python -m qbbr.scripts.audit_sydney_semantics

MPLCONFIGDIR=/tmp/qbbr-mpl XDG_CACHE_HOME=/tmp/qbbr-cache OMP_NUM_THREADS=1 \
  .venv/bin/python -m qbbr.scripts.audit_sydney_semantics --execute --workers 4
```

The default output is `reports/sydney_semantics_audit/`; an existing directory
is refused to protect completed results. Supply a fresh `--out` for another
run. `--substep-s` permits a separately labelled numerical sensitivity check.
The manifest records calibration, config, trace and checkpoint hashes.
Individual episodes are saved as they finish; the final report includes
raw metrics, paired deltas, action shares and phase occupancy.
Phase shares describe the raw ProbeBW phase variable, which is retained during
STARTUP/ProbeRTT; they are not shares of all mutually exclusive TCP modes.
Selected-gain diagnostics describe the post-EMA request in legacy mode and the
unsmoothed categorical request in native mode, not an ACK-level pacing trace.

Regression coverage includes autonomous stock probing; gain-independent
UP/DOWN/REFILL, STARTUP and ProbeRTT behavior; exact gain application; explicit
phase fallback; unchanged legacy stock behavior; and fail-closed constrained
selection on incomplete, duplicate or nonfinite results.

## Findings from the September 16 audit

Validation chose `highest_permitted` in legacy mode and **stock** in the
CRUISE-only proxy. These choices were written before evaluation began.

On reused evaluation runs 8–10, the simple highest-permitted rule reproduces
the saved trace-trained QA2C seed-0 report's metrics and action shares to
numerical precision on all three traces: median throughput +1.3293%, RTT p90
−0.8990 ms, retransmissions −0.03178/s. This reproduces the reported +1.33%
headline without learning. It is not proof that the policies agree on every
possible state, and does not change the original failed >1.5% success gate.

| Simple policy in CRUISE-only proxy | Throughput delta | RTT p90 delta | Retransmissions delta |
|---|---:|---:|---:|
| Highest permitted | +1.4800% | +6.3657 ms | +0.16971/s |
| Gain 1.10 when permitted | +0.8942% | +1.4794 ms | +0.02333/s |
| Gain 1.10 / drain at 0.90 | −2.6291% | −5.6145 ms | −0.30851/s |
| Gain 1.25 / drain at 0.90 | −1.9330% | −1.0283 ms | −0.30423/s |

These are medians across the three evaluation traces. The aggressive rules
trade additional delay and retransmissions for throughput; the drain-capable
rules trade throughput for reduced delay and retransmissions. Neither is a
simultaneous improvement in all three metrics.

Halving the integration step on validation run 6 from 20 ms to 10 ms keeps the
highest-permitted-vs-stock changes close: throughput +0.9947% → +0.9839%,
RTT p90 +3.5647 → +3.6092 ms, retransmissions +0.18863 → +0.18714/s. This
supports numerical stability of that particular tradeoff, not model accuracy.

The audit demonstrates why the baseline and actuator semantics must be
settled before further training. It does not rule out all possible policies
or establish how a newly trained policy would perform in a validated kernel.
The historical successful-looking gain is reproducible by a simple rule;
the revised proxy supplies no validation-qualified throughput improvement
under the stated no-increase constraints.

Full local artifacts are in `reports/sydney_semantics_audit/`: `report.json`,
`summary.md`, the frozen `selection.json`, runtime/input hashes, and the
simple-rule equivalence and numerical-sensitivity checks.

All 150 planned episodes completed with finite metrics and complete seed/trace
coverage. Transferred v14 QA2C and A2C lose median throughput of 3.6289% and
4.4865%, respectively, in the CRUISE-only proxy; neither passes its safety
constraints. This is transfer without retraining, not evidence that either
learner could never adapt to different dynamics.

Verification passed: 113 targeted regression tests, followed by the updated
15-test audit suite (114 distinct tests in total). Input/source hashes and
the persisted validation selection were checked after execution. The
numerical sensitivity adds two episodes beyond the 150 comparison episodes.

Generate the comparison figure with:

```sh
MPLCONFIGDIR=/tmp/qbbr-mpl .venv/bin/python -m qbbr.scripts.make_sydney_audit_figure \
  reports/sydney_semantics_audit/report.json
```
