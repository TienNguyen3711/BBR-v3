# BBR-v3 kernel adapter audit

This follow-up to the [Sydney semantics audit](sydney_semantics_audit.md)
checks the actual actuator before new training. It is a laboratory adapter
bring-up, not a Starlink performance result. Runtime evidence is collected in
`reports/kernel_adapter_audit/`; build instructions are in
[`kernel/bbr_qrl/README.md`](../kernel/bbr_qrl/README.md).

## Completed checks — 17 September 2026

The pinned kernel booted, both modules loaded, and all **19 TCP flows** passed
the runtime assertions. All five fixed-point actions were observed in eligible
CRUISE. Clear, one-second expiry and deferred mid-CRUISE command changes passed.
The loss experiment recorded 180 retransmitted segments and 1,236 guarded
CRUISE ACKs. All 566 sampled observations with two active adapter sockets used
native gains. Socket ownership returned to zero, and every native phase was
exercised. The host regression suites passed **63 tests**.

The three-run medians are:

| Arm | Receiver Mbit/s | Sampled smoothed-RTT p90 (ms) | Retransmitted segments |
|---|---:|---:|---:|
| Stock `bbr` | 18.3814 | 82.689 | 0 |
| `bbr_qrl`, disabled | 18.3633 | 85.759 | 0 |
| `bbr_qrl`, gain 1.0 | 18.3528 | 88.074 | 0 |

Relative to stock, disabled control changed throughput by −0.099% and RTT p90
by +3.070 ms; gain 1.0 changed them by −0.156% and +5.385 ms. Stock RTT p90
ranged from 81.183 to 91.988 ms across the three repetitions. These results
support successful actuator bring-up, **not a passed performance-equivalence
gate or an RTT improvement**. No training or Starlink deployment was run.

The report stores individual runs, sampled state, counters, medians and ranges.
`provenance.json` records source and artifact hashes, the verification image,
test suites and timer configuration. The copied module contains the expected
diagnostic counter and its compiled source matches the workspace. Failed or
interrupted harness attempts remain under `tmp/kernel_adapter_audit/` and are
excluded from the completed report.

## Problems corrected

The original annotated patch was not an executable unified diff. Its weak
no-op hooks could not be overridden by loading a separate controller module.
The replacement builds a single composite `bbr_qrl.ko` with the control object
linked directly to a separately named copy of the pinned BBR implementation.
Stock `bbr` remains available for comparison.

The adapter accepts exactly five declared decimal actions and a disable
sentinel. Invalid input leaves the request unchanged. Tests compile the
actual C parser, including a 6,001-value decimal sweep, rather than testing
a Python reimplementation. Requests have a one-second lease, latch on actual
CRUISE entry, and cancel on clear, expiry or loss/recovery/CE exclusion.
Cancellation takes effect on ACK processing. Multiple active adapter sockets
fail closed; this is a single-flow laboratory interface.

Socket release is guarded by BBR initialization: TCP can release an algorithm
selected before connect without first calling its init callback. Uninitialized
release must not decrement the count of another live socket.

## Pinned source and isolation

The tested source is google/bbr commit
`b16a9a18178b55103fbd2028967952db52d19f36`, Linux 6.8.0-rc3, BBR version 3.
`prepare_source.py` requires the exact upstream source SHA-256 before generating
the adapter. This is a reproducible pinned target, not a claim about the latest
upstream version or compatibility with an arbitrary host kernel.

A minimal arm64 kernel, stock module and experimental module run in QEMU
inside an unprivileged Docker container, with no project mounts or external
network. High-resolution timers are enabled for TCP pacing. Test traffic uses
MTU 1500, a 10 ms / 20 Mbit/s loopback netem qdisc, large socket buffers, and
three rotated repetitions of stock, disabled control and gain 1.0.

The qdisc shapes both data and ACK traffic on the same loopback interface.
QEMU scheduling, shared shaping, startup and drain time affect the measured
rate. These short runs check behavior and expose gross regressions; they do
not establish statistically powered performance equivalence or predict a
Starlink throughput gain. RTT values are sampled TCP smoothed RTT, not
individual packet RTT observations. Retransmissions are TCP_INFO segment
counts per flow, not the simulator's retransmission-rate metric.

## Simulator equivalence remains open

Follow-up: [action and testbed alignment](action_testbed_alignment.md) implements
the actuator and measurement fixes below as a new opt-in mode. The following
table records the fluid audit mode that produced the original Sydney results.

| Contract element | Current CRUISE-only fluid audit | Working adapter contract |
|---|---|---|
| Native DOWN gain | 0.90 | 232/256 = 0.90625 |
| Requested action values | Unquantized decimal values | 192, 230, 256, 282, 320 divided by 256 |
| Command application | Can change within CRUISE | Latches once at an eligible CRUISE entry |
| Phase progression | Simplified fixed-round proxy | Pinned upstream ACK-driven BBR transitions |
| Loss/recovery exclusion | Fluid-model guards and continuous indicator | TCP_CA_Open and no loss/CE in the current sample |
| Userspace outage | No command lease in the proxy | One-second lease; restore native on next ACK |
| Concurrency | Single simulated flow | Multiple adapter sockets disable overrides |

Changing the DOWN constant alone will not make the simulator equivalent. The
next experiment needs explicit requested-versus-applied action telemetry,
entry latching, quantization, cancellation and lease semantics, plus stock
fidelity checks against kernel replay. Preserve the old fluid mode and its
historical outputs as a separate experiment.

At the time of this audit, `qbbr/env/testbed_env.py` needed integration work: it writes
only after observing CRUISE and has no independent lease heartbeat. That
cannot guarantee a fresh request at the next entry. Its telemetry estimates
throughput from cwnd/RTT, uses placeholder inflight/queue features, and treats
the cumulative retransmission counter as consecutive high-loss observations.
Those quantities must be replaced with measured interval deltas and appropriate
state measurements before a trained policy is evaluated through that environment.
The guest audit uses its own receiver byte counter and TCP_INFO sampling;
it does not qualify this existing environment for field use.

After those checks pass, use a small Sydney downlink / second-city pilot with
fixed train/validation/test splits and frozen throughput, RTT and retransmission
acceptance criteria. Expand to the six-city matrix only if the pilot beats
stock and simple masked rules without violating the selected constraints.
A controlled Starlink trial follows held-out evaluation and deployment-kernel
validation. No new checkpoint is qualified by this adapter bring-up alone.
