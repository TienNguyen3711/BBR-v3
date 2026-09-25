# Action and testbed alignment — 17 September 2026

The new opt-in `kernel_action_semantics` fluid mode uses the kernel adapter's
five fixed-point gains (192, 230, 256, 282, 320 / 256), native DOWN 232/256,
CRUISE-entry latching, explicit clear and one-second command expiry. Pending
requests can be submitted outside CRUISE. Changing a request in CRUISE does
not change its current latched gain; after cancellation it can latch again
only at a subsequent eligible entry. A loss/CE exclusion cancels the latch.

Enable all four values from
[`kernel_action_alignment.yaml`](../qbbr/configs/kernel_action_alignment.yaml)
in a new experiment's `dynamics_overrides`. Historical modes/configurations
keep their previous numerical behavior. The simulator reports requested,
latched and applied fixed-point units plus time spent at each applied gain.
A reconfiguration freeze explicitly clears the command rather than requesting
1.0 and waiting for another entry. Direct substep callers can use
`refresh_command=False` to simulate an interrupted command stream and
`action_eligible=False` to exercise recovery exclusion.

This aligns the **actuator contract**, not the entire BBR implementation.
Fluid phase durations remain simplified. Loss is still a calibrated demand
proxy: fractional retransmission demand accumulates into packet loss signals
for the next substep, and the queue-based CE proxy supplies exclusion signals.
Neither is an ACK-level TCP recovery model. Kernel replay fidelity remains a
separate gate before interpreting a pilot as deployment evidence.

## Testbed integration

`TestbedEnv` now measures sender acknowledged-byte deltas over monotonic elapsed
time. It no longer infers throughput from cwnd/RTT. Interval retransmission
counts drive the consecutive-loss guard; an old retransmission no longer
keeps triggering it forever. TCP delivery-rate samples and minimum RTT supply
BDP, with actual outstanding-packet counts supplying an inflight estimate.
Queue occupancy is explicitly an **inflight-minus-BDP estimate**, not a direct
bottleneck queue measurement. Acknowledged goodput is not a receiver application
byte counter. Returned telemetry identifies these measurement sources.

Both the testbed and `LinuxTcpInfoSource` now use one length-checked Linux
TCP_INFO decoder. The old field reader incorrectly treated several extended
64-bit counters as fields in the 104-byte u32 prefix. Missing extended fields
now fail explicitly instead of producing fabricated throughput.

A 100 ms background heartbeat refreshes a request independently of policy
computation or the current BBR phase. Each policy submission has a watchdog
validity of the next decision interval plus one second. If no new submission
arrives, the thread writes clear; if the process/thread dies, the kernel's
one-second lease expires on ACK processing. Heartbeat I/O failures surface to
the caller. Episode completion, exceptions, reset and `close()` stop the
thread and clear the request. Use the environment as a context manager or call
`close()` when abandoning an episode; it does not own or close the TCP socket.
The socket must already use the matching `bbr_qrl` adapter on the sender.

The existing testbed observation remains six-dimensional; the fluid/field
seven-feature policies need an explicit compatible observation interface.
These measurement fixes alone do not qualify a saved checkpoint for deployment.

## Verification

The 176-test regression run passed, including the historical fluid/Sydney,
transport, field-state, action and control suites. The tests compile and execute the actual C controller transition functions,
with only kernel locking/time primitives stubbed, and compare their state
transitions against the Python actuator. They exercise gain quantization,
mid-CRUISE updates, exclusion, clear and expiry. Additional tests cover queued
requests outside CRUISE, interval counters, 64-bit decoding, counter wrap,
heartbeat refresh/watchdog/failure/cleanup and environment shutdown.

An isolated Linux loopback test sent 1,000,000 bytes: the shared decoder's
acknowledged-byte delta matched the receiver's 1,000,000 bytes, with zero
retransmissions. This validates the decoder on Linux; it is not a Starlink
benchmark or an end-to-end policy deployment test.
