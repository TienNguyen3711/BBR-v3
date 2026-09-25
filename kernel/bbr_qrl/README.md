# Isolated BBR-v3 adapter bring-up

The adapter now builds as **one composite module**, `bbr_qrl.ko`, registering
TCP congestion control `bbr_qrl`. The original `bbr` implementation stays
separate. Do not load this laboratory module into a shared production kernel.
The disposable QEMU guest below supplies the exact matching kernel.

On 17 September 2026, the kernel boot and **19-flow runtime audit passed**,
alongside **63 host regression tests**. See the
[measured results and remaining gaps](../../docs/kernel_adapter_audit.md).
This validates laboratory behavior; performance equivalence remains open.

The previous annotated patch's weak functions could not be replaced by loading
a companion module. They have been removed. `prepare_source.py` now emits a
real, hash-checked patch and a separately named copy of the pinned BBR source;
the control object is linked directly with that copy.

## Source and build

Pinned repository: `https://github.com/google/bbr`

Commit: `b16a9a18178b55103fbd2028967952db52d19f36` (Linux 6.8.0-rc3).

`net/ipv4/tcp_bbr.c` SHA-256:
`6d88372b861cfbabb560dc65c4341933ebfd89f33a65df67ec9b1f0dd667b062`.

The generator refuses any other file content. Stock congestion-control logic
is preserved; the laboratory copy omits BPF kfunc registration and adds control
initialization, socket lifecycle tracking and a pacing-only hook at the end of
each ACK's BBR processing, including the fast path. No stock source file is
overwritten by the build. The full upstream kernel remains inside the build
image; it is not vendored into this repository.

From the Codebase directory (native arm64 Linux/Docker build):

```sh
docker build -t qbbr-kernel-audit:20260917 -f kernel/bbr_qrl/Dockerfile.audit kernel/bbr_qrl
docker build -t qbbr-kernel-verify:20260917 -f kernel/bbr_qrl/Dockerfile.verify kernel/bbr_qrl
docker run --rm --network none --read-only --tmpfs /tmp:rw,size=64m \
  --cap-drop ALL --security-opt no-new-privileges --cpus 2 --memory 1g \
  qbbr-kernel-verify:20260917 > guest.log 2>&1
```

Only the files allowlisted in `.dockerignore` enter the build context. There
are no project bind mounts, privileged containers, external guest networking,
or kernel changes to the existing Docker VM. QEMU boots the built kernel and
loads both stock and experimental modules **inside the guest**.

A successful QEMU exit alone is not a test pass. The serial log must contain
`QBBR_KERNEL_AUDIT_PASS` and the JSON result. `extract_report.py` validates and
extracts that record. A guest assertion failure produces a failure marker.

```sh
python3 kernel/bbr_qrl/extract_report.py guest.log --out reports/kernel_adapter_audit/report.json
```

The verification build cleans inherited object files before compiling, so
COPY-preserved timestamps cannot silently retain an older control object.

## Executable interface

The module creates `/sys/kernel/debug/tcp_bbr_qrl/`:

- `pacing_gain_override` (root read/write): request exactly `0.75`, `0.90`,
  `1.00`, `1.10`, or `1.25`. `-1` disables the override. Decimal strings may
  have up to six fractional digits; undeclared values and malformed input
  fail without changing the previous request.
- `cruise_active` (root read): whether the last observed ACK was eligible.
- `status` (root read): requested/latched/applied/native gains in units of
  1/256, active socket count, phase and cumulative ACK counters.

Commands latch **once on actual CRUISE entry**, provided the socket is not in
congestion recovery and the current sample has no loss/CE signal. A command
change during CRUISE waits for the next entry. A loss/CE indication cancels
the current override until a later CRUISE entry; it does not alter BBR's
recovery logic. Disabled requests, expiry, and ineligible phases use the
native gain. Expiry and disabling take effect on the **next ACK**, not via an
asynchronous packet-transmission interrupt.

A request has a **one-second lease**. Userspace must refresh it, for example
every 100 ms, even if the policy choice is unchanged. A dead agent therefore
cannot leave a permanent override. `qbbr.action.bbr_hook.set_pacing_gain`
validates the same five requests but does not run a background heartbeat.

Requested decimal gains quantize to 192, 230, 256, 282 and 320. Readback
prints the fixed-point result to six decimals; it does not echo the original
request. Do not feed the quantized readback in as a new action.

Single-flow scope is enforced: when multiple `bbr_qrl` sockets coexist,
overrides are disabled on ACK processing. Releasing the owner clears its
request; no socket pointer is dereferenced by the controller. Stock `bbr`
sockets are not registered with this control interface and are unaffected.
The global status is diagnostic and is not a per-socket multi-flow telemetry
API. It can retain its last ACK observation during idle periods.

## Validation boundaries

The C parser is compiled and exercised directly by
`qbbr/tests/test_bbr_qrl_kernel_contract.py`; it is no longer a Python mirror.
The QEMU harness uses shaped loopback TCP to check registration, native phase
gains, exact command application, CRUISE-entry latching, clear/expiry, loss,
and fail-closed concurrent flows. It compares three short runs each of stock,
disabled control and gain 1.0, with rotated order. These are smoke comparisons,
not powered performance-equivalence tests or Starlink measurements.

The pinned source's DOWN gain is **232/256**, not the fluid audit's 0.90.
That audit also applies requests within CRUISE and uses simplified timers;
the adapter latches on entry and retains upstream phase transitions. Therefore
**the existing fluid simulator is not yet action-equivalent**. Do not promote
its old checkpoints or begin a six-city retraining under an equivalence claim.

The out-of-tree module, initramfs and kernel are for the matching QEMU test
kernel only. They are not deployable artifacts for an arbitrary host kernel.
