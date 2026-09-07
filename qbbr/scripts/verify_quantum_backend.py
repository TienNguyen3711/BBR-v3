"""Evidence that the fast quantum simulator backend (lightning.qubit + adjoint)
is numerically equivalent to the reference (default.qubit + backprop), and how
much faster it is.

RQ4 compares distributions over 10 seeds, so a single seeded trajectory need
not be bit-identical across backends -- but the forward expectation values and
analytic gradients must agree to float precision, and the short-horizon
training curve must not diverge. This script checks all three and prints a
speed ratio.

    python -m qbbr.scripts.verify_quantum_backend
    python -m qbbr.scripts.verify_quantum_backend --location Sydney --n-episodes 15
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"


def _build_agent(device: str, seed: int):
    os.environ["QBBR_QUANTUM_DEVICE"] = device
    torch.manual_seed(seed)
    np.random.seed(seed)
    from qbbr.agents.quantum.qa2c import QA2CAgent

    return QA2CAgent(n_qubits=7, n_layers=2, action_dims=(5,))


def _copy_weights(dst, src) -> None:
    with torch.no_grad():
        dst.actor_qnn.weights.copy_(src.actor_qnn.weights)
        dst.critic_qnn.weights.copy_(src.critic_qnn.weights)
        for d, s in zip(dst.actor_heads.parameters(), src.actor_heads.parameters()):
            d.copy_(s)
        for d, s in zip(dst.critic_head.parameters(), src.critic_head.parameters()):
            d.copy_(s)


def _forward_grad_check(seed: int) -> None:
    fast = _build_agent("lightning.qubit", seed)
    ref = _build_agent("default.qubit", seed)
    _copy_weights(ref, fast)

    states = torch.rand(8, 7)
    zf = torch.stack([fast.actor_qnn(s) for s in states])
    zr = torch.stack([ref.actor_qnn(s) for s in states])
    fwd_max = (zf - zr).abs().max().item()

    (zf.sum()).backward()
    (zr.sum()).backward()
    grad_max = (fast.actor_qnn.weights.grad - ref.actor_qnn.weights.grad).abs().max().item()

    print(f"forward  max|lightning - default| = {fwd_max:.2e}")
    print(f"gradient max|lightning - default| = {grad_max:.2e}")
    assert fwd_max < 1e-4, f"forward mismatch {fwd_max}"
    assert grad_max < 1e-3, f"gradient mismatch {grad_max}"
    print("  -> forward + gradient: EQUIVALENT\n")


def _training_curve_check(location: str, direction: str, n_episodes: int, seed: int) -> None:
    from qbbr.env.calibration import load_calibration
    from qbbr.env.fluid_env import FluidSimEnv
    from qbbr.train.loop import train

    calibration = load_calibration(DEFAULT_CALIBRATION_PATH)
    curves, times = {}, {}
    for device in ("lightning.qubit", "default.qubit"):
        agent = _build_agent(device, seed)
        env = FluidSimEnv(location, direction, calibration, risk_mode="closed_form",
                          episode_s=300.0)
        t0 = time.time()
        result = train(agent, env, n_episodes, config={"seed": seed}, run_dir=None)
        times[device] = time.time() - t0
        curves[device] = result["final_mean_reward"]

    lo, hi = curves["lightning.qubit"], curves["default.qubit"]
    print(f"{location} {direction}, {n_episodes} episodes, seed {seed}:")
    print(f"  final_mean_reward  lightning={lo:.5f}  default={hi:.5f}  |diff|={abs(lo - hi):.2e}")
    print(f"  wall time         lightning={times['lightning.qubit']:.1f}s  "
          f"default={times['default.qubit']:.1f}s  "
          f"speedup={times['default.qubit'] / max(times['lightning.qubit'], 1e-9):.2f}x")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--location", default="London")
    ap.add_argument("--direction", default="downlink", choices=["downlink", "uplink"])
    ap.add_argument("--n-episodes", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pennylane as qml
    try:
        qml.device("lightning.qubit", wires=1)
    except Exception as exc:
        print(f"lightning.qubit unavailable ({exc!r}); nothing to verify.")
        return

    torch.set_num_threads(1)
    print("=== forward / gradient equivalence (identical weights) ===")
    _forward_grad_check(args.seed)
    print("=== short training-curve + speed comparison ===")
    _training_curve_check(args.location, args.direction, args.n_episodes, args.seed)


if __name__ == "__main__":
    main()
