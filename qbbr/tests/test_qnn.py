from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import pennylane as qml
import pytest
import torch

from qbbr.agents.quantum.qnn import build_qnn


@contextmanager
def _force_device(name: str):
    old = os.environ.get("QBBR_QUANTUM_DEVICE")
    os.environ["QBBR_QUANTUM_DEVICE"] = name
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("QBBR_QUANTUM_DEVICE", None)
        else:
            os.environ["QBBR_QUANTUM_DEVICE"] = old


def _lightning_available() -> bool:
    try:
        qml.device("lightning.qubit", wires=1)
        return True
    except Exception:
        return False


def test_param_count_matches_18l_for_l2_and_l3():
    assert sum(p.numel() for p in build_qnn(n_layers=2).parameters()) == 36
    assert sum(p.numel() for p in build_qnn(n_layers=3).parameters()) == 54


def test_output_shape_and_bounds():
    qnn = build_qnn(n_layers=2)
    out = qnn(torch.rand(6))
    assert out.shape == (6,)
    assert np.all(out.detach().numpy() >= -1.0 - 1e-6)
    assert np.all(out.detach().numpy() <= 1.0 + 1e-6)


def test_gradients_flow_to_weights():
    qnn = build_qnn(n_layers=2)
    out = qnn(torch.rand(6))
    out.sum().backward()
    assert qnn.weights.grad is not None
    assert torch.isfinite(qnn.weights.grad).all()


def test_two_instances_have_independent_weights():
    torch.manual_seed(0)
    qnn_a = build_qnn(n_layers=2)
    qnn_b = build_qnn(n_layers=2)
    assert not torch.allclose(qnn_a.weights, qnn_b.weights)


def test_reupload_variant_still_matches_param_count():
    qnn = build_qnn(n_layers=2, reupload=True)
    assert sum(p.numel() for p in qnn.parameters()) == 36


@pytest.mark.skipif(not _lightning_available(), reason="lightning.qubit not installed")
@pytest.mark.parametrize("reupload", [False, True])
def test_lightning_matches_default_qubit(reupload):
    """The fast backend (lightning.qubit + adjoint) must be numerically
    equivalent to the reference (default.qubit + backprop): same forward
    expectation values and same analytic gradients for identical weights."""
    torch.manual_seed(0)
    with _force_device("lightning.qubit"):
        fast = build_qnn(n_layers=2, reupload=reupload)
    with _force_device("default.qubit"):
        ref = build_qnn(n_layers=2, reupload=reupload)
    with torch.no_grad():
        ref.weights.copy_(fast.weights)

    x = torch.rand(6)
    out_fast = fast(x.clone())
    out_ref = ref(x.clone())
    torch.testing.assert_close(out_fast, out_ref, atol=1e-5, rtol=1e-4)

    out_fast.sum().backward()
    out_ref.sum().backward()
    torch.testing.assert_close(fast.weights.grad, ref.weights.grad, atol=1e-5, rtol=1e-3)
