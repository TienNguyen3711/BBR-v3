from __future__ import annotations

import numpy as np
import torch

from qbbr.agents.quantum.qnn import build_qnn


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
