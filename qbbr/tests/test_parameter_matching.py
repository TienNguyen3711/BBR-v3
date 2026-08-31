from __future__ import annotations

import pytest

from qbbr.agents.matching import build_full_parameter_matched_classical, full_parameter_match


@pytest.mark.parametrize("n_layers", [2, 3])
def test_current_scenario_a_rq4_architecture_is_exactly_full_parameter_matched(n_layers):
    match = full_parameter_match(n_qubits=7, n_layers=n_layers, action_dims=(5,))
    assert match.exact
    classical, built_match = build_full_parameter_matched_classical(n_layers=n_layers)
    assert built_match == match
    assert classical.param_count() == match.quantum_params


def test_reuploading_does_not_change_the_full_parameter_budget():
    standard = full_parameter_match(n_qubits=7, n_layers=2, action_dims=(5,), reupload=False)
    reupload = full_parameter_match(n_qubits=7, n_layers=2, action_dims=(5,), reupload=True)
    assert standard.quantum_params == reupload.quantum_params
    assert standard.exact and reupload.exact


def test_unmatchable_state_dimension_cannot_be_used_for_rq4_without_architecture_revision():
    match = full_parameter_match(n_qubits=8, n_layers=2, action_dims=(5,))
    assert not match.exact
    with pytest.raises(ValueError, match="no exact full-agent parameter match"):
        build_full_parameter_matched_classical(n_qubits=8, n_layers=2, action_dims=(5,))
