import numpy as np

from qbbr.validation.queue import queue_model_diagnostics


def test_no_service_samples_is_unavailable_not_cs2_one() -> None:
    result = queue_model_diagnostics(None)
    assert result.service_cv_squared is None
    assert result.mg1_assumption_supported is None


def test_queue_diagnostics_estimate_trace_quantities() -> None:
    service = np.array([0.01, 0.02, 0.015, 0.012, 0.018])
    arrivals = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    result = queue_model_diagnostics(service, arrivals)
    assert result.sample_count == 5
    assert result.service_cv_squared is not None
    assert result.measured_arrival_rate_pps == 10.0
