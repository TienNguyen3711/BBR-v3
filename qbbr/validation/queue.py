"""Diagnostics for, rather than an assumption of, M/G/1 service behaviour."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

try:  # scipy is optional for the core package, but preferred for validation.
    from scipy.stats import kstest
except ImportError:  # pragma: no cover - exercised on minimal installations
    kstest = None


@dataclass(frozen=True)
class QueueModelDiagnostics:
    sample_count: int
    mean_service_s: float | None
    service_cv_squared: float | None
    exponential_ks_statistic: float | None
    exponential_ks_pvalue: float | None
    measured_arrival_rate_pps: float | None
    mg1_assumption_supported: bool | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def queue_model_diagnostics(
    service_times_s: Sequence[float] | np.ndarray | None,
    arrival_times_s: Sequence[float] | np.ndarray | None = None,
    alpha: float = 0.05,
) -> QueueModelDiagnostics:
    """Estimate c²_s and test exponential service time against packet traces.

    A non-significant KS result is weak evidence only; it does not prove an
    M/G/1 approximation is correct.  Empty/non-positive samples are explicitly
    reported as unavailable rather than coerced into c²_s=1.
    """

    if service_times_s is None or len(service_times_s) == 0:
        return QueueModelDiagnostics(0, None, None, None, None, None, None)
    service = np.asarray(service_times_s, dtype=float)
    service = service[np.isfinite(service) & (service > 0)]
    if len(service) == 0:
        return QueueModelDiagnostics(0, None, None, None, None, None, None)
    mean = float(np.mean(service))
    cv_squared = float(np.var(service, ddof=0) / (mean * mean))
    ks_statistic: float | None = None
    pvalue: float | None = None
    supported: bool | None = None
    if kstest is not None:
        result = kstest(service, "expon", args=(0, mean))
        ks_statistic = float(result.statistic)
        pvalue = float(result.pvalue)
        supported = pvalue >= alpha
    rate: float | None = None
    if arrival_times_s is not None and len(arrival_times_s) >= 2:
        arrivals = np.asarray(arrival_times_s, dtype=float)
        arrivals = arrivals[np.isfinite(arrivals)]
        span = float(np.max(arrivals) - np.min(arrivals)) if len(arrivals) >= 2 else 0.0
        if span > 0:
            rate = float((len(arrivals) - 1) / span)
    return QueueModelDiagnostics(
        sample_count=len(service),
        mean_service_s=mean,
        service_cv_squared=cv_squared,
        exponential_ks_statistic=ks_statistic,
        exponential_ks_pvalue=pvalue,
        measured_arrival_rate_pps=rate,
        mg1_assumption_supported=supported,
    )
