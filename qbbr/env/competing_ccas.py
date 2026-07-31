from __future__ import annotations

from dataclasses import dataclass

_MIN_WINDOW_BYTES = 2.0 * 1500.0  # 2 MSS, a floor so a flow never fully collapses to zero

# Documented defaults, not values taken from any paper -- see module docstring.
_CUBIC_GROWTH_BPS = 2.0e6  # ~2 Mbps/s window growth pressure when uncongested
_CUBIC_DECREASE_BPS = 6.0e6  # backoff pressure when congested
_VEGAS_STEP_BPS = 1.0e6  # window adjustment rate once Vegas's delta crosses a threshold
_VEGAS_ALPHA_BYTES = 2.0 * 1500.0  # "under-utilizing" threshold, in bytes (~2 MSS of slack)
_VEGAS_BETA_BYTES = 4.0 * 1500.0  # "congested" threshold, in bytes (~4 MSS of slack)
_HYBLA_REFERENCE_RTT_S = 0.025  # Hybla's own convention: a 25ms reference RTT


@dataclass(frozen=True)
class CompetingCCAState:
    window_bytes: float


def offered_rate_bps(state: CompetingCCAState, shared_rtt_s: float) -> float:
    """Standard window/RTT throughput formula -- the same "demand" quantity for any window-based flow."""
    return state.window_bytes / shared_rtt_s


def _clamp_window(window_bytes: float) -> float:
    return max(window_bytes, _MIN_WINDOW_BYTES)


def update_cubic(
    state: CompetingCCAState, congestion_signal: float, dt_s: float
) -> CompetingCCAState:
    """Loss-reactive, moderate growth/backoff -- see module docstring for the simplification."""
    delta = _CUBIC_GROWTH_BPS * (1.0 - congestion_signal) - _CUBIC_DECREASE_BPS * congestion_signal
    return CompetingCCAState(window_bytes=_clamp_window(state.window_bytes + delta * dt_s))


def update_vegas(
    state: CompetingCCAState, rtt_base_s: float, shared_rtt_s: float, dt_s: float
) -> CompetingCCAState:
    """Delay-reactive: Vegas's classic Delta = window*(1 - RTT_base/RTT_shared) comparison."""
    delta_bytes = state.window_bytes * (1.0 - rtt_base_s / shared_rtt_s)
    if delta_bytes < _VEGAS_ALPHA_BYTES:
        window_bytes = state.window_bytes + _VEGAS_STEP_BPS * dt_s
    elif delta_bytes > _VEGAS_BETA_BYTES:
        window_bytes = state.window_bytes - _VEGAS_STEP_BPS * dt_s
    else:
        window_bytes = state.window_bytes
    return CompetingCCAState(window_bytes=_clamp_window(window_bytes))


def update_hybla(
    state: CompetingCCAState, congestion_signal: float, shared_rtt_s: float, dt_s: float
) -> CompetingCCAState:
    """Loss-reactive like Cubic, but growth scaled by rho = RTT/RTT_reference (Hybla's own feature)."""
    rho = shared_rtt_s / _HYBLA_REFERENCE_RTT_S
    delta = _CUBIC_GROWTH_BPS * rho * (1.0 - congestion_signal) - _CUBIC_DECREASE_BPS * congestion_signal
    return CompetingCCAState(window_bytes=_clamp_window(state.window_bytes + delta * dt_s))


_UPDATE_FNS = {
    "cubic": lambda state, ctx: update_cubic(state, ctx["congestion_signal"], ctx["dt_s"]),
    "vegas": lambda state, ctx: update_vegas(state, ctx["rtt_base_s"], ctx["shared_rtt_s"], ctx["dt_s"]),
    "hybla": lambda state, ctx: update_hybla(state, ctx["congestion_signal"], ctx["shared_rtt_s"], ctx["dt_s"]),
}


def update_competing_cca(name: str, state: CompetingCCAState, ctx: dict) -> CompetingCCAState:
    if name not in _UPDATE_FNS:
        raise ValueError(f"unknown competing CCA: {name!r}; expected one of {list(_UPDATE_FNS)}")
    return _UPDATE_FNS[name](state, ctx)
