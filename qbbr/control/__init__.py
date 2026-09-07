"""Contracts and safety checks for the active native-action RL programme.

The historical numeric action configurations in :mod:`qbbr.action` supply the
frozen BBR choice set for the new study; QRL may select, but not expand, it.
"""

from .contracts import (
    ActionSpec,
    MDPContract,
    ObservationSpec,
    RewardSpec,
    RunManifest,
    SafetySpec,
)
from .space import NativeActionSpace, NativeObservationNormalizer, encode_observation
from .native_action_selector import NativeActionSelector

__all__ = [
    "ActionSpec",
    "MDPContract",
    "ObservationSpec",
    "RewardSpec",
    "RunManifest",
    "SafetySpec",
    "NativeActionSpace",
    "NativeObservationNormalizer",
    "encode_observation",
    "NativeActionSelector",
]
