"""Policy-input ablations: fixed dimensions, unchanged physical safety state."""
from dataclasses import replace
import numpy as np
import torch


FEATURE_SETS = {
    "full": (), "telemetry": (3, 4, 5, 6), "telemetry_queue": (4, 5, 6),
    "no_queue": (3,), "no_handover": (4,), "no_failure": (5,), "no_phase": (6,),
}


def mask_features(state, variant):
    if variant not in FEATURE_SETS:
        raise ValueError(f"Unknown state ablation: {variant}")
    value = np.asarray(state, dtype=np.float32).copy()
    if value.shape[-1] != 7 or not np.isfinite(value).all():
        raise ValueError("Ablation requires finite seven-feature observations")
    value[..., list(FEATURE_SETS[variant])] = 0.5
    return value


class PolicyObservationEnv:
    """Safety sees raw features; actor/critic/replay see only ablated features.

    Agents used with this wrapper MUST have selector=None: the complete safety
    mask is computed here and stored in rollouts, including during updates.
    Masks intentionally reveal safety constraints, identically for every arm.
    """
    def __init__(self, env, selector, variant="full"):
        mask_features(np.zeros(7), variant)
        self.env, self.selector, self.variant = env, selector, variant
        self.raw_state = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed=None):
        self.raw_state = self.env.reset(seed=seed)
        return mask_features(self.raw_state, self.variant)

    def allowed_action_indices(self):
        if self.raw_state is None:
            raise RuntimeError("reset before querying action mask")
        return self.selector.admissible_actions(
            torch.zeros(5), torch.as_tensor(self.raw_state), self.env.allowed_action_indices())

    def step(self, action):
        if action not in self.allowed_action_indices():
            raise ValueError("Action violates physical safety mask")
        self.raw_state, reward, done, info = self.env.step(action)
        return mask_features(self.raw_state, self.variant), reward, done, {
            **info, "raw_state": np.asarray(self.raw_state).tolist(),
            "state_ablation": self.variant,
        }
