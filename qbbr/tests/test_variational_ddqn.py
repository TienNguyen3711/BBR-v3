import numpy as np

from qbbr.agents.quantum.variational_ddqn import VariationalDoubleDQN


def test_variational_double_dqn_respects_fixed_action_mask_and_updates() -> None:
    agent = VariationalDoubleDQN(observation_dim=2, action_count=3, n_layers=1, seed=0)
    state = np.array([0.1, 0.2], dtype=np.float32)
    assert agent.act(state, (1,), epsilon=0.0) == 1
    for _ in range(4):
        agent.observe(state, 1, 1.0, state, False, (1,))
    loss = agent.optimize(batch_size=4)
    assert loss is not None
    assert np.isfinite(loss)
