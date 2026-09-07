import numpy as np

from qbbr.agents.quantum.recurrent_prioritized_ddqn import (
    RecurrentPrioritizedVariationalDoubleDQN,
)


def test_recurrent_prioritized_double_dqn_respects_mask_and_updates() -> None:
    agent = RecurrentPrioritizedVariationalDoubleDQN(
        observation_dim=2, action_count=3, quantum_dim=2, n_layers=1, seed=0
    )
    history = np.array([[0.1, 0.2], [0.2, 0.3]], dtype=np.float32)
    assert agent.act(history, (1,), epsilon=0.0) == 1
    for _ in range(4):
        agent.observe(history, 1, 1.0, history, False, (1,))
    loss = agent.optimize(batch_size=4)
    assert loss is not None
    assert np.isfinite(loss)
    assert all(priority > 0 for priority in agent.replay.priorities)
