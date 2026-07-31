from __future__ import annotations

import numpy as np

from qbbr.train.buffer import RolloutBuffer


def test_add_appends_to_all_lists():
    buf = RolloutBuffer()
    buf.add(np.zeros(6), 2, -0.5, 1.0, 0.3)
    buf.add(np.ones(6), 1, -0.6, 0.5, 0.4)
    assert len(buf) == 2
    assert buf.actions == [2, 1]
    assert buf.rewards == [1.0, 0.5]
    assert buf.log_probs == [-0.5, -0.6]
    assert buf.values == [0.3, 0.4]


def test_clear_empties_everything():
    buf = RolloutBuffer()
    buf.add(np.zeros(6), 0, -0.1, 0.0, 0.0)
    buf.clear()
    assert len(buf) == 0
    assert buf.states == []
    assert buf.actions == []
