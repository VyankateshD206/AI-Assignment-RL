"""Q-learning target checks."""
import unittest

import numpy as np

from thermorl.q_learning import QLearning, discretize


class QLearningUpdateTests(unittest.TestCase):
    def setUp(self):
        self.agent = QLearning(learning_rate=1.0, gamma=0.5)
        self.state = np.zeros(17)
        self.next_state = self.state.copy()
        self.next_state[0] = 0.8
        future = np.full(16, 100.0)
        future[3] = 4.0
        self.agent.table[discretize(self.next_state)] = future

    def test_target_uses_only_feasible_future_values(self):
        # Integer masks use 0/1 as flags, not as indices into the Q table.
        mask = [0] * 16
        mask[3] = 1
        self.agent.learn(self.state, 0, 2.0, self.next_state, False, mask)
        self.assertEqual(self.agent.table[discretize(self.state)][0], 4.0)

    def test_terminal_target_is_only_the_reward(self):
        self.agent.learn(self.state, 0, 2.0, self.next_state, True, [0] * 16)
        self.assertEqual(self.agent.table[discretize(self.state)][0], 2.0)

    def test_nonterminal_empty_mask_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "feasible"):
            self.agent.learn(self.state, 0, 2.0, self.next_state, False, [0] * 16)
