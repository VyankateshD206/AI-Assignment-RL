"""Both agents must respect the classroom power-limit mask."""
import unittest

import numpy as np

from thermorl.q_learning import QLearning

try:
    from thermorl.dqn import DQN
except ImportError:
    DQN = None


class QLearningActionTests(unittest.TestCase):
    agent_class = QLearning

    def test_only_feasible_action_is_selected(self):
        agent = self.agent_class(seed=7)
        mask = [False] * 16
        mask[5] = True
        for epsilon in (0.0, 1.0):
            self.assertEqual(agent.act(np.zeros(17), mask, epsilon), 5)

    def test_empty_mask_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "feasible"):
            self.agent_class().act(np.zeros(17), np.zeros(16, dtype=bool))

    def test_wrong_mask_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "16"):
            self.agent_class().act(np.zeros(17), [True])


@unittest.skipIf(DQN is None, "Install requirements-dqn.txt to check DQN")
class DQNActionTests(QLearningActionTests):
    agent_class = DQN


if __name__ == "__main__":
    unittest.main()
