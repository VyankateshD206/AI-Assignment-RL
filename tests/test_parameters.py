"""Reject settings that silently stop learning or produce invalid values."""
import unittest

from thermorl.q_learning import QLearning

try:
    from thermorl.dqn import DQN
except ImportError:
    DQN = None


class ParameterTests(unittest.TestCase):
    def test_q_learning_rejects_invalid_settings(self):
        for settings in ({"learning_rate": 0}, {"learning_rate": 1.1},
                         {"gamma": -0.1}, {"gamma": 1.1}, {"gamma": float("nan")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                QLearning(**settings)

    @unittest.skipIf(DQN is None, "Install requirements-dqn.txt to check DQN")
    def test_dqn_rejects_invalid_settings(self):
        for settings in ({"learning_rate": float("inf")}, {"gamma": float("nan")},
                         {"batch_size": 0}, {"batch_size": 1.5}, {"warmup": -1},
                         {"replay_capacity": 10}, {"target_interval": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                DQN(**settings)
