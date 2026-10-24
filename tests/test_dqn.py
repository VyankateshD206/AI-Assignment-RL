"""Small deterministic checks of DQN targets, without long training runs."""
import unittest

import numpy as np

try:
    import torch
    from thermorl.dqn import DQN
except ImportError:
    DQN = None


@unittest.skipIf(DQN is None, "Install requirements-dqn.txt to check DQN")
class DQNUpdateTests(unittest.TestCase):
    def setUp(self):
        self.agent = DQN(batch_size=1, warmup=1, replay_capacity=2, gamma=0.5)
        self.state = np.zeros(17, dtype=np.float32)
        with torch.no_grad():
            for model in (self.agent.online, self.agent.target):
                for parameter in model.parameters():
                    parameter.zero_()
            self.agent.target[-1].bias.fill_(100.0)
            self.agent.target[-1].bias[3] = 4.0

    def test_target_uses_only_feasible_future_values(self):
        mask = [False] * 16
        mask[3] = True
        loss = self.agent.learn(self.state, 0, 2.0, self.state, False, mask)
        # Target = 2 + 0.5*4 = 4; Huber loss against zero is 4 - 0.5.
        self.assertAlmostEqual(loss, 3.5)

    def test_terminal_empty_mask_keeps_loss_and_weights_finite(self):
        loss = self.agent.learn(self.state, 0, 2.0, self.state, True, [False] * 16)
        self.assertAlmostEqual(loss, 1.5)
        self.assertTrue(all(torch.isfinite(p).all() for p in self.agent.online.parameters()))

    def test_nonterminal_empty_mask_is_rejected_before_replay(self):
        with self.assertRaisesRegex(ValueError, "feasible"):
            self.agent.learn(self.state, 0, 2.0, self.state, False, [False] * 16)
        self.assertEqual(self.agent.size, 0)
