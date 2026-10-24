# Change notes

## October 24, 2026

Eight small commits, in the recommended reading order:

1. **Setup:** added dependency lists and ignored generated results. Q-learning
   works without installing PyTorch.
2. **Commands:** connected the existing train/evaluate functions to
   `python -m thermorl`; the training and evaluation loops stay in `experiment.py`.
3. **Action masks:** both agents now reject empty or wrongly shaped masks rather
   than choosing an invalid action or failing with an unclear NumPy error.
4. **Q-learning targets:** convert integer 0/1 masks to boolean flags. Otherwise,
   NumPy reads them as table indices and can bootstrap from an infeasible action.
5. **DQN terminal targets:** explicitly zero the future value for an ended day.
   An empty next mask produces a maximum of negative infinity; multiplying it by
   zero previously produced NaN. Continuing transitions still need a valid action.
6. **Parameters:** reject invalid learning rates, discounts, and replay settings
   before training. For example, a replay buffer smaller than warmup never learns.
7. **Tuning:** exposed learning rate and gamma as CLI flags. Existing metadata
   records them, and the previous defaults remain the same.
8. **Preparation:** documented the observation/action/reward loop, checks,
   evaluation interpretation, and the limited follow-up schedule.

The target tests use fixed values so you can check the arithmetic by hand.
The original thermal model, reward weights, network architecture, and default
training schedule are unchanged. These are correctness and usability changes;
no improvement in trained-policy performance is claimed without a longer study.

Validation: all 14 unit tests passed; Python compilation and Git whitespace
checks passed. Q-learning trained for two episodes with custom tuning values;
DQN trained for three episodes, exercising replay updates. Validation evaluation
completed 40 Q-learning/baseline and 20 DQN/baseline daily rollouts across all five
scenarios, with finite metrics and zero power-cap violations. Three invalid CLI
configurations were rejected before creating an output directory.
