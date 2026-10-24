# Changelog

## October 24, 2026

- Added dependency lists and setup commands. PyTorch is optional for Q-learning.
- Added `python -m thermorl train` and `python -m thermorl evaluate` commands.
- Both agents reject empty or incorrectly shaped action masks.
- Q-learning converts integer masks to boolean flags before indexing Q-values.
- DQN sets terminal future values to zero to avoid NaN with an empty next mask.
- Added validation for learning rates, gamma, and replay settings.
- Added `--learning-rate` and `--gamma` training options with saved metadata.
- Documented observations, actions, rewards, and evaluation commands.

Checks: 14 unit tests passed. Short training runs completed for Q-learning
(two episodes) and DQN (three episodes). Evaluation completed 60 controller-day
rollouts across five scenarios with finite metrics and no power-cap violations.
Three invalid CLI configurations were rejected before creating output folders.
