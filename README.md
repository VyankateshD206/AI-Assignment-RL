# AI-Assignment-RL
Assignment 2: Real-Life Use Cases of Reinforcement Learning (RL)

ThermoRL simulates cooling two classrooms under a shared electricity limit.
It includes tabular Q-learning, a small DQN, and three rule-based controllers.
Weather and class schedules are synthetic, not measurements from a real building.

## Setup

Use Python 3.10 or newer. From the repository folder, create a local environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Only needed if you want to train or evaluate DQN:
python -m pip install -r requirements-dqn.txt
```

Generated models, CSV files, and plots belong in `outputs/`, which Git ignores.

## Training and evaluation

```powershell
python -m thermorl train --algo q_learning --episodes 10 --seed 0 --output outputs/q_demo
python -m thermorl evaluate --models outputs/q_demo --days 2 --output outputs/eval_demo
# For DQN, use --algo dqn and a different output folder when training.
```

Use a fresh output folder for each run. The example runs ten training episodes.
For command options, run
`python -m thermorl train --help` or `python -m thermorl evaluate --help`.

To change the training parameters, add `--learning-rate 0.1 --gamma 0.95` and use a new
output folder. Learning rate controls update size; gamma controls how much future
rewards matter. The chosen values are saved in `metadata.json`. Defaults are
unchanged: Q-learning uses 0.15, DQN uses 0.001, and both use gamma 0.99.

## RL model

1. `env.py` returns 17 normalized observations about temperatures, occupancy,
   timetable, weather, power limits, and previous AC settings.
2. Each room has four equipment modes: off, fan, AC, or AC plus fan. This gives
   16 joint actions. The action mask excludes combinations above the power cap.
3. The agent explores with probability epsilon; otherwise it chooses a feasible
   action with the highest estimated value. Evaluation uses epsilon zero.
4. The reward is negative energy, discomfort, fairness, and switching costs.
   A higher return therefore means a smaller weighted total cost.
5. Q-learning updates `Q(s,a)` toward `reward + gamma * max(Q(next_state))`,
   considering only feasible next actions. At the end of a day, the target is
   just the reward: values are not carried across the reset.
6. DQN approximates these values with a small neural network, learns from a
   replay buffer, and periodically copies weights to a target network.

Q-learning uses coarse state bins, so different physical states can share a
table entry.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests cover action masks, Q-value targets, terminal updates, and
invalid parameters. DQN tests skip when PyTorch is not installed.

Use `--split validation` for tuning. Reserve the default `--split test` for the
final comparison. Evaluation pairs controllers on identical day seeds; heatwave
days are excluded from training. `summary.csv` contains energy, discomfort, and
power-cap metrics. Compare these on held-out days to assess performance.

The GUI server code exists, but this checkout does not include its `thermorl/web/`
assets. Use the command-line experiments above.
