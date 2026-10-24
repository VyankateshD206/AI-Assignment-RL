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

## Run a small experiment

```powershell
python -m thermorl train --algo q_learning --episodes 10 --seed 0 --output outputs/q_demo
python -m thermorl evaluate --models outputs/q_demo --days 2 --output outputs/eval_demo
# For DQN, use --algo dqn and a different output folder when training.
```

Use a fresh output folder for each run. Ten episodes are a quick check, not
evidence of a well-trained policy. For command options, run
`python -m thermorl train --help` or `python -m thermorl evaluate --help`.

For a controlled tuning run, add `--learning-rate 0.1 --gamma 0.95` and use a new
output folder. Learning rate controls update size; gamma controls how much future
rewards matter. The chosen values are saved in `metadata.json`. Defaults are
unchanged: Q-learning uses 0.15, DQN uses 0.001, and both use gamma 0.99.

## Understand the RL loop

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

The tabular state bins are intentionally coarse, so different physical states
can share a table entry. This keeps the example simple but limits accuracy.

## Check and compare

```powershell
python -m unittest discover -s tests -v
```

Tests cover feasible actions, hand-calculated targets, terminal updates, and
invalid parameters. DQN tests skip when PyTorch is not installed.

Use `--split validation` for tuning. Reserve the default `--split test` for the
final comparison. Evaluation pairs controllers on identical day seeds; heatwave
days are excluded from training. Read `summary.csv` for energy, discomfort, and
power-cap violations, not just the training return. A short successful run proves
the pipeline works; it does not establish that RL beats the baselines.

The GUI server code exists, but this checkout does not include its `thermorl/web/`
assets. Use the command-line experiments above for now.

## Small-change preparation schedule

| Date (India time) | Scope |
| --- | --- |
| October 24, 2026 | Eight focused commits: setup, CLI, action masks, Q targets, DQN terminal targets, parameter checks, tuning flags, and this guide. |
| October 25, 2026, 18:00 | About five small, useful commits focused on explanations and focused checks; only fix agent behavior if a concrete bug is found. |
| October 26, 2026, 18:00 | About five small preparation and review commits; preserve the algorithms and finish the change notes. |

Later work should stay within about 30 changed executable lines per day, avoid
new algorithms or dependencies, and use fewer commits if there is no useful work
left. Each change should include a short reason in a nearby comment or the notes.
See [CHANGELOG.md](CHANGELOG.md) for the explanation of each change.
