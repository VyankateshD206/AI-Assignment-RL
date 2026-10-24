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
