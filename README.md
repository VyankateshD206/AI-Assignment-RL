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
