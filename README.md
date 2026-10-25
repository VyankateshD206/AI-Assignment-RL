# ThermoRL

**Occupancy- and weather-aware classroom cooling using reinforcement learning.**

ThermoRL is a Python simulation of two classrooms sharing a limited cooling power
budget. Q-learning and DQN control air conditioners and fans using temperature,
occupancy, announced class schedules, weather and available power. The project
studies the tradeoff between electricity use and comfort. No hardware or external
services are required.

## Requirements

- Python 3.10 or newer; the project was tested with Python 3.12.
- NumPy: numerical calculations and the Q-table.
- Gymnasium: reinforcement learning environment.
- PyTorch: DQN network, replay updates and model loading.
- Matplotlib: training and evaluation plots.
- pytest: automated tests.

Dependency constraints are defined in `pyproject.toml`. The GUI uses the Python
standard library and local HTML/CSS/JavaScript; Node.js and a frontend build are
not required. Training and inference run on CPU.

## Installation

Run from a terminal on Linux:

```bash
git clone https://github.com/VyankateshD206/AI-Assignment-RL.git
cd AI-Assignment-RL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

On Windows, create the environment with `py -3 -m venv .venv` and activate it in
PowerShell with `.venv\Scripts\Activate.ps1`. On macOS, omit the CPU-index command;
the editable installation installs PyTorch from the default package index.

Run all commands below from the repository root with the environment activated.

## Run the GUI

```bash
python gui.py
```

Open **http://127.0.0.1:8000**. If the port is occupied, use `python gui.py --port 8002`.
Stop the server with **Ctrl+C**.

The GUI presents the problem formulation, saved learning curves, policy comparison
and a step-by-step classroom replay. It loads the trained agents from `models/dqn`
and `models/q_learning`. Saved weights remain frozen during playback.

The initial demonstration uses the shared-power-limit scenario and day seed 42.
Change either under **Test different conditions**, then select **Run all policies**.
Every policy receives identical generated day inputs. **Inspect policy** changes
the displayed replay while retaining every policy in the comparison.

## Train agents

These commands reproduce the training settings and seeds of the bundled agents:

```bash
python train.py --algo dqn --episodes 400 --seed 2 --output outputs/dqn_reproduction
python train.py --algo q_learning --episodes 1500 --seed 0 --output outputs/q_reproduction
```

Training samples normal, restricted-power, stronger-solar and announced-schedule-change
days. Heatwave is excluded from training. Epsilon decreases from 1.0 to 0.05 over
80% of the training budget. Both agents use discount factor 0.99.

| Setting | Q-learning | DQN |
|---|---|---|
| Learning rate | 0.15 | 0.0003 |
| Representation | `compact_v1` discrete state | 21 derived comfort-aware inputs |
| Function approximator | Sparse Q-table | 21 → 64 ReLU → 64 ReLU → 16 |
| Replay / minibatch | Not applicable | 50,000 transitions / 64 |
| Replay warmup / target copy | Not applicable | 500 transitions / 250 optimizer updates |

Validation runs at episode 0, every 50 episodes and at completion, using two fixed
days per training scenario (seeds 1000000 and 1000001). The default selection rule
keeps the checkpoint with highest mean original validation return. The bundled
checkpoints were selected at DQN episode 350 and Q-learning episode 1350.

Each run saves `agent.pt` or `agent.json`, `last_agent.*`, `config.json`,
`metadata.json`, `training.csv`, `validation.csv` and `training_curve.png`.
Checkpoints support frozen inference; they do not contain complete optimizer,
replay and random-generator state for exact training resumption. Use a new output
directory for each run.

To demonstrate newly trained agents:

```bash
python gui.py --models outputs/dqn_reproduction outputs/q_reproduction
```

## Reproduce the evaluation

Evaluate the bundled agents and all three comparison baselines on 50 paired days:

```bash
python evaluate.py --models models/dqn models/q_learning --split test --days 10 --seed-offset 3000000 --output outputs/report_evaluation
```

This runs five scenarios with ten days each: normal, restricted power, stronger
solar exposure, announced timetable changes and heatwave. Every policy is evaluated
on the same scenario/seed pairs. The seed range reproduces an already inspected
comparison; it is not an untouched test set for future tuning.

Outputs include daily metrics, scenario summaries, reward diagnostics, a comparison
plot, first-day traces and evaluation metadata. Comfort summaries include pooled
event/time counts. To evaluate a single scenario, add `--scenarios restricted`.

Use separate training, validation and test data for new experiments. Training day
seeds are below 1000000; validation seeds lie in 1000000–1999999 and test seeds start
at 2000000. Distinct training seeds measure learning variability; day seeds change
the simulated conditions. A GUI demonstration day does not establish performance.

## Evaluation measures

- **Arrival discomfort:** percentage of occupied class starts outside the comfort
  band, measured before the first action using the previously active fan.
- **First-15-minute discomfort:** uncomfortable occupied room-time during the first
  15 clock minutes of each class.
- **Whole-day comfort:** comfortable occupied room-minutes divided by occupied
  room-minutes; percentages pool counts across the evaluated days.
- **Energy:** mean total daily equipment electricity, including empty-room precooling.

The comfort proxy is air temperature minus a configured 1.5°C offset while a fan is
active; the comfort band is 23–26°C. This is a simulation assumption, not a calibrated
physiological comfort model. The power cap is enforced through feasible-action masks.

Thermostat, timetable and rotation are comparison baselines only. They never supply
training labels or override learned actions. The solution methods are Q-learning and
standard DQN, trained through interaction and temporal-difference updates.

The current agents do not establish energy savings at comparable comfort versus the
strong timetable baseline. Subsequent DQN tuning did not satisfy all comfort/energy
replacement checks, so the previous trained DQN is retained.

## Verification

```bash
python -m pytest -q
python train.py --help
python evaluate.py --help
python gui.py --help
```

## Project structure

```text
thermorl/       Environment, RL agents, metrics, experiment runner and GUI
configs/        Simulator configuration
models/         Trained agents, metadata and learning histories
scripts/        Experiment and reporting utilities
tests/          Automated verification
train.py        Training entry point
evaluate.py     Paired evaluation entry point
gui.py          Local GUI entry point
report/         LaTeX report source, graphs and report data
```

**Team:** Vyankatesh Deshpande (B23CS1079), Rajas Kadu (B23CH1039).
