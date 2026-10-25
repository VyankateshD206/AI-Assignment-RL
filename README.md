# ThermoRL

**Fair, timetable-aware classroom cooling under shared power limits.**

A Python-only reinforcement learning mini-project. Two simulated classrooms share
an electrical cooling allowance. An agent controls AC and fans using temperatures,
occupancy, announced class schedules, weather, and power availability. No hardware,
sensors, IoT services, external APIs, or live building control are involved.

The research question is whether RL can improve class-start comfort and worst-room
comfort under restricted power without excessive electricity use. Savings and
superiority over baselines are hypotheses, not established results.

## Setup

Python 3.10+ is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m pytest -q
```

The first installation command selects CPU PyTorch. All algorithms run on CPU.
The editable install obtains the other dependencies from `pyproject.toml`.

## Interactive GUI

From the repository root, start the local dashboard:

```bash
source .venv/bin/activate
python gui.py
```

Open **http://127.0.0.1:8000** in a browser. The minimal teaching interface has three sections:

1. **Problem & model:** state, actions, thermal dynamics, reward and loaded parameters.
2. **Decision walkthrough:** a synchronized two-classroom schematic with occupied
   desks, AC airflow, fan rotation, temperature changes, comfort status and shared
   power allocation. Below it are before-action observations, feasible actions and
   actual Q-values, heat balance and reward.
3. **Training & results:** Q-learning/DQN update rules, actual saved training history,
   and a same-day controller comparison.

Use Next decision, Previous, Reset or the timeline to illustrate the agent/environment
loop. Play animates the same frozen rollout. Saved logs show training; the UI never
pretends to train a model during playback. CSV export includes the full generated day.

The room view renders the same Python trajectory: each filled dot represents one
student, equipment indicators show the selected action, and the large readings are
the resulting air temperatures. Comfort badges use the fan-adjusted proxy only for
occupied rooms. Blue and violet consistently identify the two rooms in the diagram,
power bar and chart. Ubuntu Sans and Ubuntu Sans Mono are bundled locally for offline
use; their licence is in `thermorl/web/FONT-LICENSE.txt`.

![ThermoRL classroom visualization](docs/gui-rooms-preview.jpg)

The dashboard automatically discovers Q-learning and DQN checkpoints under
`outputs/*/` when the server starts. With no checkpoints, the three rule-based
controllers still work. Restart after training new runs, or select models explicitly:

```bash
python gui.py --port 8001 --models outputs/q_seed0 outputs/dqn_seed0
```

Use `--models` with no directories for a baseline-only demo. Use `--config` to select
a physical environment configuration; compared checkpoints must match it.

All physics, decisions, rewards and metrics are computed by the existing Python
environment and controllers. The HTML/CSS/JavaScript browser interface only presents
the Python-generated rollout and plays it back. It does not duplicate the thermal
model or train policies during playback. Each displayed action applies to the
interval identified in the state/transition panels. Both before- and after-action
temperatures are shown. Summary metrics accumulate only up to the selected time. CSV export and the comparison
view cover the whole day.

This is a local application, bound to loopback only. No hosting account, external
API, JavaScript build tool, hardware, or extra Python dependency is required.
Exploratory GUI comparisons are not substitutes for held-out, multi-seed evaluation.
Stop the server with Ctrl+C.

## Train and evaluate

```bash
python train.py --algo q_learning --episodes 300 --seed 0 --output outputs/q_seed0
python train.py --algo dqn --episodes 300 --seed 0 --output outputs/dqn_seed0
python evaluate.py --models outputs/q_seed0 outputs/dqn_seed0 --split validation --days 10 --output outputs/validation
```

Each training command saves an inference checkpoint, environment configuration,
run metadata and dependency versions, per-episode CSV metrics, and a training plot.
Use a fresh output directory for each run. Checkpoints support evaluation; optimizer,
replay-buffer, and random-generator states for resuming training are not saved.

`agent.json` or `agent.pt` stores the selected validation checkpoint; `last_agent.*`
stores the final episode's weights. The default selection uses mean validation
return. `metadata.json` records the selected episode and training parameters.

After selecting hyperparameters using **validation** days, freeze the setup and run:

```bash
python evaluate.py --models outputs/q_seed0 outputs/dqn_seed0 --split test --days 20 --output outputs/test
```

Evaluation also runs three baselines. Outputs include `daily_metrics.csv`,
`summary.csv`, an energy-versus-comfort plot, and per-controller temperature,
occupancy, and power traces for the first evaluation day of the first scenario.
Use `--scenarios restricted` to focus on the shared-power case. Baselines can also
be evaluated without models:

```bash
python evaluate.py --scenarios restricted --days 5 --output outputs/baselines
```

The examples use 300 episodes. Without `--episodes`, the CLI uses 1,500 for
Q-learning and 400 for DQN. Inspect validation
performance before choosing larger budgets or changing reward weights. Exploratory
training returns are not directly comparable to greedy held-out evaluation returns.

## Original algorithms and baselines

- **Q-learning:** sparse NumPy Q-table, epsilon-greedy exploration, `compact_v1`
  state bins by default, and feasible-action Bellman targets. `--q-state legacy`
  selects the original discretization.
- **DQN:** the CLI uses a PyTorch 21–64–64–16 MLP with `comfort_v1` features,
  replay buffer (50,000), target network
  copied every 250 updates, batch size 64, 500-transition warmup, Huber loss,
  gradient clipping, Adam learning rate 0.0003, and discount 0.99.
- Q-learning uses learning rate 0.15 and discount 0.99. Both agents decay epsilon
  linearly from 1.0 to 0.05 over the first 80% of training episodes.
- **Thermostat:** occupancy-aware hysteresis; power conflicts favor current
  occupancy-weighted cooling need.
- **Timetable:** thermostat with a fixed 30-minute pre-cooling window.
- **Rotation:** occupied rooms alternate priority every 15 minutes when power
  is limited.

`--dqn-features legacy` uses the original 17 inputs. DQN training rewards are
divided by `--reward-scale` (default 10); reported physical returns are unscaled.

All controllers receive the same observations and must select feasible joint actions.
The baselines are comparators; the project's learned solutions are Q-learning and DQN.
SARSA and PPO are outside the implemented scope.

## Environment and formulation

Each step represents 5 minutes; one episode is a 24-hour finite task (288 steps).
Each room has four modes: off, fan, AC, AC+fan. Joint action `4 * room0_mode +
room1_mode` produces 16 actions. AC consumes 3 kW and removes 12 kW of heat
(assumed constant COP of 4). The room fan group consumes 0.12 kW. Normal capacity
is 6.5 kW; restricted capacity is 3.3 kW from 11:00 to 16:00. Thus restricted
operation can support one AC plus both fan groups. All-off is always feasible.

An action mask rejects infeasible combinations. It is applied during exploration,
greedy selection, and next-state value estimation. This establishes electrical
feasibility, not a guarantee that every class remains comfortable.

Each room follows a first-order energy balance, using kW, hours, and kWh/°C:

```text
T_next = T + dt / C * [UA*(T_out-T) + solar_gain*irradiance
                      + occupants*person_heat + fan_electric - AC_cooling]
```

Room 0 receives twice the solar gain of room 1 and has a smaller thermal capacity.
There is no inter-room heat transfer. Fan motor power adds heat; fans improve only
an explicit comfort proxy: `T_proxy = T_air - 1.5°C` while running. This offset,
the 23–26°C comfort band, and the building parameters are assumed, uncalibrated
simulation choices. The proxy is not PMV, a medical metric, or validated comfort.
Humidity, ventilation, radiant temperature, compressor minimum run times, and
variable AC efficiency are not modeled. Switching is discouraged by a penalty.

The 17-dimensional observation contains:

| Indices | Quantity | Encoding |
|---|---|---|
| 0–1 | Indoor air temperatures | / 50°C |
| 2–3 | Current room occupancies | / 60 |
| 4–5 | Time until next class | capped at 240 min, / 240 |
| 6–7 | Occupancy of next announced class | / 60 |
| 8 | Outdoor temperature | / 50°C |
| 9 | Solar irradiance | / 1000 W/m² |
| 10–11 | Time of day | sine, cosine |
| 12 | Current power cap | / 8 kW |
| 13–14 | Previous AC states | 0 or 1 |
| 15 | Time until next cap change | capped at 240 min, / 240 |
| 16 | Next announced cap | / 8 kW |

Observations are normalized by scale, not clipped to [0,1]. Schedules and power
restrictions are announced at reset. Cancellation experiments are **announced**
cancellations, not a test of predicting unknown events. Future schedules are
summarized, so this is an approximate MDP observation rather than a claim of exact
Markov sufficiency. Q-learning introduces further aggregation; its large possible
state space can remain sparsely visited. Its training CSV reports visited states.

For comfort violation `v_i = max(23 - T_proxy_i, 0, T_proxy_i - 26)`:

```text
energy       = equipment_power * dt
 discomfort  = dt * sum((occupants_i / 60) * v_i**2)
 worst_room  = dt * max(v_i**2 over currently occupied rooms, default 0)
 switches    = number of AC on/off changes
reward       = -(0.3*energy + 2.0*discomfort + 1.0*worst_room + 0.02*switches)
```

The worst-room term encourages fairness but does not guarantee long-term equality.
All terms are computed for the transition's current occupants and resulting room
temperatures. At day-end, Gymnasium returns `truncated=True`; learning deliberately
uses zero terminal bootstrap because the experiment defines an independent finite
daily task. No future return is carried across daily resets.

Parameters and reward weights are editable through a JSON configuration:

```bash
python train.py --algo dqn --config configs/default.json --episodes 300 --seed 1 --output outputs/dqn_custom
```

## Scenarios, metrics, and experimental discipline

Synthetic daily profiles include weather variability, timetable jitter, and class
sizes from 20 to 50. These are illustrative hot-arid conditions, not measured
Jodhpur weather or a calibrated campus model.

- `normal`: both AC units may run throughout the day.
- `restricted`: a shared midday cap permits one AC at a time.
- `unequal`: higher solar input increases the effect of unequal room exposure.
- `schedule_changes`: announced cancellations and delays under restricted power.
- `heatwave`: hotter weather, withheld from the training scenario mixture.

Training day seeds are below 1,000,000; validation seeds start at 1,000,000 and
test seeds at 2,000,000. All controllers evaluate the same days and initial
conditions. Evaluation is greedy and never updates model weights or Q-values.
Physical configurations must match across compared models. All evaluation returns
use the evaluation configuration's reward weights, including for ablated models.

Metrics include energy, occupied-room degree-hours, student degree-hours, mean
student discomfort, violation percentage, class-start violation percentage (first
15 minutes), each room's mean discomfort, worst-room discomfort, the gap between
occupied rooms, AC switches, and power-cap violations. Comfort violation percentage
counts occupied room-intervals, not students; student weighting is reported separately.
No-occupancy metrics use zero as a convention.

Run at least three, preferably five, independent training seeds. Pass their
separately named model directories to evaluation; per-run results remain separate.
`summary.csv` standard deviations are over **evaluation days**, not training seeds.
Aggregate independent training runs separately for report-level uncertainty. Do not
claim improvements from a single seed or lower energy paired with much worse comfort.

## Ablations

```bash
python train.py --algo dqn --episodes 300 --seed 0 --no-timetable --output outputs/dqn_no_timetable_seed0
python train.py --algo dqn --episodes 300 --seed 0 --no-fairness --output outputs/dqn_no_fairness_seed0
```

`--no-timetable` removes explicit future class information, while retaining current
occupancy and time. It does not prevent learning typical schedules. `--no-fairness`
removes only the worst-room reward term. Compare each with the full controller on
identical held-out days and repeat across training seeds.

## Structure

- `thermorl/config.py`: units, assumptions and configuration validation.
- `thermorl/scenarios.py`: seeded weather, timetables and power profiles.
- `thermorl/env.py`: Gymnasium environment, transitions, observations and masks.
- `thermorl/q_learning.py`, `thermorl/dqn.py`: original agent implementations.
- `thermorl/baselines.py`: comparison controllers.
- `thermorl/metrics.py`, `thermorl/experiment.py`: metrics, training and evaluation.
- `thermorl/gui.py`, `thermorl/web/`: local server and interactive simulation GUI.
- `tests/`: simulator, agent-update, checkpoint and metric checks.

The implementation provides a runnable experimental foundation.
