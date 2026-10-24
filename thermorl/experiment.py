"""Training and paired, held-out evaluation entry points."""
import argparse
from dataclasses import asdict, replace
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
from .baselines import RuleController
from .config import Config
from .env import ClassroomEnv
from .metrics import summarize
from .q_learning import QLearning
from .scenarios import SCENARIOS

TRAIN_SCENARIOS = SCENARIOS[:-1]  # Heatwave remains an out-of-distribution test.


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty table")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rollout(env, controller, seed):
    observation, info = env.reset(seed=seed)
    records, total_reward = [], 0.0
    while True:
        action = controller.act(observation, info["action_mask"])
        observation, reward, terminated, truncated, info = env.step(action)
        records.append(info)
        total_reward += reward
        if terminated or truncated:
            break
    return summarize(records, env.config.step_minutes) | {"return": total_reward}, records


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Train an original ThermoRL agent on synthetic days")
    parser.add_argument("--algo", choices=["q_learning", "dqn"], required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float,
                        help="Update size; defaults to 0.15 for Q-learning or 0.001 for DQN")
    parser.add_argument("--gamma", type=float, default=0.99, help="Future reward discount in [0, 1]")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--no-timetable", action="store_true")
    parser.add_argument("--no-fairness", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.seed < 0:
        parser.error("episodes must be positive and seed nonnegative")
    # Q-learning and DQN use different default learning rates.
    learning_rate = args.learning_rate if args.learning_rate is not None else (0.001 if args.algo == "dqn" else 0.15)
    if not np.isfinite(learning_rate) or learning_rate <= 0 or (args.algo == "q_learning" and learning_rate > 1):
        parser.error("learning-rate must be finite and positive, and at most 1 for Q-learning")
    if not 0 <= args.gamma <= 1:
        parser.error("gamma must be in [0, 1]")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be empty; use a new run name")
    plotting()  # Fail before training if the plotting installation is broken.
    args.output.mkdir(parents=True, exist_ok=True)
    config = Config.load(args.config)
    if args.no_fairness:
        config = replace(config, fairness_weight=0)
    if args.algo == "dqn":
        from .dqn import DQN
        agent, checkpoint = DQN(seed=args.seed, learning_rate=learning_rate, gamma=args.gamma,
                                use_timetable=not args.no_timetable), "agent.pt"
    else:
        agent, checkpoint = QLearning(seed=args.seed, learning_rate=learning_rate, gamma=args.gamma,
                                      use_timetable=not args.no_timetable), "agent.json"
    config.save(args.output / "config.json")
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "gymnasium", "matplotlib"]}
    if args.algo == "dqn":
        versions["torch"] = importlib.metadata.version("torch")
    metadata = {"algorithm": args.algo, "seed": args.seed, "episodes": args.episodes,
                "training_scenarios": list(TRAIN_SCENARIOS), "checkpoint": checkpoint,
                "use_timetable": not args.no_timetable, "python": platform.python_version(),
                "versions": versions, "checkpoint_purpose": "greedy evaluation, not optimizer resumption",
                "agent_hyperparameters": {"learning_rate": agent.learning_rate, "gamma": agent.gamma},
                "epsilon": {"start": 1.0, "end": 0.05, "decay_fraction": 0.8}}
    if args.algo == "dqn":
        metadata["agent_hyperparameters"].update({"batch_size": agent.batch_size,
            "warmup": agent.warmup, "target_interval": agent.target_interval,
            "replay_capacity": agent.capacity})
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rng = np.random.default_rng(args.seed)
    rows = []
    for episode in range(args.episodes):
        scenario = str(rng.choice(TRAIN_SCENARIOS))
        day_seed = int(rng.integers(0, 1_000_000))
        env = ClassroomEnv(config, scenario)
        observation, info = env.reset(seed=day_seed)
        epsilon = 1 - 0.95 * min(episode / max(1, 0.8 * (args.episodes - 1)), 1)
        records, total_reward, losses = [], 0.0, []
        while True:
            action = agent.act(observation, info["action_mask"], epsilon=epsilon)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Each day is a finite-horizon task; no value is carried across resets.
            loss = agent.learn(observation, action, reward, next_observation, done, info["action_mask"])
            if loss is not None:
                losses.append(loss)
            observation = next_observation
            total_reward += reward
            records.append(info)
            if done:
                break
        rows.append({"episode": episode + 1, "scenario": scenario, "day_seed": day_seed,
                     "epsilon": epsilon, "return": total_reward,
                     "mean_loss": float(np.mean(losses)) if losses else 0.0,
                     "visited_states": len(agent.table) if args.algo == "q_learning" else 0,
                     **summarize(records, config.step_minutes)})
        if episode == 0 or (episode + 1) % max(1, args.episodes // 10) == 0:
            print(f"{args.algo} episode {episode + 1}/{args.episodes}: return={total_reward:.2f}, epsilon={epsilon:.3f}", flush=True)
    agent.save(args.output / checkpoint)
    write_csv(args.output / "training.csv", rows)
    plot_training(rows, args.output, args.algo)
    print(f"Saved evaluation checkpoint and training log to {args.output}")


def plot_training(rows, output, algorithm):
    plt = plotting()
    fig, axis = plt.subplots(figsize=(8, 4))
    returns = np.array([float(r["return"]) for r in rows])
    axis.plot(np.arange(1, len(rows) + 1), returns, alpha=0.35, label="Exploratory daily return")
    window = min(20, len(rows))
    axis.plot(np.arange(window, len(rows) + 1), np.convolve(returns, np.ones(window) / window, mode="valid"), label=f"{window}-day mean")
    axis.set(xlabel="Training episode", ylabel="Return", title=f"{algorithm}: training only")
    axis.legend()
    fig.tight_layout()
    fig.savefig(Path(output) / "training_curve.png", dpi=160)
    plt.close(fig)


def trace_rows(records):
    rows = []
    for record in records:
        row = {key: record[key] for key in ["minute", "outdoor", "action", "power_kw", "cap_kw"]}
        for room in range(2):
            for key in ["temperature", "effective_temperature", "occupancy", "violation"]:
                row[f"{key}_{room}"] = float(record[key][room])
        rows.append(row)
    return rows


def evaluate_main(argv=None):
    parser = argparse.ArgumentParser(description="Compare frozen agents and baselines on paired held-out synthetic days")
    parser.add_argument("--models", type=Path, nargs="*", default=[])
    parser.add_argument("--scenarios", choices=SCENARIOS, nargs="+", default=list(SCENARIOS))
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    if args.days < 1 or args.days > 1_000_000:
        parser.error("days must be between 1 and 1000000")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be empty; use a new evaluation name")
    plotting()  # Validate plotting before running the evaluation batch.
    config = Config.load(args.config or (args.models[0] / "config.json" if args.models else None))
    controllers = {name: RuleController(name) for name in ["thermostat", "timetable", "rotation"]}
    model_metadata = {}
    physical_config = {k: v for k, v in asdict(config).items() if not k.endswith("weight")}
    for folder in args.models:
        if folder.name in controllers:
            parser.error("Model folder names must be unique and differ from baseline names")
        saved = Config.load(folder / "config.json")
        if {k: v for k, v in asdict(saved).items() if not k.endswith("weight")} != physical_config:
            parser.error("All models must use the same physical environment configuration")
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata["algorithm"] == "dqn":
            from .dqn import DQN
            agent = DQN.load(folder / metadata["checkpoint"])
        else:
            agent = QLearning.load(folder / metadata["checkpoint"])
        controllers[folder.name] = agent
        model_metadata[folder.name] = metadata
    args.output.mkdir(parents=True, exist_ok=True)
    config.save(args.output / "config.json")
    seed_offset = 1_000_000 if args.split == "validation" else 2_000_000
    rows, traces = [], {}
    for scenario in args.scenarios:
        for day in range(args.days):
            seed = seed_offset + day
            for name, controller in controllers.items():
                metrics, records = rollout(ClassroomEnv(config, scenario), controller, seed)
                rows.append({"controller": name, "scenario": scenario, "day_seed": seed, **metrics})
                if day == 0 and scenario == args.scenarios[0]:
                    traces[name] = records
        print(f"Evaluated {scenario}: {args.days} paired days", flush=True)
    write_csv(args.output / "daily_metrics.csv", rows)
    numeric = list(rows[0])[3:]
    summary = []
    for scenario in args.scenarios:
        for name in controllers:
            subset = [r for r in rows if r["controller"] == name and r["scenario"] == scenario]
            row = {"controller": name, "scenario": scenario, "days": args.days}
            for metric in numeric:
                values = np.array([r[metric] for r in subset], dtype=float)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary.append(row)
    write_csv(args.output / "summary.csv", summary)
    (args.output / "metadata.json").write_text(json.dumps({
        "split": args.split, "day_seeds": [seed_offset + i for i in range(args.days)],
        "scenarios": args.scenarios, "models": model_metadata,
        "uncertainty": "Summary std is across weather/timetable days, not independent training seeds.",
    }, indent=2) + "\n")
    plt = plotting()
    fig, axis = plt.subplots(figsize=(8, 5))
    for name in controllers:
        subset = [r for r in rows if r["controller"] == name]
        axis.scatter(np.mean([r["energy_kwh"] for r in subset]),
                     np.mean([r["worst_room_mean_discomfort_c"] for r in subset]), label=name)
    axis.set(xlabel="Mean daily energy (kWh)", ylabel="Mean worst-room discomfort (°C)", title="Held-out energy–comfort comparison")
    axis.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(args.output / "comparison.png", dpi=160)
    plt.close(fig)
    for name, records in traces.items():
        write_csv(args.output / f"trace_{name}.csv", trace_rows(records))
        plot_trace(plt, records, args.output / f"trace_{name}.png", name, config)
    print(f"Saved daily metrics, scenario summaries and plots to {args.output}")


def plot_trace(plt, records, path, name, config):
    hour = (np.array([r["minute"] for r in records]) + config.step_minutes) / 60
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for room in range(2):
        axes[0].plot(hour, [r["temperature"][room] for r in records], label=f"Room {room}: air")
        axes[0].plot(hour, [r["effective_temperature"][room] for r in records], linestyle=":", label=f"Room {room}: comfort proxy")
        axes[1].step(hour, [r["occupancy"][room] for r in records], where="pre", label=f"Room {room}")
    axes[0].axhspan(config.comfort_low_c, config.comfort_high_c, color="green", alpha=0.1)
    axes[0].set(ylabel="Temperature (°C)", title=f"{name}: one held-out day")
    axes[1].set(ylabel="Occupants")
    axes[2].step(hour, [r["power_kw"] for r in records], where="pre", label="Cooling equipment")
    axes[2].step(hour, [r["cap_kw"] for r in records], where="pre", linestyle="--", label="Power limit")
    axes[2].set(ylabel="Power (kW)", xlabel="Hour of day", xlim=(0, 24))
    for axis in axes:
        axis.legend(fontsize="small", loc="upper right")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
