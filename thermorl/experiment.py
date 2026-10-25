"""Training and paired, held-out evaluation entry points."""
import argparse
from dataclasses import asdict, replace
import csv
import importlib.metadata
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
from .baselines import RuleController
from .config import Config
from .env import ClassroomEnv
from .metrics import METRIC_DEFINITIONS, pooled_comfort_metrics, summarize
from .q_learning import QLearning
from .scenarios import SCENARIOS

TRAIN_SCENARIOS = SCENARIOS[:-1]  # Heatwave remains an out-of-distribution test.


def checkpoint_rank(metrics, selection, comfort_floor, reference=None):
    """Predeclared ordering; missing the comfort floor is never success."""
    if selection == 'return':
        return (metrics['mean_return'],)
    if selection == 'nonregression':
        from .study import nonregression
        eligible = reference is not None and nonregression(metrics, reference)
        return (int(eligible), -metrics['mean_energy_kwh'] if eligible else metrics['mean_return'],
                metrics['mean_comfort_pct'], metrics['mean_return'])
    comfort, energy = metrics['mean_comfort_pct'], metrics['mean_energy_kwh']
    feasible = comfort >= comfort_floor
    return (int(feasible), -energy if feasible else comfort,
            comfort if feasible else -energy, metrics['mean_return'])


def initialize_dqn(agent, folder, config):
    """Warm-start weights only; optimizer, replay and exploration are fresh."""
    from .dqn import DQN
    metadata = json.loads((folder / 'metadata.json').read_text())
    if metadata['algorithm'] != 'dqn' or not metadata.get('training_complete', True) or asdict(Config.load(folder / 'config.json')) != asdict(config):
        raise ValueError('Warm start requires a DQN with the same environment and reward')
    source = folder / metadata['checkpoint']
    saved = DQN.load(source)
    if (saved.feature_mode, saved.comfort_low, saved.comfort_high, saved.reward_scale, saved.use_timetable) != (
            agent.feature_mode, agent.comfort_low, agent.comfort_high, agent.reward_scale, agent.use_timetable):
        raise ValueError('Warm-start representation, reward scale and timetable inputs must match')
    agent.online.load_state_dict(saved.online.state_dict())
    agent.target.load_state_dict(saved.online.state_dict())
    return {'source_run': str(folder), 'source_checkpoint_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'source_selected_episode': metadata.get('selected_episode', metadata.get('episodes')),
            'replay_optimizer_reset': True}


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
        seen = (controller.state_key(observation) in controller.table) if isinstance(controller, QLearning) else None
        action = controller.act(observation, info["action_mask"])
        observation, reward, terminated, truncated, info = env.step(action)
        info["tabular_state_seen"] = seen
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


def reward_diagnostics(records):
    result = {"penalty_" + key: float(sum(r["reward_" + key] for r in records))
              for key in ("energy", "discomfort", "fairness", "switching")}
    magnitudes = [sum(r["reward_" + key] for key in ("energy", "discomfort", "fairness", "switching")) for r in records]
    result["absolute_reward_p95"] = float(np.percentile(magnitudes, 95))
    return result


def validate_agent(agent, config, days, seed_offset=1_000_000, aggregation='daily_mean'):
    """Frozen greedy validation on fixed seeds, excluding the held-out heatwave."""
    measurements, seen = [], []
    for scenario in TRAIN_SCENARIOS:
        for day in range(days):
            metrics, records = rollout(ClassroomEnv(config, scenario), agent, seed_offset + day)
            metrics.update(reward_diagnostics(records))
            measurements.append(metrics)
            seen.extend(r["tabular_state_seen"] for r in records if r["tabular_state_seen"] is not None)
    result = {}
    for key in measurements[0]:
        values = [m[key] for m in measurements if m[key] is not None]
        result['mean_' + key] = float(np.mean(values)) if values else None
    result["unseen_state_pct"] = 100 * (1 - float(np.mean(seen))) if seen else ""
    if aggregation == 'pooled':
        from .study import score_days
        result.update(score_days(measurements))
    if isinstance(agent, QLearning):
        updated = sum(int(np.count_nonzero(counts)) for counts in agent.action_visits.values())
        result['stored_action_entries_updated_pct'] = (100 * updated / (16 * len(agent.table))
            if agent.table and agent.action_visits else None)
    return result


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Train an original ThermoRL agent on synthetic days")
    parser.add_argument("--algo", choices=["q_learning", "dqn"], required=True)
    parser.add_argument("--episodes", type=int, help="Training budget; defaults to 400 for DQN or 1500 for Q-learning")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float,
                        help="Update size; defaults to 0.15 for Q-learning or 0.0003 for DQN")
    parser.add_argument("--gamma", type=float, default=0.99, help="Future reward discount in [0, 1]")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--no-timetable", action="store_true")
    parser.add_argument("--no-fairness", action="store_true")
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--validation-days", type=int, default=2, help="Fixed validation days per in-distribution scenario")
    parser.add_argument('--validation-seed-offset', type=int, default=1_000_000)
    parser.add_argument('--validation-aggregation', choices=['daily_mean', 'pooled'], default='daily_mean')
    parser.add_argument('--reference-model', type=Path, help='Frozen DQN bundle for nonregression checkpoint selection')
    parser.add_argument("--epsilon-decay", type=float, default=0.8, help="Fraction of episodes before epsilon reaches 0.05")
    parser.add_argument("--dqn-features", choices=["legacy", "comfort_v1"], default="comfort_v1")
    parser.add_argument("--reward-scale", type=float, default=10.0, help="Divide DQN training rewards by this positive constant; physical rewards stay unchanged")
    parser.add_argument("--q-state", choices=["legacy", "compact_v1", "comfort_v2"], default="compact_v1")
    parser.add_argument('--q-visit-decay', type=float, default=0.0)
    parser.add_argument('--q-count-exploration', action='store_true')
    parser.add_argument('--epsilon-end', type=float, default=0.05)
    parser.add_argument('--epsilon-start', type=float, default=1.0)
    parser.add_argument('--initialize-from', type=Path, help='DQN run directory: copy selected weights, start fresh replay and optimizer')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--target-interval', type=int, default=250, help='DQN optimizer updates between target copies')
    parser.add_argument('--train-frequency', type=int, default=1, help='Environment transitions per DQN optimizer update')
    parser.add_argument('--selection', choices=['return', 'comfort_energy', 'nonregression'], default='return')
    parser.add_argument('--comfort-floor', type=float, default=90.0, help='Validation comfort threshold for comfort_energy selection')
    args = parser.parse_args(argv)
    if args.episodes is None:
        args.episodes = 400 if args.algo == "dqn" else 1500
    if args.validation_every < 1 or args.validation_days < 1 or not 0 < args.epsilon_decay <= 1:
        parser.error("Validation sizes must be positive and epsilon-decay in (0, 1]")
    if not 1_000_000 <= args.validation_seed_offset < args.validation_seed_offset + args.validation_days <= 2_000_000:
        parser.error('Validation seeds must stay in [1000000, 1999999], outside training')
    if args.selection == 'nonregression' and (args.algo != 'dqn' or not args.reference_model or args.validation_aggregation != 'pooled'):
        parser.error('nonregression requires DQN, reference-model and pooled validation')
    if not np.isfinite(args.reward_scale) or args.reward_scale <= 0:
        parser.error("reward-scale must be finite and positive")
    if not 0 <= args.epsilon_end <= 1 or not 0 <= args.comfort_floor <= 100:
        parser.error('epsilon-end must be in [0,1] and comfort-floor in [0,100]')
    if not args.epsilon_end <= args.epsilon_start <= 1 or (args.initialize_from and args.algo != 'dqn'):
        parser.error('epsilon-start must be between epsilon-end and1; initialize-from requires DQN')
    if not np.isfinite(args.q_visit_decay) or args.q_visit_decay < 0:
        parser.error('q-visit-decay must be finite and nonnegative')
    if args.batch_size < 1 or args.warmup < 0 or max(args.batch_size, args.warmup) > 50000 or min(args.target_interval, args.train_frequency) < 1:
        parser.error('Invalid DQN batch, warmup, target interval or training frequency')
    if args.episodes < 1 or args.seed < 0:
        parser.error("episodes must be positive and seed nonnegative")
    # Q-learning and DQN use different default learning rates.
    learning_rate = args.learning_rate if args.learning_rate is not None else (0.0003 if args.algo == "dqn" else 0.15)
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
                                use_timetable=not args.no_timetable, feature_mode=args.dqn_features,
                                comfort_low=config.comfort_low_c, comfort_high=config.comfort_high_c,
                                reward_scale=args.reward_scale, batch_size=args.batch_size, warmup=args.warmup,
                                target_interval=args.target_interval, train_frequency=args.train_frequency), "agent.pt"
    else:
        agent, checkpoint = QLearning(seed=args.seed, learning_rate=learning_rate, gamma=args.gamma,
                                      use_timetable=not args.no_timetable, state_mode=args.q_state,
                                      comfort_low=config.comfort_low_c, comfort_high=config.comfort_high_c,
                                      visit_decay=args.q_visit_decay, count_exploration=args.q_count_exploration), "agent.json"
    initialization = initialize_dqn(agent, args.initialize_from, config) if args.initialize_from else None
    reference_validation, reference_model = None, None
    if args.reference_model:
        from .dqn import DQN
        folder = args.reference_model
        saved = json.loads((folder / 'metadata.json').read_text())
        if saved['algorithm'] != 'dqn' or not saved.get('training_complete', True) or asdict(Config.load(folder / 'config.json')) != asdict(config):
            parser.error('Reference must be a completed DQN with identical environment and reward')
        source = folder / saved['checkpoint']
        reference_model = {'path': str(folder), 'checkpoint_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        reference_validation = validate_agent(DQN.load(source), config, args.validation_days,
            args.validation_seed_offset, args.validation_aggregation)
    config.save(args.output / "config.json")
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "gymnasium", "matplotlib"]}
    if args.algo == "dqn":
        versions["torch"] = importlib.metadata.version("torch")
    metadata = {"algorithm": args.algo, "seed": args.seed, "episodes": args.episodes,
                "training_scenarios": list(TRAIN_SCENARIOS), "checkpoint": checkpoint,
                "training_complete": False, "completed_episodes": 0,
                "use_timetable": not args.no_timetable, "python": platform.python_version(),
                "versions": versions, "checkpoint_purpose": "greedy evaluation, not optimizer resumption",
                "agent_hyperparameters": {"learning_rate": agent.learning_rate, "gamma": agent.gamma},
                "epsilon": {"start": args.epsilon_start, "end": args.epsilon_end, "decay_fraction": args.epsilon_decay},
                "initialization": initialization,
                "reference_model": reference_model, "reference_validation": reference_validation,
                "selection_mode": args.selection, "comfort_floor": args.comfort_floor,
                "selection": ("Highest mean original (unscaled) return on fixed validation days; earliest checkpoint wins ties" if args.selection == 'return' else
                              f"Lowest mean daily energy among checkpoints with mean validation comfort >= {args.comfort_floor}%; if none qualify, highest comfort. Earliest wins exact ties."),
                "validation": {"every": args.validation_every, "days_per_scenario": args.validation_days,
                               "aggregation": args.validation_aggregation,
                               "scenarios": list(TRAIN_SCENARIOS), "seeds": [args.validation_seed_offset + d for d in range(args.validation_days)]}}
    if args.selection == 'nonregression':
        metadata['selection'] = 'Preserve all three pooled comfort quantities and reduce energy versus paired frozen reference; rank qualifying checkpoints by lower energy, higher comfort, higher original return; earliest exact-rank tie. Fallback highest return is ineligible'
    if args.algo == "dqn":
        metadata["agent_hyperparameters"].update({"batch_size": agent.batch_size,
            "warmup": agent.warmup, "target_interval": agent.target_interval,
            "replay_capacity": agent.capacity, "feature_mode": agent.feature_mode,
            "network_inputs": agent.input_size, "reward_scale": agent.reward_scale,
            "train_frequency": agent.train_frequency})
    else:
        metadata["agent_hyperparameters"]["state_mode"] = agent.state_mode
        metadata['agent_hyperparameters'].update({'visit_decay': agent.visit_decay, 'count_exploration': agent.count_exploration})
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rng = np.random.default_rng(args.seed)
    rows, validation_rows = [], []
    best_rank = None
    def validate(episode):
        nonlocal best_rank
        metadata["completed_episodes"] = episode
        metrics = validate_agent(agent, config, args.validation_days, args.validation_seed_offset, args.validation_aggregation)
        rank = checkpoint_rank(metrics, args.selection, args.comfort_floor, reference_validation)
        improved = best_rank is None or rank > best_rank
        if improved:
            best_rank = rank
            agent.save(args.output / checkpoint)
            metadata["selected_episode"] = episode
            metadata["selected_validation"] = metrics
            metadata['selection_comfort_floor_met'] = metrics['mean_comfort_pct'] >= args.comfort_floor
            if args.selection == 'nonregression':
                metadata['selection_nonregression_met'] = bool(rank[0])
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        validation_rows.append({"episode": episode, "selected_when_evaluated": improved, **metrics})
        write_csv(args.output / "validation.csv", validation_rows)
        print(f"validation episode {episode}: comfort={metrics['mean_comfort_pct']:.1f}% energy={metrics['mean_energy_kwh']:.2f} return={metrics['mean_return']:.2f}" + (" [best]" if improved else ""), flush=True)
    validate(0)
    for episode in range(args.episodes):
        scenario = str(rng.choice(TRAIN_SCENARIOS))
        day_seed = int(rng.integers(0, 1_000_000))
        env = ClassroomEnv(config, scenario)
        observation, info = env.reset(seed=day_seed)
        epsilon = args.epsilon_start - (args.epsilon_start - args.epsilon_end) * min(episode / max(1, args.epsilon_decay * (args.episodes - 1)), 1)
        records, total_reward, losses = [], 0.0, []
        while True:
            action = agent.act(observation, info["action_mask"], epsilon=epsilon)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Each day is a finite-horizon task; no value is carried across resets.
            loss = agent.learn(observation, action, reward, next_observation, done, info["action_mask"],
                               **({'scenario': scenario} if args.algo == 'dqn' else {}))
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
                     **(agent.training_diagnostics() if args.algo == 'dqn' else {}),
                     **summarize(records, config.step_minutes), **reward_diagnostics(records)})
        # Save incremental logs so interrupted experiments retain diagnostic evidence.
        if (episode + 1) % args.validation_every == 0 or episode + 1 == args.episodes:
            write_csv(args.output / "training.csv", rows)
            validate(episode + 1)
        if episode == 0 or (episode + 1) % max(1, args.episodes // 10) == 0:
            print(f"{args.algo} episode {episode + 1}/{args.episodes}: return={total_reward:.2f}, epsilon={epsilon:.3f}", flush=True)
    agent.save(args.output / ("last_" + checkpoint))
    write_csv(args.output / "training.csv", rows)
    plot_training(rows, args.output, args.algo)
    metadata["training_complete"] = True
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Best checkpoint: episode {metadata['selected_episode']} of {args.episodes}", flush=True)
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
            for key in ["temperature", "temperature_before", "effective_temperature", "occupancy", "violation",
                        "class_start", "arrival_comfort_proxy", "arrival_violation"]:
                row[f"{key}_{room}"] = float(record[key][room])
        rows.append(row)
    return rows


def evaluate_main(argv=None):
    parser = argparse.ArgumentParser(description="Compare frozen agents and baselines on paired held-out synthetic days")
    parser.add_argument("--models", type=Path, nargs="*", default=[])
    parser.add_argument("--scenarios", choices=SCENARIOS, nargs="+", default=list(SCENARIOS))
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument('--seed-offset', type=int, help='First paired day seed; defaults to 1000000 validation / 2000000 test')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    if args.days < 1 or args.days > 1_000_000:
        parser.error("days must be between 1 and 1000000")
    seed_offset = args.seed_offset if args.seed_offset is not None else (1_000_000 if args.split == 'validation' else 2_000_000)
    lower, upper = (1_000_000, 1_999_999) if args.split == 'validation' else (2_000_000, 2**32 - 1)
    if not lower <= seed_offset <= upper or seed_offset + args.days - 1 > upper:
        parser.error('Evaluation seeds must stay outside training and in the selected validation/test partition')
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
    rows, traces, diagnostics = [], {}, []
    for scenario in args.scenarios:
        for day in range(args.days):
            seed = seed_offset + day
            for name, controller in controllers.items():
                metrics, records = rollout(ClassroomEnv(config, scenario), controller, seed)
                rows.append({"controller": name, "scenario": scenario, "day_seed": seed, **metrics})
                seen = [r["tabular_state_seen"] for r in records if r["tabular_state_seen"] is not None]
                diagnostics.append({"controller": name, "scenario": scenario, "day_seed": seed,
                    "unseen_state_pct": 100 * (1 - float(np.mean(seen))) if seen else "",
                    **reward_diagnostics(records)})
                if day == 0 and scenario == args.scenarios[0]:
                    traces[name] = records
        print(f"Evaluated {scenario}: {args.days} paired days", flush=True)
    write_csv(args.output / "daily_metrics.csv", rows)
    write_csv(args.output / "diagnostics.csv", diagnostics)
    numeric = list(rows[0])[3:]
    summary = []
    for scenario in args.scenarios:
        for name in controllers:
            subset = [r for r in rows if r["controller"] == name and r["scenario"] == scenario]
            row = {"controller": name, "scenario": scenario, "days": args.days}
            for metric in numeric:
                values = np.array([r[metric] for r in subset if r[metric] is not None], dtype=float)
                row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
                row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row.update({key + '_pooled': value for key, value in pooled_comfort_metrics(subset).items()})
            summary.append(row)
    write_csv(args.output / "summary.csv", summary)
    (args.output / "metadata.json").write_text(json.dumps({
        "split": args.split, "day_seeds": [seed_offset + i for i in range(args.days)],
        "scenarios": args.scenarios, "models": model_metadata,
        "metric_definitions": METRIC_DEFINITIONS,
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
