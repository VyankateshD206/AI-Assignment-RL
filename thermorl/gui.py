"""Local GUI server. Every displayed trajectory comes from the Python environment.

The browser only plays back immutable rollouts; there is no duplicate JS thermal
model, training during playback, external service, or hardware connection.
"""
import argparse
import csv
from dataclasses import asdict
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .baselines import RuleController
from .config import Config
from .env import ACTIONS, ClassroomEnv
from .metrics import METRIC_DEFINITIONS, summarize
from .q_learning import QLearning
from .scenarios import SCENARIOS

WEB_ROOT = Path(__file__).with_name("web")
BASELINES = {
    "timetable": ("Timetable controller", "Pre-cools rooms 30 minutes before an announced class."),
    "thermostat": ("Occupancy thermostat", "Responds to occupied-room temperatures with hysteresis."),
    "rotation": ("Fair rotation", "Alternates cooling priority between rooms every 15 minutes."),
}
SCENARIO_DETAILS = {
    "normal": ("Normal operation", "Both AC units can run all day. A useful reference case."),
    "restricted": ("Shared power limit", "From 11:00 to 16:00, the 3.3 kW allowance supports one AC and both fans."),
    "unequal": ("Stronger solar exposure", "Higher solar input makes the sun-exposed room harder to cool. The midday power limit still applies."),
    "schedule_changes": ("Timetable changes", "Announced cancellations and delays test whether controllers avoid unnecessary pre-cooling."),
    "heatwave": ("Heatwave", "Outdoor temperatures are 4°C higher. This scenario is withheld from the training mixture."),
}


def physical_config(config):
    return {k: v for k, v in asdict(config).items() if not k.endswith("weight")}


class SimulationApp:
    """Read-only model registry and bounded cache of reproducible full-day rollouts."""

    def __init__(self, config=None, model_paths=(), study_path=None):
        self.config = config or Config()
        self.controllers = {key: RuleController(key) for key in BASELINES}
        self.model_details = []
        self.warnings = []
        self.study = None
        if study_path:
            try:
                summary = json.loads(Path(study_path).read_text())
                if (summary['total_days'] != summary['scenario_count'] * summary['days_per_scenario'] or
                        summary['partition'] not in ('tuning', 'confirmation', 'final')):
                    raise ValueError('invalid study dimensions or partition')
                self.study = summary
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.warnings.append(f'Could not load saved study summary: {error}')
        for folder in model_paths:
            folder = Path(folder).resolve()
            try:
                metadata = json.loads((folder / "metadata.json").read_text())
                if not metadata.get("training_complete", True):
                    raise ValueError("training is still in progress")
                model_config = Config.load(folder / "config.json")
                if physical_config(model_config) != physical_config(self.config):
                    raise ValueError("physical environment configuration differs from the GUI configuration")
                algorithm = metadata["algorithm"]
                if algorithm not in ("q_learning", "dqn"):
                    raise ValueError("unsupported algorithm")
                checkpoint = (folder / metadata["checkpoint"]).resolve()
                if not checkpoint.is_relative_to(folder):
                    raise ValueError("checkpoint must be inside its run directory")
                if algorithm == "dqn":
                    from .dqn import DQN
                    controller = DQN.load(checkpoint)
                    name = "DQN"
                else:
                    controller = QLearning.load(checkpoint)
                    name = "Q-learning"
                key = f"model_{len(self.model_details)}"
                self.controllers[key] = controller
                self.model_details.append({
                    "id": key, "label": f"{name} · {folder.name}", "kind": "rl",
                    "description": "Frozen learned policy; greedy evaluation, no training during playback.",
                    "algorithm": algorithm, "episodes": metadata.get("episodes"),
                    "training_seed": metadata.get("seed"), "run": folder.name,
                    "selected_episode": metadata.get("selected_episode", metadata.get("episodes")),
                    "selection": metadata.get("selection", "Final checkpoint; no periodic validation"),
                    "selection_mode": metadata.get('selection_mode', 'return'),
                    "comfort_floor": metadata.get('comfort_floor'),
                    "selection_comfort_floor_met": metadata.get('selection_comfort_floor_met'),
                    "initialization": metadata.get('initialization'),
                    "feature_mode": getattr(controller, "feature_mode", getattr(controller, "state_mode", "legacy")),
                    "network_inputs": getattr(controller, "input_size", None),
                    "reward_scale": getattr(controller, "reward_scale", 1.0),
                    "validation_history": self.read_validation(folder / "validation.csv"),
                    "selected_validation": metadata.get("selected_validation", {}),
                    "use_timetable": metadata.get("use_timetable", True),
                    "hyperparameters": metadata.get("agent_hyperparameters", {
                        "learning_rate": controller.learning_rate, "gamma": controller.gamma}),
                    "training_history": self.read_history(folder / "training.csv"),
                })
            except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
                self.warnings.append(f"Could not load {folder.name}: {error}")

    @staticmethod
    def read_history(path):
        """Display saved measurements only; missing logs are never fabricated."""
        if not path.exists():
            return []
        try:
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            history = [{key: float(row[key]) for key in
                        ("episode", "return", "epsilon", "mean_loss", "visited_states")}
                       for row in rows]
            if not all(np.isfinite(list(row.values())).all() for row in history):
                return []
            return history
        except (OSError, ValueError, KeyError, TypeError):
            return []

    @staticmethod
    def read_validation(path):
        fields = ("episode", "mean_return", "mean_energy_kwh", "mean_comfort_pct",
                  "mean_overheating_pct", "mean_overcooling_pct")
        try:
            with path.open() as stream:
                rows = [{key: float(row[key]) for key in fields} for row in csv.DictReader(stream)]
            return rows if all(np.isfinite(list(row.values())).all() for row in rows) else []
        except (OSError, ValueError, KeyError, TypeError):
            return []

    def catalog(self):
        return {
            "controllers": [{"id": key, "label": label, "description": description,
                             "kind": "baseline", "role": "comparison_only"}
                            for key, (label, description) in BASELINES.items()] + self.model_details,
            "scenarios": [{"id": key, "label": SCENARIO_DETAILS[key][0],
                           "description": SCENARIO_DETAILS[key][1]} for key in SCENARIOS],
            "config": asdict(self.config), "warnings": self.warnings,
            "simulation_only": True,
            "study": self.study,
        }

    @lru_cache(maxsize=32)
    def simulate(self, controller_id, scenario, seed):
        if controller_id not in self.controllers:
            raise ValueError("Unknown controller")
        if scenario not in SCENARIOS:
            raise ValueError("Unknown scenario")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be an integer between 0 and 4294967295")
        controller = self.controllers[controller_id]
        env = ClassroomEnv(self.config, scenario)
        observation, info = env.reset(seed=seed)
        c = self.config
        next_minutes, next_counts = env.next_classes()
        frames = [{
            "minute": 0, "temperature": env.temperature.tolist(),
            "effective_temperature": env.temperature.tolist(),
            "outdoor": float(env.day.outdoor[0]), "solar": float(env.day.solar[0]),
            "occupancy": env.day.occupancy[0].tolist(), "power_kw": 0.0,
            "cap_kw": float(env.day.cap[0]), "equipment": [[0, 0], [0, 0]],
            "action": None, "reward": 0.0, "return": 0.0, "energy_kwh": 0.0,
            "class_start": [False, False],
            "arrival_comfort_proxy": [None, None], "arrival_violation": [None, None],
            "room_energy_kwh": [0.0, 0.0], "comfort_pct": None,
            "worst_room_discomfort_c": 0.0, "ac_switches": 0, "cap_violations": 0,
            "next_class_minutes": next_minutes.tolist(), "next_class_occupancy": next_counts.tolist(),
            "feasible_actions": int(info["action_mask"].sum()),
            "reward_terms": {"energy": 0.0, "discomfort": 0.0, "fairness": 0.0, "switching": 0.0},
        }]
        records = []
        total_return = energy = 0.0
        switches = cap_violations = 0
        room_energy = np.zeros(2)
        occupied_intervals = np.zeros(2)
        degree_intervals = np.zeros(2)
        comfortable = 0
        while True:
            mask = info["action_mask"]
            feasible = int(mask.sum())
            next_minutes, next_counts = env.next_classes()
            t = env.t
            before = env.temperature.copy()
            input_observation = observation.copy()
            q_values = None
            state_seen = None
            if isinstance(controller, QLearning):
                key = controller.state_key(observation)
                state_seen = key in controller.table
                q_values = controller.table.get(key, np.zeros(16)).tolist()
            elif controller_id not in BASELINES:
                import torch
                with torch.no_grad():
                    q_values = controller.online(torch.from_numpy(controller.features(observation))).tolist()
            action = controller.act(observation, mask)
            observation, reward, terminated, truncated, info = env.step(action)
            records.append(info)
            total_return += reward
            energy += info["energy_kwh"]
            switches += info["switches"]
            cap_violations += int(info["power_kw"] > info["cap_kw"] + 1e-9)
            equipment = ACTIONS[action]
            room_energy += (equipment[:, 0] * c.ac_electric_kw + equipment[:, 1] * c.fan_electric_kw) * c.step_minutes / 60
            occupied = info["occupancy"] > 0
            occupied_intervals += occupied
            degree_intervals += info["violation"] * occupied
            comfortable += int(np.sum(occupied & (info["violation"] == 0)))
            room_mean = np.divide(degree_intervals, occupied_intervals,
                                  out=np.zeros(2), where=occupied_intervals > 0)
            frames.append({
                "minute": (t + 1) * c.step_minutes,
                "temperature_before": before.tolist(),
                "class_start": info["class_start"].tolist(),
                "arrival_comfort_proxy": info["arrival_comfort_proxy"].tolist(),
                "arrival_violation": info["arrival_violation"].tolist(),
                "observation": input_observation.tolist(),
                "action_mask": mask.tolist(), "q_values": q_values, "state_seen": state_seen,
                "heat_terms_kw": {
                    "outdoor": (np.array(c.conductance_kw_per_c) * (env.day.outdoor[t] - before)).tolist(),
                    "solar": (np.array(c.solar_gain_kw_per_wm2) * env.day.solar[t]).tolist(),
                    "occupants": (info["occupancy"] * c.person_heat_kw).tolist(),
                    "fan": (equipment[:, 1] * c.fan_electric_kw).tolist(),
                    "ac": (-equipment[:, 0] * c.ac_cooling_kw).tolist(),
                },
                "interval_start_minute": t * c.step_minutes,
                "temperature": info["temperature"].tolist(),
                "effective_temperature": info["effective_temperature"].tolist(),
                "outdoor": info["outdoor"], "solar": float(env.day.solar[t]),
                "occupancy": info["occupancy"].tolist(), "power_kw": info["power_kw"],
                "cap_kw": info["cap_kw"], "equipment": equipment.tolist(), "action": action,
                "reward": reward, "return": total_return, "energy_kwh": energy,
                "room_energy_kwh": room_energy.tolist(),
                "comfort_pct": 100 * comfortable / occupied_intervals.sum() if occupied_intervals.sum() else None,
                "worst_room_discomfort_c": float(max(room_mean)),
                "ac_switches": switches, "cap_violations": cap_violations,
                "next_class_minutes": next_minutes.tolist(), "next_class_occupancy": next_counts.tolist(),
                "feasible_actions": feasible,
                "reward_terms": {key: info[f"reward_{key}"] for key in ["energy", "discomfort", "fairness", "switching"]},
            })
            if terminated or truncated:
                break
        timetable = []
        for room in range(2):
            classes = []
            for t in np.flatnonzero(env.day.class_start[:, room]):
                end = int(t) + 1
                while end < env.horizon and env.day.occupancy[end, room] > 0 and not env.day.class_start[end, room]:
                    end += 1
                classes.append({"start": int(t) * c.step_minutes, "end": end * c.step_minutes,
                                "occupancy": int(env.day.occupancy[t, room])})
            timetable.append(classes)
        return {
            "controller": controller_id, "scenario": scenario, "seed": seed,
            "step_minutes": c.step_minutes, "frames": frames,
            "metric_definitions": METRIC_DEFINITIONS,
            "metrics": summarize(records, c.step_minutes) | {"return": total_return},
            "timetable": timetable,
        }


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logging.info(fmt, *args)

        def send_bytes(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'self'")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, data):
            self.send_bytes(status, json.dumps(data, allow_nan=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            target = urlsplit(self.path)
            try:
                if target.path == "/api/catalog":
                    self.send_json(200, app.catalog())
                elif target.path == "/api/simulate":
                    query = parse_qs(target.query)
                    controller = query.get("controller", ["timetable"])[0]
                    scenario = query.get("scenario", ["restricted"])[0]
                    seed = int(query.get("seed", ["42"])[0])
                    self.send_json(200, app.simulate(controller, scenario, seed))
                elif target.path in {"/", "/index.html", "/styles.css", "/app.js", "/favicon.svg", "/UbuntuSans.ttf", "/UbuntuSansMono.ttf", "/FONT-LICENSE.txt"}:
                    name = "index.html" if target.path == "/" else target.path[1:]
                    mime = {".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml", ".ttf": "font/ttf", ".txt": "text/plain"}[Path(name).suffix]
                    self.send_bytes(200, (WEB_ROOT / name).read_bytes(), f"{mime}; charset=utf-8")
                else:
                    self.send_json(404, {"error": "Not found"})
            except (ValueError, TypeError) as error:
                self.send_json(400, {"error": str(error)})
            except Exception:
                logging.exception("Simulation request failed")
                self.send_json(500, {"error": "The simulation could not be completed. Check the Python terminal for details."})
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the local Python-backed ThermoRL simulation GUI")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", type=Path)
    parser.add_argument('--study', type=Path, help='Optional saved study summary JSON; separate from demo-day results')
    parser.add_argument("--models", nargs="*", type=Path, default=None,
                        help="Model run directories; default: curated models/ bundles, falling back to outputs/")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    bundled = sorted(Path("models").glob("*/metadata.json"))
    models = args.models if args.models is not None else (bundled or sorted(Path("outputs").glob("*/metadata.json")))
    if args.models is None:
        discovered = []
        for path in models:
            try:
                if ((path.parent / "config.json").exists()
                        and json.loads(path.read_text()).get("algorithm") in ("q_learning", "dqn")):
                    discovered.append(path.parent)
            except (OSError, ValueError, AttributeError):
                print(f"Warning: skipping unreadable metadata in {path.parent.name}", flush=True)
        models = discovered
    app = SimulationApp(Config.load(args.config), models, study_path=args.study)
    for warning in app.warnings:
        print(f"Warning: {warning}", flush=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    except OSError as error:
        parser.error(str(error))
    print(f"ThermoRL simulation: http://127.0.0.1:{args.port}", flush=True)
    print(f"Python engine ready · {len(app.controllers)} controllers · simulation only", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSimulation server stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
