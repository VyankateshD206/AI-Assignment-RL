"""GUI data must remain faithful to the research environment; no duplicated physics."""
import io
import json
from pathlib import Path

import numpy as np
import pytest

from thermorl.baselines import RuleController
from thermorl.config import Config
from thermorl.env import ACTIONS, ClassroomEnv
from thermorl.experiment import rollout
from thermorl.gui import SimulationApp, make_handler
from thermorl.q_learning import QLearning


@pytest.mark.parametrize("controller", ["thermostat", "timetable", "rotation"])
def test_gui_rollout_matches_reference_environment(controller):
    app = SimulationApp()
    data = app.simulate(controller, "restricted", 42)
    metrics, records = rollout(ClassroomEnv(), RuleController(controller), 42)
    assert data["metrics"] == pytest.approx(metrics)
    assert len(data["frames"]) == len(records) + 1 == 289
    assert data["frames"][0]["minute"] == 0
    assert data["frames"][-1]["minute"] == 1440
    for frame, record in zip(data["frames"][1:], records):
        assert frame["minute"] == record["minute"] + 5
        assert frame["interval_start_minute"] == record["minute"]
        np.testing.assert_array_equal(frame["temperature"], record["temperature"])
        np.testing.assert_array_equal(frame["occupancy"], record["occupancy"])
        np.testing.assert_array_equal(frame["class_start"], record["class_start"])
        np.testing.assert_array_equal(frame["temperature_before"], record["temperature_before"])
        np.testing.assert_array_equal(frame["arrival_comfort_proxy"], record["arrival_comfort_proxy"])
        np.testing.assert_array_equal(frame["arrival_violation"], record["arrival_violation"])
        np.testing.assert_array_equal(frame["equipment"], ACTIONS[record["action"]])
        assert frame["power_kw"] <= frame["cap_kw"] + 1e-9
        assert frame["reward"] == pytest.approx(-sum(frame["reward_terms"].values()))
    final = data["frames"][-1]
    assert final["energy_kwh"] == pytest.approx(metrics["energy_kwh"])
    assert sum(final["room_energy_kwh"]) == pytest.approx(metrics["energy_kwh"])
    assert final["comfort_pct"] == pytest.approx(100 - metrics["comfort_violation_pct"])
    assert final["worst_room_discomfort_c"] == pytest.approx(metrics["worst_room_mean_discomfort_c"])
    assert final["cap_violations"] == metrics["cap_violations"]
    # Public payload contains finite JSON primitives only.
    json.dumps(data, allow_nan=False)


def test_initial_comfort_is_not_misleading_and_seed_reproducible():
    app = SimulationApp()
    first = app.simulate("timetable", "heatwave", 14)
    second = SimulationApp().simulate("timetable", "heatwave", 14)
    assert first == second
    assert first["frames"][0]["comfort_pct"] is None
    assert len(first["timetable"]) == 2
    assert all(len(classes) == 4 for classes in first["timetable"])


def test_gui_no_classes_exposes_null_arrival_and_zero_time_denominators(monkeypatch):
    import thermorl.env as env_module
    original = env_module.make_day
    def empty_day(*args):
        day = original(*args)
        day.occupancy[:] = 0
        day.class_start[:] = False
        return day
    monkeypatch.setattr(env_module, 'make_day', empty_day)
    header, body = request(SimulationApp(), '/api/simulate?controller=timetable')
    assert '200 OK' in header
    m = json.loads(body)['metrics']
    assert m['arrival_discomfort_pct'] is None
    assert m['occupied_room_minutes'] == m['early_class_occupied_minutes'] == m['class_arrivals'] == 0


@pytest.mark.parametrize("seed", [-1, 2**32, 0.5, True])
def test_gui_rejects_invalid_seeds(seed):
    with pytest.raises(ValueError, match="Seed"):
        SimulationApp().simulate("timetable", "restricted", seed)


def test_model_registry_loads_checkpoints_and_ignores_bad_runs(tmp_path):
    folder = tmp_path / "test_policy"
    folder.mkdir()
    Config().save(folder / "config.json")
    QLearning().save(folder / "agent.json")
    (folder / "metadata.json").write_text(json.dumps({
        "algorithm": "q_learning", "checkpoint": "agent.json", "episodes": 2, "seed": 7,
    }))
    bad = tmp_path / "bad_run"
    app = SimulationApp(model_paths=[folder, bad])
    assert len(app.catalog()["controllers"]) == 4
    assert app.model_details[0]["episodes"] == 2
    assert app.warnings and "bad_run" in app.warnings[0]
    data = app.simulate("model_0", "normal", 42)
    assert len(data["frames"]) == 289


class MemoryConnection:
    """Exercise the real HTTP handler without sandbox-sensitive network sockets."""
    def __init__(self, request):
        self.request = io.BytesIO(request)
        self.response = bytearray()

    def makefile(self, mode, *args):
        return self.request

    def sendall(self, data):
        self.response.extend(data)


def request(app, target):
    connection = MemoryConnection(f"GET {target} HTTP/1.0\r\nHost: localhost\r\n\r\n".encode())
    make_handler(app)(connection, ("127.0.0.1", 1234), object())
    header, body = bytes(connection.response).split(b"\r\n\r\n", 1)
    return header.decode(), body


def test_http_routes_json_validation_and_static_assets():
    app = SimulationApp()
    header, body = request(app, "/api/catalog")
    assert "200 OK" in header
    assert len(json.loads(body)["controllers"]) == 3
    header, body = request(app, "/api/simulate?scenario=restricted&seed=42")
    assert "200 OK" in header
    assert len(json.loads(body)["frames"]) == 289
    data = json.loads(body)
    metrics, records = rollout(ClassroomEnv(), RuleController('timetable'), 42)
    assert data['metric_definitions']['version'] == 'comfort_v2'
    assert data['metrics']['arrival_discomfort_pct'] == metrics['arrival_discomfort_pct']
    assert data['metrics']['early_class_occupied_minutes'] == metrics['early_class_occupied_minutes']
    start = next(i for i, r in enumerate(records) if r['class_start'].any())
    assert data['frames'][start + 1]['arrival_comfort_proxy'] == records[start]['arrival_comfort_proxy'].tolist()
    header, body = request(app, "/api/simulate?seed=-1")
    assert "400 Bad Request" in header
    assert "Seed" in json.loads(body)["error"]
    header, body = request(app, "/api/simulate?controller=unknown")
    assert "400 Bad Request" in header
    for asset, mime in [("/", "text/html"), ("/app.js", "text/javascript"), ("/styles.css", "text/css"), ("/favicon.svg", "image/svg+xml"), ("/UbuntuSans.ttf", "font/ttf"), ("/UbuntuSansMono.ttf", "font/ttf"), ("/FONT-LICENSE.txt", "text/plain")]:
        header, body = request(app, asset)
        assert "200 OK" in header and mime in header
        assert body
    for forbidden in ["/../config.py", "/%2e%2e/config.py", "/outputs/q_seed0/agent.json"]:
        header, body = request(app, forbidden)
        assert "404 Not Found" in header


def test_teaching_transition_exposes_actual_state_mask_and_heat_balance():
    config = Config()
    app = SimulationApp(config)
    data = app.simulate('timetable', 'restricted', 42)
    env = ClassroomEnv(config)
    observation, info = env.reset(seed=42)
    for frame in data['frames'][1:]:
        np.testing.assert_array_equal(frame['observation'], observation)
        np.testing.assert_array_equal(frame['action_mask'], info['action_mask'])
        np.testing.assert_array_equal(frame['temperature_before'], env.temperature)
        assert frame['q_values'] is None  # A baseline must not be given invented learned values.
        heat = np.array(list(frame['heat_terms_kw'].values())).sum(axis=0)
        expected = np.array(frame['temperature_before']) + heat * config.step_minutes / 60 / np.array(config.thermal_capacity_kwh_per_c)
        np.testing.assert_allclose(expected, frame['temperature'])
        observation, _, _, _, info = env.step(frame['action'])


def test_q_values_and_training_history_are_read_from_saved_run(tmp_path):
    folder = tmp_path / 'q_run'
    folder.mkdir()
    Config().save(folder / 'config.json')
    agent = QLearning()
    env = ClassroomEnv()
    obs, _ = env.reset(seed=42)
    from thermorl.q_learning import discretize
    agent.table[discretize(obs)] = np.arange(16, dtype=float)
    agent.save(folder / 'agent.json')
    (folder / 'metadata.json').write_text(json.dumps({
        'algorithm': 'q_learning', 'checkpoint': 'agent.json', 'episodes': 1, 'seed': 0,
    }))
    (folder / 'training.csv').write_text('episode,return,epsilon,mean_loss,visited_states\n1,-7,1,0.5,12\n')
    app = SimulationApp(model_paths=[folder])
    history = app.catalog()['controllers'][-1]['training_history']
    assert history == [{'episode': 1.0, 'return': -7.0, 'epsilon': 1.0, 'mean_loss': 0.5, 'visited_states': 12.0}]
    frame = app.simulate('model_0', 'restricted', 42)['frames'][1]
    assert frame['q_values'] == list(range(16))
    assert frame['state_seen'] is True
    assert frame['action'] == 15
    assert len(app.controllers['model_0'].table) == 1  # GUI never learns.
    (folder / 'training.csv').write_text('bad,data\n1,2\n')
    assert SimulationApp.read_history(folder / 'training.csv') == []
