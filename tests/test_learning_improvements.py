import csv
import json
import numpy as np
import pytest
from thermorl.config import Config
from thermorl.env import ClassroomEnv
from thermorl.experiment import validate_agent
from thermorl.q_learning import QLearning


def test_compact_table_reuses_weather_states_and_respects_occupancy_and_ablation(tmp_path):
    obs, _ = ClassroomEnv().reset(seed=3)
    agent = QLearning(state_mode='compact_v1')
    changed = obs.copy()
    changed[8:12] = [0.9, 1.0, 0, 1]
    assert agent.state_key(obs) == agent.state_key(changed)
    changed[2] = 0.5
    assert agent.state_key(obs) != agent.state_key(changed)
    no_timetable = QLearning(state_mode='compact_v1', use_timetable=False)
    changed = obs.copy()
    changed[4:8] = [0, 0, 1, 1]
    assert no_timetable.state_key(obs) == no_timetable.state_key(changed)
    agent.table[agent.state_key(obs)] = np.arange(16.)
    agent.save(tmp_path / 'q.json')
    loaded = QLearning.load(tmp_path / 'q.json')
    assert loaded.state_mode == 'compact_v1'
    np.testing.assert_array_equal(loaded.table[loaded.state_key(obs)], np.arange(16.))


def test_validation_uses_fixed_held_out_days_without_learning_or_rng_changes():
    agent = QLearning(state_mode='compact_v1')
    rng_before = json.dumps(agent.rng.bit_generator.state)
    result = validate_agent(agent, Config(), 1)
    assert result['unseen_state_pct'] == 100
    assert agent.table == {} and agent.visits == 0
    assert json.dumps(agent.rng.bit_generator.state) == rng_before
    assert result == validate_agent(agent, Config(), 1)


def test_dqn_comfort_features_and_scaled_reward_survive_checkpoint(tmp_path):
    torch = pytest.importorskip('torch')
    from thermorl.dqn import DQN
    agent = DQN(feature_mode='comfort_v1', reward_scale=10, batch_size=1, warmup=1)
    obs = np.zeros(17, dtype=np.float32)
    obs[:2] = np.array([20, 30]) / 50
    original = obs.copy()
    f = agent.features(obs)
    np.testing.assert_allclose(f[:2], [-0.9, 1.1], atol=1e-6)
    np.testing.assert_allclose(f[17:], [0.6, 0, 0, 0.8], atol=1e-6)
    assert agent.features(np.stack([obs, obs])).shape == (2, 21)
    np.testing.assert_array_equal(obs, original)
    for p in agent.online.parameters():
        p.data.zero_()
    loss = agent.learn(obs, 0, -20, obs, True, np.zeros(16, dtype=bool))
    assert loss == pytest.approx(1.5)  # Huber(-20/10), terminal bootstrap zero.
    agent.save(tmp_path / 'agent.pt')
    loaded = DQN.load(tmp_path / 'agent.pt')
    assert loaded.feature_mode == 'comfort_v1' and loaded.reward_scale == 10
    torch.testing.assert_close(loaded.online(torch.from_numpy(f)), agent.online(torch.from_numpy(f)))


def test_comfort_diagnostics_partition_occupied_room_time():
    from thermorl.baselines import RuleController
    from thermorl.experiment import rollout
    metrics, _ = rollout(ClassroomEnv(), RuleController('thermostat'), 42)
    assert metrics['comfort_pct'] + metrics['overheating_pct'] + metrics['overcooling_pct'] == pytest.approx(100)
    assert metrics['overheating_pct'] + metrics['overcooling_pct'] == pytest.approx(metrics['comfort_violation_pct'])


@pytest.mark.parametrize('algorithm', ['dqn', 'q_learning'])
def test_gui_loads_new_representation_and_actual_validation_history(tmp_path, algorithm):
    from thermorl.gui import SimulationApp
    Config().save(tmp_path / 'config.json')
    obs, _ = ClassroomEnv().reset(seed=42)
    if algorithm == 'dqn':
        torch = pytest.importorskip('torch')
        from thermorl.dqn import DQN
        agent = DQN(feature_mode='comfort_v1', reward_scale=10)
        checkpoint = 'agent.pt'
        with torch.no_grad():
            expected = agent.online(torch.from_numpy(agent.features(obs))).numpy()
    else:
        agent = QLearning(state_mode='compact_v1')
        checkpoint = 'agent.json'
        expected = np.arange(16, dtype=float)
        agent.table[agent.state_key(obs)] = expected.copy()
    agent.save(tmp_path / checkpoint)
    (tmp_path / 'metadata.json').write_text(json.dumps({'algorithm': algorithm, 'episodes': 50,
        'selected_episode': 25, 'checkpoint': checkpoint, 'training_complete': True,
        'initialization': {'source_selected_episode': 350, 'replay_optimizer_reset': True} if algorithm == 'dqn' else None}))
    (tmp_path / 'validation.csv').write_text('episode,mean_return,mean_energy_kwh,mean_comfort_pct,mean_overheating_pct,mean_overcooling_pct\n25,-3,20,90,7,3\n')
    app = SimulationApp(model_paths=[tmp_path])
    detail = app.catalog()['controllers'][-1]
    assert detail['selected_episode'] == 25
    if algorithm == 'dqn':
        assert detail['initialization']['source_selected_episode'] == 350
    assert detail['validation_history'][0]['mean_comfort_pct'] == 90
    frame = app.simulate('model_0', 'restricted', 42)['frames'][1]
    np.testing.assert_allclose(frame['q_values'], expected, rtol=1e-6)
    if algorithm == 'q_learning':
        assert frame['state_seen'] is True
        assert len(app.controllers['model_0'].table) == 1
