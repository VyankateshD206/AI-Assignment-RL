import json
import numpy as np
import pytest
from thermorl.env import ClassroomEnv
from thermorl.q_learning import QLearning


def test_refined_state_separates_midband_and_arrival_lead_and_supports_ablation():
    obs, _ = ClassroomEnv().reset(seed=42)
    obs[:2] = [23.5 / 50, 25.5 / 50]
    obs[4:8] = [20 / 240, 20 / 240, 0.5, 0.5]
    agent = QLearning(state_mode='comfort_v2')
    key = agent.state_key(obs)
    changed = obs.copy(); changed[0] = 25.5 / 50
    assert key != agent.state_key(changed)
    changed = obs.copy(); changed[4] = 45 / 240
    assert key != agent.state_key(changed)
    no_schedule = QLearning(state_mode='comfort_v2', use_timetable=False)
    assert no_schedule.state_key(obs) == no_schedule.state_key(changed)


def test_counts_are_per_action_decay_updates_and_survive_roundtrip(tmp_path):
    obs, _ = ClassroomEnv().reset(seed=42)
    agent = QLearning(state_mode='comfort_v2', learning_rate=1, visit_decay=1, count_exploration=True)
    agent.learn(obs, 0, -2, obs, True, np.zeros(16, bool))
    agent.learn(obs, 0, -4, obs, True, np.zeros(16, bool))
    agent.learn(obs, 1, -6, obs, True, np.zeros(16, bool))
    key = agent.state_key(obs)
    # alpha = initial_lr / sqrt(1 + previous_count / visit_decay).
    assert agent.table[key][0] == pytest.approx(-2 - 2 / np.sqrt(2))
    assert agent.table[key][1] == -6
    np.testing.assert_array_equal(agent.action_visits[key][:3], [2, 1, 0])
    agent.save(tmp_path / 'q.json')
    loaded = QLearning.load(tmp_path / 'q.json')
    assert loaded.visit_decay == 1 and loaded.count_exploration
    np.testing.assert_array_equal(loaded.action_visits[key], agent.action_visits[key])


def test_exploration_visits_feasible_untried_actions_and_greedy_is_pure():
    obs, _ = ClassroomEnv().reset(seed=42)
    agent = QLearning(state_mode='comfort_v2', count_exploration=True)
    mask = np.zeros(16, bool); mask[[3, 8]] = True
    first = agent.act(obs, mask, epsilon=1)
    agent.learn(obs, first, -1, obs, True, mask)
    second = agent.act(obs, mask, epsilon=1)
    assert {first, second} == {3, 8}
    snapshot = json.dumps(agent.rng.bit_generator.state)
    counts = {k: v.copy() for k, v in agent.action_visits.items()}
    agent.act(obs, mask)
    assert snapshot == json.dumps(agent.rng.bit_generator.state)
    for key in counts:
        np.testing.assert_array_equal(counts[key], agent.action_visits[key])


def test_dqn_records_every_transition_but_updates_at_frequency_and_roundtrips(tmp_path):
    from thermorl.dqn import DQN
    agent = DQN(feature_mode='comfort_v1', batch_size=1, warmup=1, train_frequency=4, target_interval=7)
    obs = np.zeros(17, np.float32)
    for _ in range(3):
        assert agent.learn(obs, 0, -1, obs, True, np.zeros(16, bool)) is None
    assert agent.size == 3 and agent.updates == 0
    assert agent.learn(obs, 0, -1, obs, True, np.zeros(16, bool)) is not None
    assert agent.size == 4 and agent.updates == 1
    agent.save(tmp_path / 'dqn.pt')
    loaded = DQN.load(tmp_path / 'dqn.pt')
    assert loaded.train_frequency == 4 and loaded.target_interval == 7 and loaded.batch_size == 1


def test_comfort_selection_prefers_feasible_low_energy_and_does_not_hide_missed_floor():
    from thermorl.experiment import checkpoint_rank
    def metrics(comfort, energy):
        return {'mean_comfort_pct': comfort, 'mean_energy_kwh': energy, 'mean_return': -energy}
    assert checkpoint_rank(metrics(91, 20), 'comfort_energy', 90) > checkpoint_rank(metrics(99, 50), 'comfort_energy', 90)
    assert checkpoint_rank(metrics(91, 50), 'comfort_energy', 90) > checkpoint_rank(metrics(89, 0), 'comfort_energy', 90)
    assert checkpoint_rank(metrics(80, 30), 'comfort_energy', 90) > checkpoint_rank(metrics(75, 0), 'comfort_energy', 90)
    assert checkpoint_rank(metrics(50, 10), 'return', 90) > checkpoint_rank(metrics(90, 20), 'return', 90)


def test_cli_new_controls_are_saved_and_frozen_validation_preserves_counts(tmp_path):
    from thermorl.experiment import train_main, validate_agent
    from thermorl.config import Config
    train_main(['--algo','q_learning','--episodes','2','--q-state','comfort_v2',
                '--q-visit-decay','100','--q-count-exploration','--epsilon-start','0.10','--epsilon-end','0.02',
                '--selection','comfort_energy','--validation-days','1','--output',str(tmp_path / 'run')])
    metadata = json.loads((tmp_path / 'run/metadata.json').read_text())
    assert metadata['epsilon']['end'] == 0.02
    assert metadata['epsilon']['start'] == 0.10
    import csv
    with (tmp_path / 'run/training.csv').open() as stream:
        training = list(csv.DictReader(stream))
    assert float(training[0]['epsilon']) == pytest.approx(0.10)
    assert float(training[-1]['epsilon']) == pytest.approx(0.02)
    assert metadata['selection_mode'] == 'comfort_energy'
    assert metadata['comfort_floor'] == 90
    assert metadata['agent_hyperparameters']['visit_decay'] == 100
    assert 'stored_action_entries_updated_pct' in metadata['selected_validation']
    assert 'selection_comfort_floor_met' in metadata
    from thermorl.gui import SimulationApp
    detail = SimulationApp(model_paths=[tmp_path / 'run']).catalog()['controllers'][-1]
    assert detail['selection_mode'] == 'comfort_energy'
    assert detail['selection_comfort_floor_met'] == metadata['selection_comfort_floor_met']
    agent = QLearning.load(tmp_path / 'run/last_agent.json')
    before = {k: v.copy() for k, v in agent.action_visits.items()}
    validation = validate_agent(agent, Config(), 1)
    assert validation['stored_action_entries_updated_pct'] > 0
    assert before.keys() == agent.action_visits.keys()
    for key in before:
        np.testing.assert_array_equal(before[key], agent.action_visits[key])


def test_evaluation_can_use_fresh_test_seeds_without_overlapping_training(tmp_path):
    from thermorl.experiment import evaluate_main
    evaluate_main(['--scenarios','normal','--days','1','--seed-offset','3000000','--output',str(tmp_path / 'test')])
    data = json.loads((tmp_path / 'test/metadata.json').read_text())
    assert data['day_seeds'] == [3_000_000]
    with pytest.raises(SystemExit):
        evaluate_main(['--days','1','--seed-offset','100','--output',str(tmp_path / 'bad')])


def test_promotion_requires_validation_pareto_improvement():
    from scripts.optimize_models import improves_validation
    def values(comfort, energy, arrival=20, early=10):
        return dict(mean_comfort_pct=comfort, mean_energy_kwh=energy,
                    mean_arrival_discomfort_pct=arrival, mean_class_start_violation_pct=early)
    old = values(75, 50)
    assert improves_validation(values(80, 40), old)
    assert improves_validation(values(75, 40), old)
    assert not improves_validation(values(74, 10), old)
    assert not improves_validation(values(90, 60), old)
    assert not improves_validation(values(80, 40, arrival=21), old)
    assert not improves_validation(values(80, 40, early=11), old)
    assert not improves_validation(values(80, 40, arrival=None), old)
    assert not improves_validation(values(80, 40, early=float('nan')), old)
    incomplete = dict(old); incomplete.pop('mean_arrival_discomfort_pct')
    assert not improves_validation(values(80, 40), incomplete)
    assert not improves_validation(old, old)


def test_old_checkpoints_without_new_settings_keep_predictions(tmp_path):
    import torch
    from thermorl.dqn import DQN
    obs, info = ClassroomEnv().reset(seed=42)
    q = QLearning(state_mode='compact_v1')
    q.table[q.state_key(obs)] = np.arange(16.)
    q.save(tmp_path / 'q.json')
    payload = json.loads((tmp_path / 'q.json').read_text())
    for key in ['action_visits', 'visit_decay', 'count_exploration']:
        payload.pop(key)
    (tmp_path / 'q.json').write_text(json.dumps(payload))
    old_q = QLearning.load(tmp_path / 'q.json')
    assert old_q.act(obs, info['action_mask']) == q.act(obs, info['action_mask'])
    assert old_q.action_visits == {} and old_q.visit_decay == 0
    dqn = DQN(feature_mode='comfort_v1', reward_scale=10)
    dqn.save(tmp_path / 'dqn.pt')
    payload = torch.load(tmp_path / 'dqn.pt', weights_only=True)
    for key in ['batch_size','warmup','replay_capacity','target_interval','train_frequency']:
        payload.pop(key)
    torch.save(payload, tmp_path / 'dqn.pt')
    old_dqn = DQN.load(tmp_path / 'dqn.pt')
    torch.testing.assert_close(old_dqn.online(torch.from_numpy(old_dqn.features(obs))),
                               dqn.online(torch.from_numpy(dqn.features(obs))))


@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_dqn_rejects_invalid_update_frequency(value):
    from thermorl.dqn import DQN
    with pytest.raises(ValueError):
        DQN(train_frequency=value)


def test_promotion_rejects_mismatched_validation_days_and_candidate_targets():
    from scripts.optimize_models import validate_protocol
    import copy
    old = {'validation': {'scenarios': ['normal'], 'seeds': [1000000], 'days_per_scenario': 1}}
    new = dict(copy.deepcopy(old), comfort_floor=90)
    validate_protocol([new, copy.deepcopy(new)], old)
    different = copy.deepcopy(new); different['validation']['seeds'] = [1000001]
    with pytest.raises(ValueError, match='validation'):
        validate_protocol([new, different], old)
    different = dict(new, comfort_floor=80)
    with pytest.raises(ValueError, match='comfort'):
        validate_protocol([new, different], old)


def test_dqn_warm_start_copies_weights_with_fresh_replay_and_optimizer(tmp_path):
    import torch
    from thermorl.dqn import DQN
    from thermorl.config import Config
    from thermorl.experiment import initialize_dqn
    source = DQN(feature_mode='comfort_v1', reward_scale=10)
    source.save(tmp_path / 'agent.pt')
    Config().save(tmp_path / 'config.json')
    (tmp_path / 'metadata.json').write_text(json.dumps({'algorithm':'dqn','checkpoint':'agent.pt','selected_episode':350}))
    target = DQN(seed=2, feature_mode='comfort_v1', reward_scale=10, learning_rate=0.0001)
    initialization = initialize_dqn(target, tmp_path, Config())
    for a, b in zip(source.online.parameters(), target.online.parameters()):
        torch.testing.assert_close(a, b)
    assert target.size == target.updates == 0 and not target.optimizer.state
    assert target.learning_rate == 0.0001
    assert initialization['source_selected_episode'] == 350
    incompatible = DQN(feature_mode='legacy')
    with pytest.raises(ValueError, match='representation'):
        initialize_dqn(incompatible, tmp_path, Config())


def test_promotion_archives_original_weights_and_selection_manifest(tmp_path, monkeypatch):
    from pathlib import Path
    import sys
    from thermorl.config import Config
    from scripts.optimize_models import main
    monkeypatch.chdir(tmp_path)
    protocol = {'scenarios':['normal'], 'seeds':[1000000], 'days_per_scenario':1}
    runs = []
    for algorithm in ['dqn', 'q_learning']:
        checkpoint = 'agent.pt' if algorithm == 'dqn' else 'agent.json'
        bundle = Path('models') / algorithm; bundle.mkdir(parents=True)
        Config().save(bundle / 'config.json')
        (bundle / checkpoint).write_bytes(b'old weights')
        (bundle / 'metadata.json').write_text(json.dumps({'validation':protocol,
            'selected_validation':{'mean_comfort_pct':75, 'mean_energy_kwh':50,
                'mean_arrival_discomfort_pct':40, 'mean_class_start_violation_pct':30}}))
        for seed in range(3):
            run = Path(f'{algorithm}_seed{seed}'); run.mkdir(); runs.append(str(run))
            Config().save(run / 'config.json')
            for name in [checkpoint,'training.csv','validation.csv','training_curve.png']:
                (run / name).write_bytes(b'new data')
            (run / 'metadata.json').write_text(json.dumps({'algorithm':algorithm, 'checkpoint':checkpoint,
                'seed':seed, 'training_complete':True, 'selection_mode':'comfort_energy', 'comfort_floor':90,
                'selected_episode':100, 'validation':protocol,
                'selected_validation':{'mean_comfort_pct':90,'mean_energy_kwh':20,'mean_return':-10,
                    'mean_arrival_discomfort_pct':30,'mean_class_start_violation_pct':20}}))
    original = '{"dqn": {"source_run": "old_dqn"}, "q_learning": {"source_run": "old_q"}}\n'
    Path('models/selection.json').write_text(original)
    monkeypatch.setattr(sys, 'argv', ['optimize_models','--runs',*runs,'--output','promotion.json','--promote'])
    main()
    assert Path('models/archive/dqn_before_optimization/agent.pt').read_bytes() == b'old weights'
    assert Path('models/dqn/agent.pt').read_bytes() == b'new data'
    assert Path('models/archive/selection_before_optimization.json').read_text() == original
    assert json.loads(Path('promotion.json').read_text())['algorithms']['dqn']['promoted']
