"""Contracts that catch energy misattribution, split leakage and unfair selection."""
import copy
import json

import numpy as np
import pytest


def test_energy_diagnostics_separate_precooling_from_other_empty_use():
    from thermorl.study import diagnose_records
    from thermorl.config import Config
    # 3 kW AC for 5 minutes = .25 kWh; fan = .01 kWh.
    records = []
    for minute, action, occupancy, starts in [
        (0, 8, [0, 0], [False, False]),
        (35, 9, [0, 0], [False, False]),
        (65, 12, [30, 0], [True, False]),
    ]:
        records.append(dict(minute=minute, action=action, occupancy=np.array(occupancy),
            class_start=np.array(starts), temperature=np.array([25., 25.]),
            effective_temperature=np.array([25., 25.]), violation=np.zeros(2),
            overheated=np.zeros(2, bool), overcooled=np.zeros(2, bool),
            arrival_violation=np.zeros(2), energy_kwh=[.25, .26, .26][len(records)],
            power_kw=3.12, cap_kw=6.5, switches=0))
    result = diagnose_records(records, Config())
    assert result['occupied_ac_kwh'] == pytest.approx(.25)
    assert result['empty_soon_ac_kwh'] == pytest.approx(.25)
    assert result['empty_other_ac_kwh'] == pytest.approx(.25)
    assert result['fan_kwh'] == pytest.approx(.02)
    assert result['room_0_energy_kwh'] == pytest.approx(.76)
    assert result['room_1_energy_kwh'] == pytest.approx(.01)
    assert result['room_0_comfort_pct'] == 100
    assert result['room_1_comfort_pct'] is None


def test_study_score_pools_unequal_denominators_instead_of_daily_percentages():
    from thermorl.study import score_days
    rows = [dict(energy_kwh=10, **counts(1, 1, 10, 10, 10, 10)),
            dict(energy_kwh=20, **counts(9, 0, 90, 0, 90, 90))]
    score = score_days(rows)
    assert score['mean_energy_kwh'] == 15
    assert score['mean_arrival_discomfort_pct'] == 10
    assert score['mean_class_start_violation_pct'] == 10
    assert score['mean_comfort_pct'] == 100


def counts(arrivals, bad, early, early_bad, occupied, comfortable):
    return dict(class_arrivals=arrivals, arrival_measurements=arrivals,
        uncomfortable_arrivals=bad, early_class_occupied_minutes=early,
        early_class_uncomfortable_minutes=early_bad, occupied_room_minutes=occupied,
        comfortable_room_minutes=comfortable)


def test_protocol_rejects_overlap_and_previously_consumed_days():
    from thermorl.study import default_protocol, validate_study_protocol
    protocol = default_protocol()
    validate_study_protocol(protocol)
    overlap = copy.deepcopy(protocol)
    overlap['partitions']['confirmation']['seed_offset'] = 1_100_010
    with pytest.raises(ValueError, match='overlap'):
        validate_study_protocol(overlap)
    with pytest.raises(ValueError, match='previously used'):
        validate_study_protocol(protocol, consumed_seeds={1_200_005})
    invalid = copy.deepcopy(protocol)
    invalid['recipes'][1]['algorithm'] = 'supervised'
    with pytest.raises(ValueError, match='standard DQN'):
        validate_study_protocol(invalid)


def test_confirmation_rejects_hidden_scenario_regression_and_requires_lower_energy():
    from thermorl.study import confirmation_decision
    def score(comfort=80, energy=30, arrival=20, early=20):
        return dict(mean_comfort_pct=comfort, mean_energy_kwh=energy,
            mean_arrival_discomfort_pct=arrival, mean_class_start_violation_pct=early)
    current = {'overall': score(), 'scenarios': {'normal': score(), 'restricted': score()}}
    good = {'overall': score(energy=29),
            'scenarios': {'normal': score(energy=29), 'restricted': score(energy=29)}}
    assert confirmation_decision(good, current)['eligible']
    bad = copy.deepcopy(good)
    bad['scenarios']['restricted']['mean_arrival_discomfort_pct'] = 21
    assert not confirmation_decision(bad, current)['eligible']
    assert not confirmation_decision(current, current)['eligible']
    missing = copy.deepcopy(good); missing['scenarios'].pop('restricted')
    assert not confirmation_decision(missing, current)['eligible']


def test_paired_intervals_keep_pairs_and_refuse_missing_or_duplicate_days():
    from thermorl.study import paired_intervals
    rows = []
    for seed in range(4):
        for controller, energy in [('reference', 20 + seed), ('candidate', 18 + seed)]:
            rows.append(dict(controller=controller, scenario='normal', day_seed=seed,
                energy_kwh=energy, **counts(8, 2, 120, 30, 600, 450)))
    result = paired_intervals(rows, 'candidate', 'reference', samples=30)
    assert result['mean_energy_kwh']['difference'] == -2
    assert result['mean_energy_kwh']['ci95'] == [-2, -2]
    assert result['mean_comfort_pct']['ci95'] == [0, 0]
    with pytest.raises(ValueError, match='paired'):
        paired_intervals(rows[:-1], 'candidate', 'reference', samples=30)
    with pytest.raises(ValueError, match='duplicate'):
        paired_intervals(rows + [rows[0]], 'candidate', 'reference', samples=30)


def test_nonregression_checkpoint_rank_rejects_arrival_tradeoff():
    from thermorl.experiment import checkpoint_rank
    old = dict(mean_comfort_pct=75, mean_energy_kwh=50, mean_return=-25,
        mean_arrival_discomfort_pct=20, mean_class_start_violation_pct=20)
    good = dict(old, mean_energy_kwh=49)
    bad = dict(old, mean_comfort_pct=90, mean_energy_kwh=20, mean_arrival_discomfort_pct=30)
    assert checkpoint_rank(good, 'nonregression', 90, reference=old) > checkpoint_rank(
        bad, 'nonregression', 90, reference=old)


def test_dqn_replay_diagnostics_are_read_only_and_terminal_targets_finite():
    from thermorl.dqn import DQN
    agent = DQN(batch_size=1, warmup=1, feature_mode='comfort_v1')
    obs = np.zeros(17, np.float32); obs[2] = .5; obs[4] = 20 / 240; obs[6] = .5
    agent.learn(obs, 0, -2, obs, True, np.zeros(16, bool), scenario='restricted')
    before = json.dumps(agent.rng.bit_generator.state)
    diagnostics = agent.training_diagnostics()
    assert diagnostics['replay_occupied_transition_pct'] == 100
    assert diagnostics['replay_arrival_soon_pct'] == 100
    assert diagnostics['replay_restricted_pct'] == 100
    assert diagnostics['td_target_mean'] == pytest.approx(-2)
    assert np.isfinite(diagnostics['q_abs_max'])
    assert before == json.dumps(agent.rng.bit_generator.state)


def test_cli_uses_custom_pooled_validation_and_records_reference(tmp_path):
    from thermorl.experiment import train_main
    out = tmp_path / 'run'
    train_main(['--algo', 'dqn', '--episodes', '1', '--validation-days', '1',
        '--validation-seed-offset', '1100000', '--validation-aggregation', 'pooled',
        '--selection', 'nonregression', '--reference-model', 'models/dqn', '--output', str(out)])
    metadata = json.loads((out / 'metadata.json').read_text())
    assert metadata['validation']['seeds'] == [1_100_000]
    assert metadata['validation']['aggregation'] == 'pooled'
    assert len(metadata['reference_model']['checkpoint_sha256']) == 64
    assert 'mean_arrival_discomfort_pct' in metadata['reference_validation']
    assert isinstance(metadata['selection_nonregression_met'], bool)
    with pytest.raises(SystemExit):
        train_main(['--algo', 'dqn', '--episodes', '1', '--validation-seed-offset', '42',
            '--output', str(tmp_path / 'bad')])


def test_study_freezes_reference_and_rejects_tampering_or_repeat_initialization(tmp_path):
    from scripts.dqn_study import initialize, load_study
    root = tmp_path / 'study'
    initialize(root, consumed_roots=[])
    protocol = load_study(root)
    assert protocol['reference_checkpoint_sha256']
    assert (root / 'reference/agent.pt').read_bytes() == __import__('pathlib').Path('models/dqn/agent.pt').read_bytes()
    with pytest.raises(ValueError, match='exists'):
        initialize(root, consumed_roots=[])
    (root / 'reference/agent.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        load_study(root)


def test_confirmation_cannot_run_before_frozen_selection(tmp_path):
    from scripts.dqn_study import initialize, assess
    root = tmp_path / 'study'
    initialize(root, consumed_roots=[])
    with pytest.raises(ValueError, match='selection'):
        assess(root, 'confirmation')
    with pytest.raises(ValueError, match='confirmation'):
        assess(root, 'final')


def test_gui_separates_study_metadata_from_demo_and_labels_baseline_role(tmp_path):
    from thermorl.gui import SimulationApp
    summary = tmp_path / 'summary.json'
    summary.write_text(json.dumps({'status': 'No qualifying candidate', 'partition': 'tuning',
        'scenario_count': 4, 'days_per_scenario': 20, 'total_days': 80,
        'reference_checkpoint': 350, 'training_seeds': [0, 1, 2], 'rows': []}))
    catalog = SimulationApp(study_path=summary).catalog()
    assert catalog['study']['total_days'] == 80
    assert catalog['study']['partition'] == 'tuning'
    assert catalog['controllers'][0]['role'] == 'comparison_only'
    assert SimulationApp().catalog()['study'] is None


def completed_fixture(root):
    """Real file boundary fixture; no training/prediction behavior is mocked."""
    import shutil
    from scripts.dqn_study import initialize, load_study, run_specs
    initialize(root, consumed_roots=[])
    protocol = load_study(root)
    reference = dict(mean_comfort_pct=75., mean_energy_kwh=50., mean_return=-25.,
        mean_arrival_discomfort_pct=20., mean_class_start_violation_pct=20.)
    (root / 'diagnosis').mkdir()
    (root / 'diagnosis/scores.json').write_text(json.dumps({'reference': {'overall': reference}}))
    for recipe, seed, folder in run_specs(root, protocol):
        shutil.copytree(root / 'reference', folder)
        metadata = json.loads((folder / 'metadata.json').read_text())
        metadata.update(training_complete=True, seed=seed, selection_mode='nonregression',
            initialization=None, reference_validation=reference, selected_validation=dict(reference),
            reference_model={'checkpoint_sha256': protocol['reference_checkpoint_sha256']},
            epsilon=dict(start=1., end=.05, decay_fraction=recipe['epsilon_decay']),
            validation=dict(seeds=list(range(1_100_000, 1_100_020)),
                scenarios=protocol['partitions']['tuning']['scenarios'], aggregation='pooled'))
        metadata['agent_hyperparameters'].update(learning_rate=recipe['learning_rate'], train_frequency=1)
        (folder / 'metadata.json').write_text(json.dumps(metadata))


def test_no_qualifying_candidate_keeps_confirmation_and_final_unused(tmp_path):
    from scripts.dqn_study import select, assess
    root = tmp_path / 'study'
    completed_fixture(root)
    select(root)
    assert json.loads((root / 'selection.json').read_text())['selected'] is None
    assess(root, 'confirmation'); assess(root, 'final')
    assert not (root / 'confirmation-started.json').exists()
    assert not (root / 'final-started.json').exists()


def test_selection_rejects_reference_scores_from_different_days(tmp_path):
    from scripts.dqn_study import select
    root = tmp_path / 'study'
    completed_fixture(root)
    path = root / 'runs/control_seed0/metadata.json'
    metadata = json.loads(path.read_text())
    metadata['reference_validation']['mean_comfort_pct'] = 70
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='Reference scores'):
        select(root)


def test_final_requires_confirmation_of_the_same_frozen_checkpoint(tmp_path):
    from scripts.dqn_study import verify_confirmation
    root = tmp_path
    (root / 'confirmation').mkdir()
    (root / 'confirmation/decision.json').write_text(json.dumps(
        {'eligible': True, 'selected_checkpoint_sha256': 'different', 'protocol_sha256': 'protocol'}))
    selection = {'selected': {'checkpoint_sha256': 'frozen'}, 'protocol_sha256': 'protocol'}
    with pytest.raises(ValueError, match='same frozen'):
        verify_confirmation(root, selection)


def test_failure_trace_exposes_precooling_lead_and_joint_action_duration():
    from thermorl.study import failure_trace
    from thermorl.config import Config
    rows = [dict(minute=i * 5, action=action, power_kw=3, cap_kw=6.5, switches=0,
        temperature_before=np.array([28., 27.]), temperature=np.array([27., 26.]),
        effective_temperature=np.array([27., 26.]), arrival_violation=np.array([2., 1.]),
        occupancy=np.array([30 if i >= 2 else 0, 0]), class_start=np.array([i == 2, False]))
        for i, action in enumerate([8, 8, 0])]
    result = failure_trace(rows, Config())
    assert result[0]['room_0_next_class_minutes'] == 10
    assert result[1]['room_0_next_class_minutes'] == 5
    assert result[2]['room_0_next_class_minutes'] == 0
    assert result[0]['joint_action_run_minutes'] == 10
    assert result[1]['joint_action_run_minutes'] == 10
    assert result[0]['room_1_next_class_minutes'] is None


def test_report_marks_interrupted_assessment_as_potentially_consumed(tmp_path):
    from scripts.dqn_study import assessment_status
    (tmp_path / 'confirmation-started.json').write_text('{}')
    assert 'potentially consumed' in assessment_status(tmp_path, 'confirmation')
    assert 'unused' in assessment_status(tmp_path, 'final')
