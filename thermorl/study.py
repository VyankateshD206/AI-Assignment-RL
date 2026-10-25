"""Diagnostics and paired assessment of RL agents; never chooses cooling actions."""
import math

import numpy as np

from .env import ACTIONS
from .metrics import pooled_comfort_metrics, summarize
from .scenarios import SCENARIOS

QUALITY = {'mean_comfort_pct': 1, 'mean_energy_kwh': -1,
           'mean_arrival_discomfort_pct': -1, 'mean_class_start_violation_pct': -1}


def nonregression(candidate, reference, require_energy_gain=True):
    """Fail closed on missing evidence; comfort cannot be exchanged for energy."""
    for key, direction in QUALITY.items():
        values = candidate.get(key), reference.get(key)
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            return False
        if direction * (values[0] - values[1]) < -1e-9:
            return False
    return not require_energy_gain or candidate['mean_energy_kwh'] < reference['mean_energy_kwh'] - 1e-9


def score_days(rows):
    if not rows:
        raise ValueError('Cannot score empty days')
    pooled = pooled_comfort_metrics(rows)
    return {'mean_energy_kwh': float(np.mean([float(r['energy_kwh']) for r in rows])),
            **{'mean_' + k: v for k, v in pooled.items()}}


def grouped_scores(rows):
    return {'overall': score_days(rows), 'scenarios': {
        scenario: score_days([r for r in rows if r['scenario'] == scenario])
        for scenario in sorted({r['scenario'] for r in rows})}}


def confirmation_decision(candidate, reference):
    reasons = []
    if not nonregression(candidate['overall'], reference['overall']):
        reasons.append('Aggregate comfort worsens, energy does not decrease, or evidence is incomplete')
    if set(candidate['scenarios']) != set(reference['scenarios']):
        reasons.append('Scenario evidence does not match')
    for scenario, old in reference['scenarios'].items():
        if not nonregression(candidate['scenarios'].get(scenario, {}), old, require_energy_gain=False):
            reasons.append(f'{scenario}: at least one quantity worsens or evidence is incomplete')
    return {'eligible': not reasons, 'reasons': reasons}


def diagnose_records(records, config):
    """Partition all AC energy by occupancy/arrival lead; fans counted separately."""
    if not records:
        raise ValueError('A diagnostic trace cannot be empty')
    result = dict(occupied_ac_kwh=0., empty_soon_ac_kwh=0., empty_other_ac_kwh=0., fan_kwh=0.)
    room_energy = np.zeros(2)
    for i, record in enumerate(records):
        for room, (ac, fan) in enumerate(ACTIONS[record['action']]):
            ac_energy = float(ac * config.ac_electric_kw * config.step_minutes / 60)
            fan_energy = float(fan * config.fan_electric_kw * config.step_minutes / 60)
            room_energy[room] += ac_energy + fan_energy
            result['fan_kwh'] += fan_energy
            occupied = record['occupancy'][room] > 0
            # Future explicit starts are the announced synthetic timetable, not predictions.
            soon = any(0 < r['minute'] - record['minute'] <= 30 and
                       r['class_start'][room] and r['occupancy'][room] > 0
                       for r in records[i + 1:] if r['minute'] <= record['minute'] + 30)
            key = 'occupied_ac_kwh' if occupied else 'empty_soon_ac_kwh' if soon else 'empty_other_ac_kwh'
            result[key] += ac_energy
    if not math.isclose(sum(result.values()), sum(r['energy_kwh'] for r in records), abs_tol=1e-7):
        raise ValueError('Trace energy does not match equipment/configuration')
    for room in range(2):
        isolated = []
        for record in records:
            row = dict(record)
            row['occupancy'] = np.array(record['occupancy'], copy=True)
            row['occupancy'][1 - room] = 0
            isolated.append(row)
        metrics = summarize(isolated, config.step_minutes)
        for key in ('class_arrivals', 'arrival_measurements', 'uncomfortable_arrivals',
                    'early_class_occupied_minutes', 'early_class_uncomfortable_minutes',
                    'occupied_room_minutes', 'comfortable_room_minutes', 'overheating_pct', 'overcooling_pct'):
            result[f'room_{room}_{key}'] = metrics[key]
        result[f'room_{room}_energy_kwh'] = float(room_energy[room])
        for key, value in pooled_comfort_metrics([metrics]).items():
            result[f'room_{room}_{key}'] = value
    return result


def failure_trace(records, config):
    """Recorded temperatures, action runs, class lead and power for a failed day."""
    rows = []
    for i, record in enumerate(records):
        first = last = i
        while first and records[first - 1]['action'] == record['action']:
            first -= 1
        while last + 1 < len(records) and records[last + 1]['action'] == record['action']:
            last += 1
        row = {k: record[k] for k in ('minute', 'action', 'power_kw', 'cap_kw', 'switches')}
        row['joint_action_run_minutes'] = (last - first + 1) * config.step_minutes
        for room in range(2):
            starts = [r['minute'] - record['minute'] for r in records[i:]
                      if r['class_start'][room] and r['occupancy'][room] > 0]
            row[f'room_{room}_next_class_minutes'] = min(starts) if starts else None
            for key in ('temperature_before', 'temperature', 'effective_temperature', 'arrival_violation',
                        'occupancy', 'class_start'):
                row[f'room_{room}_{key}'] = float(record[key][room])
        rows.append(row)
    return rows


def default_protocol():
    return {'version': 'dqn_confirmation_v1', 'solution_algorithms': ['q_learning', 'dqn'],
        'reference_model': 'models/dqn', 'training_seeds': [0, 1, 2], 'episodes': 400,
        'validation_every': 50, 'aggregation': 'pooled_event_and_room_time',
        'partitions': {
            'tuning': {'seed_offset': 1_100_000, 'days': 20, 'scenarios': list(SCENARIOS[:-1])},
            'confirmation': {'seed_offset': 1_200_000, 'days': 20, 'scenarios': list(SCENARIOS)},
            'final': {'seed_offset': 4_000_000, 'days': 20, 'scenarios': list(SCENARIOS)}},
        'recipes': [
            {'name': 'control', 'algorithm': 'dqn', 'learning_rate': .0003, 'epsilon_decay': .8},
            {'name': 'lower_lr', 'algorithm': 'dqn', 'learning_rate': .0001, 'epsilon_decay': .8},
            {'name': 'slower_exploration', 'algorithm': 'dqn', 'learning_rate': .0003, 'epsilon_decay': 1.}],
        'fixed': {'gamma': .99, 'batch_size': 64, 'warmup': 500, 'target_interval': 250,
                  'train_frequency': 1, 'dqn_features': 'comfort_v1', 'reward_scale': 10.,
                  'epsilon_start': 1., 'epsilon_end': .05},
        'selection': 'All three comfort quantities non-regressing and strictly lower energy; checkpoint rank: eligible, lower energy, higher comfort, higher original return; earliest exact-rank tie. Representative: lower energy, higher comfort, recipe name, seed.',
        'confirmation_rule': 'Same aggregate guard and no worsening of any quantity within any scenario',
        'budget_stop': 'At most 9 runs / 3600 episodes; no extension after inspecting outcomes',
        'baseline_role': 'Comparison only; never provide actions or labels to RL training',
        'final_rule': 'Evaluate frozen representative once; no reselection using confirmation or final results'}


def validate_study_protocol(protocol, consumed_seeds=()):
    partitions = protocol['partitions']
    if set(partitions) != {'tuning', 'confirmation', 'final'}:
        raise ValueError('Require tuning, confirmation and final partitions')
    used = set()
    for name, partition in partitions.items():
        start, days = partition['seed_offset'], partition['days']
        if type(start) is not int or type(days) is not int or days < 1:
            raise ValueError('Invalid evaluation seed range')
        lower, upper = (2_000_000, 2**32) if name == 'final' else (1_000_000, 2_000_000)
        if not lower <= start < start + days <= upper:
            raise ValueError('Evaluation must stay outside training and in its validation/test partition')
        seeds = set(range(start, start + days))
        if seeds & used:
            raise ValueError('Evaluation partitions overlap')
        if seeds & set(consumed_seeds):
            raise ValueError('Evaluation contains previously used days')
        used |= seeds
        if not partition['scenarios'] or len(set(partition['scenarios'])) != len(partition['scenarios']) or not set(partition['scenarios']) <= set(SCENARIOS):
            raise ValueError('Unknown or duplicated scenarios')
    if set(partitions['tuning']['scenarios']) != set(SCENARIOS[:-1]) or any(
            set(partitions[k]['scenarios']) != set(SCENARIOS) for k in ('confirmation', 'final')):
        raise ValueError('Use four tuning scenarios and all five confirmation/final scenarios')
    if protocol['solution_algorithms'] != ['q_learning', 'dqn'] or any(
            r.get('algorithm') != 'dqn' for r in protocol['recipes']):
        raise ValueError('This study permits standard DQN recipes only')
    if len(protocol['recipes']) > 3 or protocol['training_seeds'] != [0, 1, 2] or protocol['episodes'] != 400:
        raise ValueError('Training budget exceeds or differs from the declared bounded experiment')
    names = [r['name'] for r in protocol['recipes']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate recipe names')


def paired_intervals(rows, candidate, reference, samples=1000, seed=2026):
    """Scenario-stratified bootstrap of matched whole days for one frozen pair."""
    pairs = {}
    for row in rows:
        if row['controller'] not in (candidate, reference):
            continue
        key = row['scenario'], int(row['day_seed'])
        bucket = pairs.setdefault(key, {})
        if row['controller'] in bucket:
            raise ValueError('duplicate controller/day evidence')
        bucket[row['controller']] = row
    if not pairs or any(set(p) != {candidate, reference} for p in pairs.values()):
        raise ValueError('All days must be paired')
    strata = {}
    for (scenario, _), pair in sorted(pairs.items()):
        strata.setdefault(scenario, []).append(pair)
    def difference(selected):
        a = score_days([p[candidate] for p in selected])
        b = score_days([p[reference] for p in selected])
        if any(a[k] is None or b[k] is None for k in QUALITY):
            raise ValueError('Complete finite comfort evidence required for intervals')
        return {k: a[k] - b[k] for k in QUALITY}
    point = difference(list(pairs.values()))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        selected = [group[i] for group in strata.values() for i in rng.integers(0, len(group), len(group))]
        draws.append(difference(selected))
    return {k: {'difference': v, 'ci95': np.percentile([d[k] for d in draws], [2.5, 97.5]).tolist()}
            for k, v in point.items()}
