"""Reproduce the bounded, RL-only DQN study without replacing default models.

python -m scripts.dqn_study init --output outputs/dqn_confirmation
python -m scripts.dqn_study diagnose --output outputs/dqn_confirmation
python -m scripts.dqn_study train --output outputs/dqn_confirmation --jobs 3
python -m scripts.dqn_study select --output outputs/dqn_confirmation
python -m scripts.dqn_study confirmation --output outputs/dqn_confirmation
python -m scripts.dqn_study final --output outputs/dqn_confirmation
python -m scripts.dqn_study report --output outputs/dqn_confirmation
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from thermorl.baselines import RuleController
from thermorl.config import Config
from thermorl.dqn import DQN
from thermorl.env import ClassroomEnv
from thermorl.experiment import rollout, write_csv, plotting
from thermorl.metrics import METRIC_DEFINITIONS
from thermorl.study import (default_protocol, validate_study_protocol, diagnose_records,
    grouped_scores, nonregression, confirmation_decision, paired_intervals, failure_trace)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    if path.exists():
        raise ValueError(f'Artifact already exists: {path}; preserve previous evidence')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def consumed_days(roots):
    seeds = set()
    for root in roots:
        for file in Path(root).rglob('metadata.json'):
            data = read(file)
            seeds.update(data.get('day_seeds', []))
            seeds.update(data.get('validation', {}).get('seeds', []))
        for file in Path(root).rglob('protocol.json'):
            for part in read(file).get('partitions', {}).values():
                seeds.update(range(part['seed_offset'], part['seed_offset'] + part['days']))
    return seeds


def initialize(root, consumed_roots=('outputs', 'models', 'docs')):
    root = Path(root)
    if root.exists():
        raise ValueError('Study output already exists; preserve previous evidence')
    protocol = default_protocol()
    validate_study_protocol(protocol, consumed_days(consumed_roots))
    source = Path(protocol['reference_model'])
    metadata = read(source / 'metadata.json')
    if metadata['algorithm'] != 'dqn' or not metadata.get('training_complete', True):
        raise ValueError('Reference must be a completed standard DQN')
    protocol['reference_checkpoint_sha256'] = digest(source / metadata['checkpoint'])
    protocol['reference_config_sha256'] = digest(source / 'config.json')
    protocol['metric_definitions'] = METRIC_DEFINITIONS
    protocol['consumed_seed_check'] = 'Checked saved metadata and reserved study protocols before creating this study'
    root.mkdir(parents=True)
    shutil.copytree(source, root / 'reference')
    save(root / 'protocol.json', protocol)
    save(root / 'protocol-lock.json', {'sha256': digest(root / 'protocol.json')})
    print(f'Frozen reference and disjoint protocol: {root}', flush=True)


def load_study(root):
    root = Path(root)
    protocol = read(root / 'protocol.json')
    if digest(root / 'protocol.json') != read(root / 'protocol-lock.json')['sha256']:
        raise ValueError('Protocol hash mismatch')
    validate_study_protocol(protocol)
    metadata = read(root / 'reference/metadata.json')
    if digest(root / 'reference' / metadata['checkpoint']) != protocol['reference_checkpoint_sha256']:
        raise ValueError('Reference checkpoint hash mismatch')
    if digest(root / 'reference/config.json') != protocol['reference_config_sha256']:
        raise ValueError('Reference configuration hash mismatch')
    return protocol


def evaluate_partition(root, partition, models, destination):
    protocol = load_study(root)
    spec = protocol['partitions'][partition]
    config = Config.load(root / 'reference/config.json')
    if destination.exists():
        raise ValueError('Evaluation output already exists; do not repeat assessment')
    controllers = {k: RuleController(k) for k in ('timetable', 'thermostat', 'rotation')}
    hashes = {}
    for label, folder in models.items():
        meta = read(folder / 'metadata.json')
        if asdict(Config.load(folder / 'config.json')) != asdict(config):
            raise ValueError('Model environment/reward configuration mismatch')
        hashes[label] = digest(folder / meta['checkpoint'])
        controllers[label] = DQN.load(folder / meta['checkpoint'])
    destination.mkdir(parents=True)
    rows = []
    for scenario in spec['scenarios']:
        for seed in range(spec['seed_offset'], spec['seed_offset'] + spec['days']):
            for name, agent in controllers.items():
                metrics, records = rollout(ClassroomEnv(config, scenario), agent, seed)
                rows.append({'controller': name, 'scenario': scenario, 'day_seed': seed,
                             **metrics, **diagnose_records(records, config)})
        print(f'{partition}: {scenario}, {spec["days"]} paired days', flush=True)
    write_csv(destination / 'daily_metrics.csv', rows)
    scores = {name: grouped_scores([r for r in rows if r['controller'] == name]) for name in controllers}
    save(destination / 'scores.json', scores)
    save(destination / 'metadata.json', {'partition': partition, 'scenarios': spec['scenarios'],
        'day_seeds': list(range(spec['seed_offset'], spec['seed_offset'] + spec['days'])),
        'model_hashes': hashes, 'protocol_sha256': digest(root / 'protocol.json'),
        'metric_definitions': METRIC_DEFINITIONS, 'synthetic': True})
    return rows, scores


def diagnose(root):
    rows, scores = evaluate_partition(root, 'tuning', {'reference': root / 'reference'}, root / 'diagnosis')
    diagnosis_report(root, rows, scores)
    save_failure_case(root)


def save_failure_case(root):
    with (root / 'diagnosis/daily_metrics.csv').open() as stream:
        reference = [r for r in csv.DictReader(stream) if r['controller'] == 'reference']
    worst = max(reference, key=lambda r: (float(r['arrival_discomfort_pct']), int(r['day_seed']), r['scenario']))
    config = Config.load(root / 'reference/config.json')
    metadata = read(root / 'reference/metadata.json')
    day = int(worst['day_seed'])
    for name, agent in {'reference': DQN.load(root / 'reference' / metadata['checkpoint']),
                        'timetable': RuleController('timetable')}.items():
        _, records = rollout(ClassroomEnv(config, worst['scenario']), agent, day)
        destination = root / 'diagnosis' / f'failure_{name}.csv'
        if destination.exists():
            raise ValueError('Failure trace already exists; preserve diagnostic evidence')
        write_csv(destination, failure_trace(records, config))
    save(root / 'diagnosis/failure_case.json', {'scenario': worst['scenario'], 'day_seed': day,
        'selection': 'Worst reference arrival discomfort; ties highest day seed then scenario',
        'purpose': 'Recorded reference/timetable comparison on an already consumed tuning day, not a new assessment'})


def diagnosis_report(root, rows, scores):
    report = ['# Frozen DQN diagnosis', '', 'Synthetic tuning data; comparison controllers do not train or override DQN.', '',
        '| Policy | Arrival discomfort % | First 15 min discomfort % | Whole-day comfort % | kWh/day |',
        '|---|---:|---:|---:|---:|']
    for name, result in scores.items():
        s = result['overall']
        report.append(f'| {name} | {s["mean_arrival_discomfort_pct"]:.2f} | {s["mean_class_start_violation_pct"]:.2f} | {s["mean_comfort_pct"]:.2f} | {s["mean_energy_kwh"]:.2f} |')
    ref = [r for r in rows if r['controller'] == 'reference']
    report += ['', '## Mean daily energy attribution', '', '| Use | kWh/day |', '|---|---:|']
    for key in ('occupied_ac_kwh', 'empty_soon_ac_kwh', 'empty_other_ac_kwh', 'fan_kwh'):
        report.append(f'| {key} | {np.mean([r[key] for r in ref]):.2f} |')
    report += ['', 'Empty-room cooling within 30 minutes of an announced start is reported separately; it may be useful precooling. Other empty-room cooling is a diagnostic category, not proof that each action was unnecessary.', '',
               '## Reference by scenario and room', '',
               '| Scenario | Room | Arrival discomfort % | First 15 min discomfort % | Whole-day comfort % | Too hot % | Too cold % | kWh/day |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
    from thermorl.study import score_days
    for scenario in sorted({r['scenario'] for r in ref}):
        group = [r for r in ref if r['scenario'] == scenario]
        for room in range(2):
            prefix = f'room_{room}_'
            isolated = [{k.removeprefix(prefix): v for k, v in row.items() if k.startswith(prefix)} for row in group]
            s = score_days(isolated)
            occupied = sum(r['occupied_room_minutes'] for r in isolated)
            hot, cold = [sum(r[key] * r['occupied_room_minutes'] for r in isolated) / occupied
                         for key in ('overheating_pct', 'overcooling_pct')]
            report.append(f'| {scenario} | {room + 1} | {s["mean_arrival_discomfort_pct"]:.2f} | {s["mean_class_start_violation_pct"]:.2f} | {s["mean_comfort_pct"]:.2f} | {hot:.2f} | {cold:.2f} | {s["mean_energy_kwh"]:.2f} |')
    report += ['',
               '## Worst arrival day (declared diagnostic example)', '']
    worst = max(ref, key=lambda r: (r['arrival_discomfort_pct'], r['day_seed'], r['scenario']))
    report.append(f'{worst["scenario"]}, seed {worst["day_seed"]}: {worst["arrival_discomfort_pct"]:.2f}% arrival discomfort. Per-room counts and hot/cold percentages for every day are in daily_metrics.csv. failure_reference.csv and failure_timetable.csv contain temperatures before/after each action, fan-adjusted comfort, class starts/lead times, joint-action run durations, switching and available/used power. Compare the 30 minutes before each start and its first 15 minutes.')
    report += ['', '## Experiment hypotheses', '',
        'Lower learning rate tests whether smaller TD updates improve stability; slower epsilon decay tests whether additional feasible-action exploration improves coverage before greedy evaluation. Neither injects a controller or labelled actions. Both are single changes to the control recipe; all three have equal budgets. The diagnosis does not establish either hypothesis.', '']
    (root / 'diagnosis/REPORT.md').write_text('\n'.join(report))


def run_specs(root, protocol):
    for recipe in protocol['recipes']:
        for seed in protocol['training_seeds']:
            yield recipe, seed, root / 'runs' / f'{recipe["name"]}_seed{seed}'


def train(root, jobs):
    protocol = load_study(root)
    if not (root / 'diagnosis/REPORT.md').is_file():
        raise ValueError('Complete diagnosis before training')
    if (root / 'selection.json').exists():
        raise ValueError('Selection is already frozen; cannot continue tuning')
    tuning = protocol['partitions']['tuning']
    def run(spec):
        recipe, seed, output = spec
        if output.exists():
            metadata = read(output / 'metadata.json')
            if metadata.get('training_complete'):
                print(f'Preserving completed run {output.name}', flush=True)
                return
            raise ValueError(f'Interrupted run {output}; no optimizer-resume claim or overwrite permitted')
        args = [sys.executable, 'train.py', '--algo', 'dqn', '--seed', str(seed),
            '--episodes', str(protocol['episodes']), '--output', str(output),
            '--config', str(root / 'reference/config.json'), '--selection', 'nonregression',
            '--reference-model', str(root / 'reference'), '--validation-aggregation', 'pooled',
            '--validation-seed-offset', str(tuning['seed_offset']), '--validation-days', str(tuning['days']),
            '--validation-every', str(protocol['validation_every']),
            '--learning-rate', str(recipe['learning_rate']), '--epsilon-decay', str(recipe['epsilon_decay'])]
        for key, value in protocol['fixed'].items():
            args.extend(['--' + key.replace('_', '-'), str(value)])
        log = root / 'logs' / f'{output.name}.log'
        log.parent.mkdir(exist_ok=True)
        print(f'Starting {output.name}', flush=True)
        env = dict(os.environ, MPLCONFIGDIR='/tmp/thermorl-mpl', XDG_CACHE_HOME='/tmp/thermorl-cache')
        with log.open('x') as stream:
            subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)
        print(f'Completed {output.name}', flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        list(executor.map(run, run_specs(root, protocol)))


def select(root):
    protocol = load_study(root)
    reference_score = read(root / 'diagnosis/scores.json')['reference']['overall']
    entries = []
    for recipe, seed, folder in run_specs(root, protocol):
        metadata = read(folder / 'metadata.json')
        tuning = protocol['partitions']['tuning']
        if (not metadata.get('training_complete') or metadata['seed'] != seed or
            metadata['algorithm'] != 'dqn' or metadata['episodes'] != protocol['episodes'] or
            metadata['selection_mode'] != 'nonregression' or metadata.get('initialization') or
            metadata['validation']['seeds'] != list(range(tuning['seed_offset'], tuning['seed_offset'] + tuning['days'])) or
            metadata['validation']['scenarios'] != tuning['scenarios'] or
            metadata['validation']['aggregation'] != 'pooled' or
            metadata['reference_model']['checkpoint_sha256'] != protocol['reference_checkpoint_sha256'] or
            asdict(Config.load(folder / 'config.json')) != asdict(Config.load(root / 'reference/config.json')) or
            metadata['agent_hyperparameters']['learning_rate'] != recipe['learning_rate'] or
            metadata['epsilon']['decay_fraction'] != recipe['epsilon_decay']):
            raise ValueError(f'Run is incomplete or does not match declared protocol: {folder}')
        for key, value in protocol['fixed'].items():
            actual = (metadata['epsilon'][key.removeprefix('epsilon_')] if key.startswith('epsilon_') else
                      metadata['agent_hyperparameters']['feature_mode' if key == 'dqn_features' else key])
            if actual != value:
                raise ValueError(f'Run setting mismatch: {folder} {key}')
        score = metadata['selected_validation']
        from thermorl.study import QUALITY
        if any(metadata['reference_validation'].get(k) is None or reference_score.get(k) is None or
               not np.isclose(metadata['reference_validation'][k], reference_score[k], rtol=0, atol=1e-9)
               for k in QUALITY):
            raise ValueError('Reference scores do not match paired tuning diagnosis')
        entries.append({'run': str(folder), 'recipe': recipe['name'], 'seed': seed,
            'episode': metadata['selected_episode'], 'score': score,
            'checkpoint_sha256': digest(folder / metadata['checkpoint']),
            'eligible': nonregression(score, metadata['reference_validation'])})
    eligible = [r for r in entries if r['eligible']]
    selected = min(eligible, key=lambda r: (r['score']['mean_energy_kwh'], -r['score']['mean_comfort_pct'],
                                          r['recipe'], r['seed'])) if eligible else None
    save(root / 'selection.json', {'data': 'Tuning only; no confirmation or final outcomes inspected',
        'protocol_sha256': digest(root / 'protocol.json'), 'runs': entries, 'selected': selected,
        'decision': 'Frozen candidate for independent confirmation' if selected else 'Keep restored default; no candidate meets all comfort/energy constraints'})
    save(root / 'selection-lock.json', {'sha256': digest(root / 'selection.json')})
    print('Selected: ' + (selected['run'] if selected else 'none; current default retained'), flush=True)


def verify_confirmation(root, selection):
    result = read(root / 'confirmation/decision.json')
    if (result.get('selected_checkpoint_sha256') != selection['selected']['checkpoint_sha256'] or
            result.get('protocol_sha256') != selection['protocol_sha256']):
        raise ValueError('Final assessment requires confirmation of the same frozen checkpoint and protocol')
    return result['eligible']


def assess(root, partition):
    load_study(root)
    if partition == 'final' and not (root / 'confirmation/decision.json').is_file() and (
            not (root / 'selection.json').exists() or read(root / 'selection.json')['selected'] is not None):
        raise ValueError('Independent confirmation must finish before final assessment')
    if not (root / 'selection.json').is_file():
        raise ValueError('Frozen tuning selection required before assessment')
    selection = read(root / 'selection.json')
    if digest(root / 'selection.json') != read(root / 'selection-lock.json')['sha256']:
        raise ValueError('Frozen selection hash mismatch')
    selected = selection['selected']
    if not selected:
        print(f'{partition}: skipped; no qualifying tuning candidate, reserved days remain unused', flush=True)
        return
    if partition == 'final' and not verify_confirmation(root, selection):
        print('final: skipped; confirmation failed, reserved days remain unused', flush=True)
        return
    folder = Path(selected['run'])
    if digest(folder / read(folder / 'metadata.json')['checkpoint']) != selected['checkpoint_sha256']:
        raise ValueError('Selected checkpoint hash mismatch')
    if selection['protocol_sha256'] != digest(root / 'protocol.json'):
        raise ValueError('Selection protocol hash mismatch')
    save(root / f'{partition}-started.json', {'selected_checkpoint_sha256': selected['checkpoint_sha256'],
        'rule': 'One assessment only; partial/failed assessment must be documented, never silently repeated'})
    rows, scores = evaluate_partition(root, partition, {'reference': root / 'reference', 'candidate': folder}, root / partition)
    decision = confirmation_decision(scores['candidate'], scores['reference'])
    decision['selected_checkpoint_sha256'] = selected['checkpoint_sha256']
    decision['protocol_sha256'] = selection['protocol_sha256']
    decision['paired_intervals'] = paired_intervals(rows, 'candidate', 'reference')
    decision['uncertainty'] = '95% paired whole-day bootstrap intervals stratified by scenario; one frozen pair, not training-seed uncertainty'
    save(root / partition / 'decision.json', decision)
    print(json.dumps({'partition': partition, 'eligible': decision['eligible'], 'reasons': decision['reasons']}), flush=True)


def report(root):
    protocol = load_study(root)
    selection = read(root / 'selection.json')
    lines = ['# Bounded standard-DQN study', '',
        'RL methods only: DQN learns from simulated transitions and TD targets; rule controllers are comparison references. No labelled controller actions, supervised pretraining or rule overrides.', '',
        '## Tuning-selected checkpoints', '',
        '| Recipe | Training seed | Episode | Arrival discomfort % | First 15 min discomfort % | Whole-day comfort % | kWh/day | Qualifies |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for run in selection['runs']:
        s = run['score']
        lines.append(f'| {run["recipe"]} | {run["seed"]} | {run["episode"]} | {s["mean_arrival_discomfort_pct"]:.2f} | {s["mean_class_start_violation_pct"]:.2f} | {s["mean_comfort_pct"]:.2f} | {s["mean_energy_kwh"]:.2f} | {run["eligible"]} |')
    lines += ['', selection['decision'] + '.', '',
        'All checkpoints above are selected using tuning data, not held-out performance. Fallback checkpoints failing the guard remain ineligible. Training-seed variation is distinct from variation across weather/timetable days.', '',
        '## Independent assessments', '']
    for part in ('confirmation', 'final'):
        file = root / part / 'decision.json'
        if file.is_file():
            result = read(file)
            lines += [f'### {part}', '', f'Passes measured replacement screen: {result["eligible"]}.', '',
                *result['reasons'], '', '| Quantity | Candidate − reference | Paired 95% interval |', '|---|---:|---|']
            for key, value in result['paired_intervals'].items():
                lines.append(f'| {key} | {value["difference"]:.3f} | {value["ci95"][0]:.3f}, {value["ci95"][1]:.3f} |')
            lines += ['', result['uncertainty'] + '.', '']
        else:
            lines += [f'{part}: {assessment_status(root, part)}', '']
    lines += ['## Limits and default model', '',
        'No default model is modified by this tool. Passing finite synthetic tests cannot guarantee every future class is comfortable. Beating the restored DQN and beating timetable at comparable comfort are separate claims. No energy-saving superiority claim is made without qualifying evidence.', '',
        'The three recipe means and sample standard deviations below describe independent training seeds, not a causal algorithm comparison or an interval over test days.', '',
        '| Recipe | Whole-day comfort mean ± seed SD | kWh/day mean ± seed SD |', '|---|---:|---:|']
    for name in sorted({r['recipe'] for r in selection['runs']}):
        group = [r['score'] for r in selection['runs'] if r['recipe'] == name]
        comfort = [s['mean_comfort_pct'] for s in group]; energy = [s['mean_energy_kwh'] for s in group]
        lines.append(f'| {name} | {np.mean(comfort):.2f} ± {np.std(comfort, ddof=1):.2f} | {np.mean(energy):.2f} ± {np.std(energy, ddof=1):.2f} |')
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    reference_scores = read(root / 'diagnosis/scores.json')
    summary_rows = [{'label': name, 'score': score['overall']} for name, score in reference_scores.items()]
    for name in sorted({r['recipe'] for r in selection['runs']}):
        group = [r['score'] for r in selection['runs'] if r['recipe'] == name]
        summary_rows.append({'label': name + ' (mean of 3 training seeds; tuning)',
            'score': {k: float(np.mean([r[k] for r in group])) for k in group[0] if k in (
                'mean_energy_kwh', 'mean_comfort_pct', 'mean_arrival_discomfort_pct', 'mean_class_start_violation_pct')}})
    tuning = protocol['partitions']['tuning']
    summary = {'partition': 'tuning', 'scenario_count': len(tuning['scenarios']),
        'days_per_scenario': tuning['days'], 'total_days': len(tuning['scenarios']) * tuning['days'],
        'reference_checkpoint': read(root / 'reference/metadata.json')['selected_episode'],
        'reference_checkpoint_sha256': protocol['reference_checkpoint_sha256'],
        'training_seeds': protocol['training_seeds'], 'status': selection['decision'] + '. Default model unchanged.',
        'rows': summary_rows}
    (root / 'gui-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    plt = plotting()
    fig, axis = plt.subplots(figsize=(8, 5))
    for name in sorted({r['recipe'] for r in selection['runs']}):
        group = [r for r in selection['runs'] if r['recipe'] == name]
        axis.scatter([r['score']['mean_energy_kwh'] for r in group],
                     [r['score']['mean_comfort_pct'] for r in group], label=name)
    ref = read(root / 'diagnosis/scores.json')
    for name in ('reference', 'timetable', 'thermostat', 'rotation'):
        s = ref[name]['overall']
        axis.scatter(s['mean_energy_kwh'], s['mean_comfort_pct'], marker='x', s=80, label=name)
    axis.set(xlabel='Mean daily energy (kWh)', ylabel='Comfortable occupied room-time (%)',
             title='Tuning evidence · 80 paired synthetic days · three training seeds')
    axis.legend(fontsize=8); fig.tight_layout(); fig.savefig(root / 'comparison.png', dpi=160); plt.close(fig)


def assessment_status(root, partition):
    if (root / f'{partition}-started.json').exists():
        return 'Started but incomplete; reserved partition potentially consumed. Do not rerun or describe these days as untouched.'
    return 'Not evaluated; reserved days remain unused.'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['init', 'diagnose', 'train', 'select', 'confirmation', 'final', 'report'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    try:
        if args.command == 'init': initialize(args.output)
        elif args.command == 'diagnose': diagnose(args.output)
        elif args.command == 'train': train(args.output, args.jobs)
        elif args.command == 'select': select(args.output)
        elif args.command in ('confirmation', 'final'): assess(args.output, args.command)
        else: report(args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
