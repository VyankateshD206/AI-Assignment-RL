"""Expanded assessment of frozen policies; preserves the original learning report.

Run: .venv/bin/python -m scripts.summarize_comfort outputs/fair_comfort_test docs/fair_comfort
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from thermorl.metrics import pooled_comfort_metrics


METRICS = ('arrival_discomfort_pct', 'class_start_violation_pct', 'comfort_pct', 'energy_kwh')


def aggregate_rows(rows, groups):
    """Each run pools its own counts; only then aggregate independent runs."""
    result = []
    for label, controllers in groups.items():
        runs = [[r for r in rows if r['controller'] == name] for name in controllers]
        if not runs or any(not run for run in runs):
            continue
        summaries = [pooled_comfort_metrics(run) | {
            'energy_kwh': float(np.mean([float(r['energy_kwh']) for r in run]))} for run in runs]
        output = {'method': label, 'runs': len(runs), 'days_per_run': len(runs[0])}
        for key in METRICS:
            values = [s[key] for s in summaries if s[key] is not None]
            # A missing run must not silently disappear from the reported mean.
            complete = len(values) == len(runs)
            output[key + '_mean'] = float(np.mean(values)) if values and complete else None
            output[key + '_seed_std'] = float(np.std(values, ddof=1)) if len(values) > 1 and complete else None
            day_stds = []
            denominator = {'arrival_discomfort_pct': 'class_arrivals',
                           'class_start_violation_pct': 'early_class_occupied_minutes',
                           'comfort_pct': 'occupied_room_minutes'}.get(key)
            for run in runs:
                days = [float(r[key]) for r in run if r.get(key) not in (None, '')
                        and (not denominator or float(r[denominator]) > 0)]
                if len(days) > 1:
                    day_stds.append(float(np.std(days, ddof=1)))
            output[key + '_day_std_mean'] = float(np.mean(day_stds)) if day_stds else None
        output['class_arrivals_per_run'] = ';'.join(str(sum(int(float(r['class_arrivals'])) for r in run)) for run in runs)
        output['cap_violations'] = sum(int(float(r['cap_violations'])) for run in runs for r in run)
        result.append(output)
    return result


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--reference', type=Path, default=Path('docs/learning_test_daily.csv'))
    args = parser.parse_args()
    metadata = json.loads((args.evaluation / 'metadata.json').read_text())
    if metadata['split'] != 'test' or metadata.get('metric_definitions', {}).get('version') != 'comfort_v2':
        parser.error('Requires a comfort_v2 test evaluation of frozen checkpoints')
    if args.destination.exists() and any(args.destination.iterdir()):
        parser.error('Report directory must be empty; preserve earlier reports')
    rows = read_csv(args.evaluation / 'daily_metrics.csv')
    old = {(r['controller'], r['scenario'], r['day_seed']): r for r in read_csv(args.reference)}
    if set(old) != {(r['controller'], r['scenario'], r['day_seed']) for r in rows}:
        parser.error('Paired controller/scenario/day keys differ from the original evaluation')
    for row in rows:
        previous = old[row['controller'], row['scenario'], row['day_seed']]
        for key in ('energy_kwh', 'comfort_pct', 'return'):
            if not np.isclose(float(row[key]), float(previous[key]), rtol=1e-10, atol=1e-10):
                parser.error(f'Changed original {key}: {row["controller"]}, {row["scenario"]}, {row["day_seed"]}')
    names = list(dict.fromkeys(r['controller'] for r in rows))
    groups = {'Timetable': ['timetable'], 'Thermostat': ['thermostat'], 'Rotation': ['rotation'],
              'Initial DQN (60 episodes)': ['dqn_seed0'],
              'Improved DQN': [n for n in names if n.startswith('dqn_comfort_seed')],
              'Initial Q-learning (60 episodes)': ['q_seed0'],
              'Compact Q-learning': [n for n in names if n.startswith('q_compact_seed')]}
    aggregate = []
    lines = ['# Fair comfort evaluation', '',
             'Expanded assessment of the same frozen policies, with no retraining or checkpoint reselection. '
             'All original daily energy, whole-day comfort and return values match the prior evaluation within 1e-10 tolerance. '
             'The original report remains unchanged.', '',
             '## Measurement and aggregation', '',
             '- Arrival discomfort: occupied explicit class-start events outside the configured band before the action; '
             'the proxy uses the fan already running before arrival.',
             '- First-15-minute discomfort: uncomfortable occupied room-minutes within the first 15 clock minutes, '
             'using post-action samples (+5, +10, +15 minutes at the default step).',
             '- Whole-day comfort: comfortable occupied room-minutes; energy includes all electricity, including empty-room pre-cooling.',
             '- Pool event/minute counts within each run before computing percentages. Average daily kWh. '
             'Then report mean ± sample SD across three independent improved-agent runs. '
             'Baselines and initial agents have one run, so no training-seed SD is estimated.',
             '- Across-day variability is exported separately as `*_day_std_mean`: '
             'the mean of within-run sample SDs of applicable daily values. It is not training uncertainty.',
             '- These pooled comfort percentages can differ from the original report’s unweighted mean of daily percentages. '
             'Daily values remain unchanged. Not applicable means a zero denominator; unavailable means missing arrival snapshots.', '',
             f'Test scenarios: {", ".join(metadata["scenarios"])}. Paired day seeds: '
             f'{metadata["day_seeds"][0]}–{metadata["day_seeds"][-1]}. Metric version: comfort_v2.', '']
    for scope in ['All scenarios', *metadata['scenarios']]:
        subset = rows if scope == 'All scenarios' else [r for r in rows if r['scenario'] == scope]
        summaries = aggregate_rows(subset, groups)
        aggregate.extend({'scope': scope, **r} for r in summaries)
        lines += [f'## {scope}', '', '| Policy | Arrival discomfort (%) ↓ | First-15-minute discomfort (%) ↓ | Whole-day comfortable (%) ↑ | Energy (kWh/day) ↓ |',
                  '|---|---:|---:|---:|---:|']
        for row in summaries:
            cells = []
            for key in METRICS:
                value, std = row[key + '_mean'], row[key + '_seed_std']
                cells.append('Not applicable / unavailable' if value is None else
                             f'{value:.2f}' + (f' ± {std:.2f}' if std is not None else ''))
            lines.append('| ' + ' | '.join([row['method'], *cells]) + ' |')
        lines.append('')
    lines += ['## Interpretation and limits', '',
              'Assess all columns together. Low energy with poor comfort is not a successful policy. '
              'A low early-class discomfort percentage does not erase uncomfortable entry conditions. '
              'Pre-cooling may be useful; its cost is already included in total kWh. '
              'These measurements do not guarantee that RL beats timetable control.', '',
              'The fan offset is a configured simulation assumption, not a physiological comfort model. '
              'Results apply to synthetic classrooms. Different training budgets and representation changes prevent algorithm-only causal claims.', '',
              '[Daily counts and metrics](daily_metrics.csv) · [Aggregate results and separate variability](aggregate.csv) · '
              '[Protocol and checkpoint metadata](metadata.json)', '']
    args.destination.mkdir(parents=True, exist_ok=True)
    for name in ['daily_metrics.csv', 'diagnostics.csv', 'metadata.json']:
        shutil.copyfile(args.evaluation / name, args.destination / name)
    with (args.destination / 'aggregate.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (args.destination / 'REPORT.md').write_text('\n'.join(lines))
    print(f'Wrote {args.destination / "REPORT.md"}; verified {len(rows)} paired daily results unchanged.')


if __name__ == '__main__':
    main()
