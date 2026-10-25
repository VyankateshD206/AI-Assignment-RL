"""Summarize paired test days without treating training seeds as independent days.

Run: .venv/bin/python scripts/summarize_learning.py outputs/learning_test docs
"""
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evaluation', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.evaluation / 'metadata.json').read_text())
    if metadata['split'] != 'test':
        parser.error('This report requires a test split, not checkpoint-selection days')
    with (args.evaluation / 'daily_metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    with (args.evaluation / 'diagnostics.csv').open() as stream:
        diagnostics = list(csv.DictReader(stream))
    args.destination.mkdir(parents=True, exist_ok=True)
    for original, saved in [('daily_metrics.csv', 'learning_test_daily.csv'), ('diagnostics.csv', 'learning_test_diagnostics.csv'), ('metadata.json', 'learning_test_metadata.json')]:
        shutil.copyfile(args.evaluation / original, args.destination / saved)
    metrics = ['comfort_pct', 'energy_kwh', 'overheating_pct', 'overcooling_pct', 'worst_room_mean_discomfort_c', 'cap_violations']
    names = list(dict.fromkeys(r['controller'] for r in rows))
    groups = {
        'Timetable baseline': ['timetable'], 'Thermostat baseline': ['thermostat'],
        'Rotation baseline': ['rotation'], 'Initial DQN (60 episodes)': ['dqn_seed0'],
        'Improved DQN': [n for n in names if n.startswith('dqn_comfort_seed')],
        'Initial Q-learning (60 episodes)': ['q_seed0'],
        'Compact Q-learning': [n for n in names if n.startswith('q_compact_seed')],
    }
    aggregate = []
    for scope in ['All scenarios', 'restricted', 'heatwave']:
        for label, controllers in groups.items():
            if not controllers or not all(n in names for n in controllers):
                continue
            result = {'scope': scope, 'method': label, 'training_runs': len(controllers)}
            for metric in metrics:
                per_run = [float(np.mean([float(r[metric]) for r in rows if r['controller'] == n and (scope == 'All scenarios' or r['scenario'] == scope)])) for n in controllers]
                result[metric + '_mean'] = float(np.mean(per_run))
                result[metric + '_seed_std'] = float(np.std(per_run, ddof=1)) if len(per_run) > 1 else 0.0
            aggregate.append(result)
    csv_path = args.destination / 'learning_test_aggregate.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader(); writer.writerows(aggregate)
    lines = ['# Measured RL improvements', '',
             'The thermal dynamics, comfort band, power constraints and reward weights are unchanged. New policies use different state representations, longer training and validation-based checkpoint selection. These bundled changes do not establish the contribution of each change separately.', '',
             '## Evaluation protocol', '',
             f"- Split: **{metadata['split']}**, {len(metadata['day_seeds'])} paired days per scenario; seeds {metadata['day_seeds'][0]}–{metadata['day_seeds'][-1]}.",
             f"- Scenarios: {', '.join(metadata['scenarios'])}.",
             '- Checkpoints are selected by highest mean original, unscaled return on fixed validation seeds 1000000 and 1000001 across the four training-distribution scenarios. Heatwave does not select checkpoints.',
             '- For new methods, average days/scenarios within each trained run first, then average those run means. ± values below are the sample standard deviation across independent training-seed means, not confidence intervals. Baselines and initial runs have one controller; their day variation is available in the raw evaluation summary.',
             '- Comfortable, too hot and too cold percentages refer to occupied room-time and partition it. Empty intervals do not earn comfort credit.', '',
             '## Test results', '']
    for scope in ['All scenarios', 'restricted', 'heatwave']:
        lines += [f'### {scope}', '', '| Method | Runs | Comfortable (%) ↑ | Energy (kWh) ↓ | Too hot (%) | Too cold (%) |', '|---|---:|---:|---:|---:|---:|']
        for row in aggregate:
            if row['scope'] != scope:
                continue
            def value(metric):
                result = f"{row[metric + '_mean']:.2f}"
                if row['training_runs'] > 1:
                    result += f" ± {row[metric + '_seed_std']:.2f}"
                return result
            lines.append(f"| {row['method']} | {row['training_runs']} | {value('comfort_pct')} | {value('energy_kwh')} | {value('overheating_pct')} | {value('overcooling_pct')} |")
        lines.append('')
    lines += ['## Selected checkpoints', '', '| Run | Training episodes | Selected episode | Representation |', '|---|---:|---:|---|']
    for name, run in metadata['models'].items():
        hp = run.get('agent_hyperparameters', {})
        representation = hp.get('feature_mode', hp.get('state_mode', 'legacy'))
        lines.append(f"| {name} | {run['episodes']} | {run.get('selected_episode', run['episodes'])} | {representation} |")
    lines += ['', '## Tabular coverage on test trajectories', '', '| Run | Unseen states (%) |', '|---|---:|']
    for name in names:
        values = [float(r['unseen_state_pct']) for r in diagnostics if r['controller'] == name and r['unseen_state_pct'] != '']
        if values:
            lines.append(f'| {name} | {np.mean(values):.2f} |')
    lines += ['', '## Reproduction', '',
              'Commands below describe the recorded runs. Use fresh output directories when rerunning; existing runs are preserved.', '',
              '```bash',
              'mkdir -p outputs',
              'cp -R models/archive/dqn_seed0 outputs/dqn_seed0',
              'cp -R models/archive/q_seed0 outputs/q_seed0',
              'for seed in 0 1 2; do',
              '  .venv/bin/python train.py --algo dqn --episodes 400 --seed "$seed" --output "outputs/dqn_comfort_seed$seed"',
              '  .venv/bin/python train.py --algo q_learning --episodes 1500 --seed "$seed" --output "outputs/q_compact_seed$seed"',
              'done',
              '.venv/bin/python evaluate.py --models outputs/dqn_seed0 outputs/q_seed0 outputs/dqn_comfort_seed0 outputs/dqn_comfort_seed1 outputs/dqn_comfort_seed2 outputs/q_compact_seed0 outputs/q_compact_seed1 outputs/q_compact_seed2 --days 10 --split test --output outputs/learning_test',
              '.venv/bin/python scripts/summarize_learning.py outputs/learning_test docs',
              '```', '',
              'The two initial 60-episode runs use legacy features/state keys and original hyperparameters (DQN learning rate 0.001, reward scale 1). To recreate them with current code, use `--episodes 60 --dqn-features legacy --learning-rate 0.001 --reward-scale 1` for DQN and `--episodes 60 --q-state legacy` for Q-learning, and evaluate their **last_agent** checkpoint to match the original final-checkpoint protocol. The unchanged initial runs, including metadata, are bundled under `models/archive/` and excluded from default GUI discovery. Newly created runs now select by validation unless their metadata explicitly points to the last checkpoint.', '']
    lines += ['', '## Interpretation limits', '',
              'Compare energy together with comfort. Lower energy caused by leaving rooms hot is not a successful control policy. Improvements over the early 60-episode agents do not imply superiority to the timetable baseline. Training budgets differ between algorithms and initial/improved runs, and no component ablations are included in this batch. The final episode need not be the selected checkpoint.', '',
              'Raw daily values, across-day variation, reward-component totals and temperature traces are in `' + str(args.evaluation) + '`. Measured power-cap violations: ' + str(sum(int(float(r['cap_violations'])) for r in rows)) + '; action masking enforces this constraint.', '',
              '[Raw daily metrics](learning_test_daily.csv) · [Reward/coverage diagnostics](learning_test_diagnostics.csv) · [Run metadata and split](learning_test_metadata.json)', '',
              '![Energy and comfort on held-out test days](learning_test_comparison.png)', '']
    (args.destination / 'LEARNING_IMPROVEMENTS.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    for row in aggregate:
        if row['scope'] != 'All scenarios':
            continue
        ax.errorbar(row['energy_kwh_mean'], row['comfort_pct_mean'], xerr=row['energy_kwh_seed_std'],
                    yerr=row['comfort_pct_seed_std'], fmt='o', capsize=3, label=row['method'])
    ax.set(xlabel='Mean daily energy (kWh)', ylabel='Occupied time comfortable (%)', ylim=(-3, 103), title='Held-out energy and comfort: initial versus improved agents')
    ax.grid(alpha=.15); ax.legend(fontsize=8, loc='best'); fig.tight_layout()
    fig.savefig(args.destination / 'learning_test_comparison.png', dpi=160)
    plt.close(fig)
    print(args.destination / 'LEARNING_IMPROVEMENTS.md')


if __name__ == '__main__':
    main()
