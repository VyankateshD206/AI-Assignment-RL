"""Report the predeclared learning study on fresh paired test days."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from scripts.summarize_comfort import aggregate_rows, METRICS, read_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.evaluation / 'metadata.json').read_text())
    promotion = json.loads(args.manifest.read_text())
    if metadata['split'] != 'test' or min(metadata['day_seeds']) < 3_000_000:
        parser.error('Requires fresh paired test seeds starting at3000000 or later')
    if args.destination.exists() and any(args.destination.iterdir()):
        parser.error('Use a fresh report directory; earlier reports must be preserved')
    rows = read_csv(args.evaluation / 'daily_metrics.csv')
    names = {r['controller'] for r in rows}
    groups = {'Timetable': ['timetable'], 'Thermostat': ['thermostat'], 'Rotation': ['rotation'],
              'Previous DQN (400 episodes)': [f'dqn_comfort_seed{i}' for i in range(3)],
              'DQN scratch trial (1000 episodes)': [f'opt_dqn_seed{i}' for i in range(3)],
              'DQN fine-tuning (400 additional episodes)': [f'opt_dqn_finetune_seed{i}' for i in range(3)],
              'Previous Q-learning (1500 episodes)': [f'q_compact_seed{i}' for i in range(3)],
              'Refined Q-learning (6000 episodes)': [f'opt_q_seed{i}' for i in range(3)]}
    if any(n not in names for group in groups.values() for n in group):
        parser.error('Requires all three old and new training seeds plus three baselines')
    lines = ['# Learning optimization study', '',
             'This study keeps simulator physics, observations, power masks, comfort thresholds and reward weights unchanged. '
             'It tests finer Q-learning states, count-balanced exploration and per-action learning-rate decay; '
             'DQN tests a different discount, learning rate and optimizer update frequency. '
             'New runs also use a different checkpoint-selection rule and larger training budgets. '
             'Results therefore assess the combined training recipe, not an isolated algorithmic improvement.', '',
             '## Protocol', '',
             '- Two seed0 pilots per algorithm chose the full-run recipe using fixed validation days only. '
             'Pilot budgets were400 DQN episodes and2500 Q-learning episodes. Full runs use seeds0,1,2.',
             '- Validation uses four training-distribution scenarios and seeds1000000/1000001. '
             'A fixed90% mean validation comfort floor was declared before trials. '
             'Among qualifying checkpoints select minimum daily energy; otherwise select maximum comfort and explicitly report failure to meet the floor. '
             'Original checkpoints used maximum unscaled validation return.',
             '- Fresh paired assessment uses seeds3000000–3000009 on all five scenarios, including heatwave. '
             'Earlier test results informed development; these are new weather/timetable realizations, not new unseen scenario types. '
             'No fresh-test result chooses a model or changes the training recipe.',
             '- The1000-episode from-scratch DQN recipe failed to improve the existing representative on validation. '
             'Before fresh tests, one separate400-episode fine-tuning recipe was added: each original seed’s validation-selected '
             'checkpoint initializes its corresponding new run, with fresh replay/optimizer, gamma0.99, learning rate0.0001, '
             'frequency1 and epsilon0.10→0.01. It includes an additional400-episode original training budget '
             '(the source checkpoint was selected at350). Failed scratch trials remain reported. '
             'No further search is performed if fine-tuning fails.',
             '- Pool event/minute counts within each trained run; then report mean ± sample SD across three independent training runs. '
             'Energy is mean daily total, including empty-room cooling. Across-day variability is exported separately in aggregate.csv.',
             '- Default GUI promotion requires validation comfort no worse AND energy no higher than the prior bundled representative, '
             'with at least one strict improvement. This guard is independent of the fresh test.', '',
             '## Validation-only GUI decisions', '']
    for algorithm, decision in promotion['algorithms'].items():
        lines += [f'- **{algorithm}:** {decision["decision"]}. '
                  f'Selected source: `{decision["selected_run"]}`. Promoted: {decision["promoted"]}.']
    representative_groups = {}
    for algorithm, decision in promotion['algorithms'].items():
        if decision.get('previous_source_run'):
            representative_groups[algorithm + ' previous representative'] = [Path(decision['previous_source_run']).name]
        if decision['selected_run']:
            representative_groups[algorithm + ' promoted representative'] = [Path(decision['selected_run']).name]
        elif decision.get('best_candidate_run'):
            representative_groups[algorithm + ' research candidate (not promoted)'] = [Path(decision['best_candidate_run']).name]
    lines += ['', '## Frozen representatives on fresh test days', '',
              'These single-run representatives were chosen by validation before fresh tests. '
              'They are distinct from the all-seed averages below. A research candidate may improve comfort at an energy cost and is not automatically a better energy-saving default.', '',
              '| Representative | Arrival discomfort (%) | First15min discomfort (%) | Whole-day comfortable (%) | Daily energy (kWh) |',
              '|---|---:|---:|---:|---:|']
    for result in aggregate_rows(rows, representative_groups):
        lines.append('| ' + ' | '.join([result['method'], *[
            f'{result[k + "_mean"]:.2f}' if result[k + '_mean'] is not None else 'Not applicable' for k in METRICS]]) + ' |')
    lines += ['', '## Selected checkpoints and measured table coverage', '',
              '| Run | Episode | Validation comfortable (%) | Validation energy (kWh) | 90% floor met | Stored Q action entries updated (%) |',
              '|---|---:|---:|---:|---|---:|']
    for name, m in metadata['models'].items():
        if not name.startswith('opt_'):
            continue
        v = m['selected_validation']
        coverage = v.get('stored_action_entries_updated_pct')
        lines.append(f'| {name} | {m["selected_episode"]} | {v["mean_comfort_pct"]:.2f} | {v["mean_energy_kwh"]:.2f} | {m["selection_comfort_floor_met"]} | '
                     + (f'{coverage:.2f}' if coverage is not None else 'Not applicable') + ' |')
    lines += ['', 'Table coverage means the fraction of all16 action entries in stored Q states updated at least once; '
              'it does not imply each feasible action is well estimated or count permanently infeasible actions out of the denominator. '
              'Counts come from actual updates, not whether a Q-value happens to equal zero.', '']
    aggregate = []
    for scope in ['All scenarios', *metadata['scenarios']]:
        subset = rows if scope == 'All scenarios' else [r for r in rows if r['scenario'] == scope]
        summary = aggregate_rows(subset, groups)
        aggregate.extend({'scope': scope, **r} for r in summary)
        lines += [f'## {scope}', '',
                  '| Method | Arrival discomfort (%) ↓ | First15min discomfort (%) ↓ | Whole-day comfortable (%) ↑ | Daily energy (kWh) ↓ |',
                  '|---|---:|---:|---:|---:|']
        for r in summary:
            cells = []
            for key in METRICS:
                value, sd = r[key + '_mean'], r[key + '_seed_std']
                cells.append('Not applicable' if value is None else f'{value:.2f}' + (f' ± {sd:.2f}' if sd is not None else ''))
            lines.append('| ' + ' | '.join([r['method'], *cells]) + ' |')
        lines.append('')
    lines += ['## Interpretation limits', '',
              'Lower energy with worse occupied comfort is not success. A baseline may still win. '
              'Validation promotion describes a representative model, while test means describe all three trained runs. '
              'Different training budgets, selection criteria and simultaneous changes prevent causal attribution to one feature. '
              'The small fixed validation set can be overfit; fresh-day results and training-seed spread must accompany claims. '
              'Tabular state aggregation still omits weather, exact occupancy and time of day, so it remains approximate. '
              'Synthetic comfort proxies do not establish real-building performance.', '',
              '[Daily metrics](daily_metrics.csv) · [Aggregate and separate variability](aggregate.csv) · '
              '[Model and split metadata](metadata.json) · [Validation promotion decisions](promotion.json)', '',
              '![Fresh paired test comparison](comparison.png)', '']
    args.destination.mkdir(parents=True, exist_ok=True)
    for name in ['daily_metrics.csv', 'diagnostics.csv', 'metadata.json']:
        shutil.copyfile(args.evaluation / name, args.destination / name)
    shutil.copyfile(args.manifest, args.destination / 'promotion.json')
    with (args.destination / 'aggregate.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0])); writer.writeheader(); writer.writerows(aggregate)
    (args.destination / 'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(9, 6))
    for r in aggregate:
        if r['scope'] != 'All scenarios':
            continue
        axis.errorbar(r['energy_kwh_mean'], r['comfort_pct_mean'],
                      xerr=r['energy_kwh_seed_std'] or 0, yerr=r['comfort_pct_seed_std'] or 0,
                      fmt='o', capsize=3, label=r['method'])
    axis.set(xlabel='Mean daily energy (kWh)', ylabel='Comfortable occupied room-time (%)',
             ylim=(-3, 103), title='Fresh paired test days: previous and optimized learning recipes')
    axis.legend(fontsize=8); axis.grid(alpha=.15); fig.tight_layout()
    fig.savefig(args.destination / 'comparison.png', dpi=160); plt.close(fig)
    print(f'Wrote {args.destination / "REPORT.md"}: {len(rows)} paired policy-days.')


if __name__ == '__main__':
    main()
