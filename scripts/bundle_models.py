"""Bundle representative runs chosen exclusively by saved validation scores.

Existing bundles are preserved. Run after training completes, before test evaluation.
"""
import json
from pathlib import Path
import shutil


def main():
    manifest = {}
    for label, pattern in [('dqn', 'dqn_comfort_seed*'), ('q_learning', 'q_compact_seed*')]:
        candidates = []
        for run in Path('outputs').glob(pattern):
            metadata = json.loads((run / 'metadata.json').read_text())
            if not (run / ('last_' + metadata['checkpoint'])).exists():
                raise ValueError(f'Training is incomplete: {run}')
            candidates.append((metadata['selected_validation']['mean_return'], run, metadata))
        if not candidates:
            raise ValueError(f'No completed runs for {pattern}')
        _, source, metadata = max(candidates, key=lambda entry: (entry[0], -entry[2]['seed']))
        target = Path('models') / label
        if target.exists() and any(target.iterdir()):
            raise ValueError(f'Bundle exists; preserve it or use a new destination: {target}')
        target.mkdir(parents=True, exist_ok=True)
        for name in [metadata['checkpoint'], 'config.json', 'training.csv', 'validation.csv', 'training_curve.png']:
            shutil.copyfile(source / name, target / name)
        metadata['source_run'] = str(source)
        metadata['bundle_selection'] = 'Highest saved mean original validation return among completed independent training runs; no test metrics used'
        (target / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        manifest[label] = {'source_run': str(source), 'seed': metadata['seed'], 'episode': metadata['selected_episode'], 'validation_return': metadata['selected_validation']['mean_return']}
        print(label, manifest[label])
    for name in ('dqn_seed0', 'q_seed0'):
        source = Path('outputs') / name
        destination = Path('models/archive') / name
        if source.exists() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
    Path('models/selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    Path('models/README.md').write_text('''# Validation-selected ThermoRL checkpoints

These two inference bundles are selected by **validation return**, before test evaluation. `selection.json` records the source run, independent training seed and selected episode. Metadata captures features, objective, parameters and selection seeds. The main report evaluates all three independent runs rather than only these representatives.

The default GUI loads these bundles when present. Use `--models` explicitly to compare other runs. Both use the unchanged default physical environment and reward weights. DQN derives 21 comfort features; Q-learning uses the compact approximate state. Checkpoints support inference, not exact optimizer/replay/RNG resumption. Training and validation CSVs are real measurements.

Reproduce source training with the commands and seeds in `docs/LEARNING_IMPROVEMENTS.md` and README. Bundling: `.venv/bin/python scripts/bundle_models.py` (fresh `models/dqn` and `models/q_learning` directories required). The original 60-episode development runs are retained under `models/archive/` for reproducible comparisons and excluded from default GUI discovery.
''')


if __name__ == '__main__':
    main()
