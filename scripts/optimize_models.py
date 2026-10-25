"""Select improved representatives using validation only; optionally promote bundles.

Run after three completed full runs per algorithm, before inspecting fresh tests.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil

from thermorl.config import Config
from thermorl.experiment import checkpoint_rank


def improves_validation(candidate, current):
    """Never trade away any comfort measure to obtain lower energy."""
    directions = {'mean_comfort_pct': 1, 'mean_energy_kwh': -1,
                  'mean_arrival_discomfort_pct': -1, 'mean_class_start_violation_pct': -1}
    differences = []
    for key, direction in directions.items():
        values = [candidate.get(key), current.get(key)]
        if any(value is None or not math.isfinite(value) for value in values):
            return False  # Legacy/missing evidence cannot prove non-regression.
        differences.append(direction * (values[0] - values[1]))
    return all(value >= -1e-9 for value in differences) and any(value > 1e-9 for value in differences)


def validate_protocol(candidates, current):
    """Compare paired validation scores only; checkpoint intervals may differ."""
    fields = ('scenarios', 'seeds', 'days_per_scenario')
    reference = current.get('validation', {})
    if any(key not in reference for key in fields):
        raise ValueError('Current validation protocol is incomplete')
    for candidate in candidates:
        protocol = candidate.get('validation', {})
        if any(protocol.get(key) != reference[key] for key in fields):
            raise ValueError('Candidates and current model must use identical validation scenarios and days')
        if candidate.get('comfort_floor') != candidates[0].get('comfort_floor'):
            raise ValueError('Candidate comfort targets must match')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--promote', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Manifest already exists; preserve it and choose a new output path')
    runs = [(p, json.loads((p / 'metadata.json').read_text())) for p in args.runs]
    if any(not m.get('training_complete') or m.get('selection_mode') != 'comfort_energy' for _, m in runs):
        parser.error('Requires completed comfort_energy runs')
    manifest = {'selection_data': 'Saved validation only; fresh test not inspected', 'algorithms': {}}
    selections = []
    for algorithm in ['dqn', 'q_learning']:
        candidates = [(p, m) for p, m in runs if m['algorithm'] == algorithm]
        if len(candidates) != 3 or {m['seed'] for _, m in candidates} != {0, 1, 2}:
            parser.error(f'Requires exactly seeds 0,1,2 for {algorithm}')
        bundle = Path('models') / algorithm
        old = json.loads((bundle / 'metadata.json').read_text())
        try:
            validate_protocol([m for _, m in candidates], old)
        except ValueError as error:
            parser.error(str(error))
        physical = asdict(Config.load(bundle / 'config.json'))
        for path, _ in candidates:
            settings = asdict(Config.load(path / 'config.json'))
            if settings != physical:
                parser.error('Candidate physical configuration or reward differs from the current bundle')
        eligible = [(p, m) for p, m in candidates if improves_validation(m['selected_validation'], old['selected_validation'])]
        best_candidate = max(candidates, key=lambda x: (*checkpoint_rank(x[1]['selected_validation'], 'comfort_energy', x[1]['comfort_floor']), -x[1]['seed']))
        selected = max(eligible, key=lambda x: (*checkpoint_rank(x[1]['selected_validation'], 'comfort_energy', x[1]['comfort_floor']), -x[1]['seed'])) if eligible else None
        item = {'previous_validation': old['selected_validation'],
                'previous_source_run': old.get('source_run'),
                'best_candidate_run': str(best_candidate[0]),
                'candidates': {str(p): {'seed': m['seed'], 'episode': m['selected_episode'], 'validation': m['selected_validation']} for p, m in candidates},
                'selected_run': str(selected[0]) if selected else None,
                'decision': 'Validation energy and all three comfort measures are non-regressing' if selected else 'Keep current bundle: complete non-regression evidence is missing or a measure worsens',
                'promoted': bool(args.promote and selected)}
        manifest['algorithms'][algorithm] = item
        if selected and args.promote:
            path, metadata = selected
            archive = Path('models/archive') / (algorithm + '_before_optimization')
            names = [metadata['checkpoint'], 'config.json', 'training.csv', 'validation.csv', 'training_curve.png']
            if archive.exists() or any(not (path / name).is_file() for name in names):
                parser.error('Archive already exists or candidate bundle is incomplete')
            selections.append((algorithm, bundle, archive, path, metadata, names))
    # Validate both algorithms before mutating either bundle.
    root_selection = json.loads(Path('models/selection.json').read_text())
    if selections:
        selection_archive = Path('models/archive/selection_before_optimization.json')
        if selection_archive.exists():
            parser.error('Original selection manifest is already archived; preserve it')
        selection_archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile('models/selection.json', selection_archive)
    for algorithm, bundle, archive, source, metadata, names in selections:
        archive.parent.mkdir(parents=True, exist_ok=True)
        bundle.rename(archive)
        bundle.mkdir()
        for name in names:
            shutil.copyfile(source / name, bundle / name)
        metadata = dict(metadata, source_run=str(source),
                        bundle_selection='Validation-only energy and all three comfort measures are non-regressing; no fresh test used')
        (bundle / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        root_selection[algorithm] = {'source_run': str(source), 'seed': metadata['seed'],
            'episode': metadata['selected_episode'], 'validation_return': metadata['selected_validation']['mean_return'],
            'selection': metadata['bundle_selection'], 'previous_bundle': str(archive)}
    if selections:
        Path('models/selection.json').write_text(json.dumps(root_selection, indent=2) + '\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: {'selected_run': v['selected_run'], 'promoted': v['promoted']} for k, v in manifest['algorithms'].items()}))


if __name__ == '__main__':
    main()
