import csv
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("algorithm", ["q_learning", "dqn"])
def test_train_save_reload_evaluate_end_to_end(tmp_path, algorithm):
    if algorithm == "dqn":
        pytest.importorskip("torch")
    model = tmp_path / algorithm
    subprocess.run([sys.executable, "train.py", "--algo", algorithm, "--episodes", "2",
                    "--seed", "0", "--output", str(model)], cwd=ROOT, check=True,
                   capture_output=True, text=True, timeout=60)
    assert (model / "training_curve.png").stat().st_size > 0
    metadata = json.loads((model / 'metadata.json').read_text())
    with (model / 'validation.csv').open() as stream:
        validation = list(csv.DictReader(stream))
    best = max(validation, key=lambda r: float(r['mean_return']))
    assert metadata['training_complete'] is True and metadata['completed_episodes'] == 2
    assert metadata['selected_episode'] == int(best['episode'])
    assert metadata['selected_validation']['mean_return'] == float(best['mean_return'])
    assert 'heatwave' not in metadata['validation']['scenarios']
    assert all(s >= 1_000_000 for s in metadata['validation']['seeds'])
    assert (model / ('last_' + metadata['checkpoint'])).exists()
    evaluation = tmp_path / "validation"
    subprocess.run([sys.executable, "evaluate.py", "--models", str(model),
                    "--scenarios", "restricted", "--days", "1", "--split", "validation",
                    "--output", str(evaluation)], cwd=ROOT, check=True,
                   capture_output=True, text=True, timeout=60)
    with (evaluation / "daily_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert {int(row["day_seed"]) for row in rows} == {1_000_000}
    assert all(int(row["cap_violations"]) == 0 for row in rows)
    assert (evaluation / "comparison.png").stat().st_size > 0
    evaluation_metadata = json.loads((evaluation / 'metadata.json').read_text())
    assert evaluation_metadata['metric_definitions']['version'] == 'comfort_v2'
    with (evaluation / 'trace_timetable.csv').open() as stream:
        trace = list(csv.DictReader(stream))
    starts = [r for r in trace if r['class_start_0'] == '1.0']
    assert starts and all(r['arrival_comfort_proxy_0'] for r in starts)
    with (evaluation / 'diagnostics.csv').open() as stream:
        diagnostics = list(csv.DictReader(stream))
    assert len(diagnostics) == 4
    for row, diagnostic in zip(rows, diagnostics):
        penalties = sum(float(diagnostic['penalty_' + key]) for key in ('energy', 'discomfort', 'fairness', 'switching'))
        assert -float(row['return']) == pytest.approx(penalties)
        assert float(row['comfort_pct']) + float(row['overheating_pct']) + float(row['overcooling_pct']) == pytest.approx(100)
