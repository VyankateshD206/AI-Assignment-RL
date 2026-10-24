import csv
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
