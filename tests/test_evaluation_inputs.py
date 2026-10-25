"""Evaluation should reject invalid inputs before loading a policy."""
import json
from unittest.mock import patch

import pytest

from thermorl.config import Config
from thermorl.experiment import evaluate_main


@pytest.mark.parametrize("algorithm", ["sarsa", None])
def test_unknown_algorithm_is_rejected(tmp_path, algorithm, capsys):
    model = tmp_path / "model"
    model.mkdir()
    Config().save(model / "config.json")
    metadata = {"checkpoint": "agent.json"}
    if algorithm is not None:
        metadata["algorithm"] = algorithm
    (model / "metadata.json").write_text(json.dumps(metadata))
    output = tmp_path / "evaluation"
    with patch("thermorl.experiment.plotting"), patch("thermorl.experiment.QLearning.load") as load:
        with pytest.raises(SystemExit) as error:
            evaluate_main(["--models", str(model), "--output", str(output)])
    assert error.value.code == 2
    assert "Unsupported model algorithm" in capsys.readouterr().err
    load.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("absolute", [False, True])
def test_checkpoint_outside_model_directory_is_rejected(tmp_path, absolute, capsys):
    model = tmp_path / "model"
    model.mkdir()
    Config().save(model / "config.json")
    checkpoint = str(tmp_path / "outside.json") if absolute else "../outside.json"
    (model / "metadata.json").write_text(json.dumps({
        "algorithm": "q_learning", "checkpoint": checkpoint}))
    output = tmp_path / "evaluation"
    with patch("thermorl.experiment.plotting"), patch(
            "thermorl.experiment.QLearning.load", side_effect=AssertionError("Invalid checkpoint reached loader")) as load:
        with pytest.raises(SystemExit) as error:
            evaluate_main(["--models", str(model), "--output", str(output)])
    assert error.value.code == 2
    assert "Checkpoint must be inside" in capsys.readouterr().err
    load.assert_not_called()
    assert not output.exists()
