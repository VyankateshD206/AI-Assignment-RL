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
