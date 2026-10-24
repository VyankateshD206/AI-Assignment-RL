import numpy as np
import pytest
from thermorl.metrics import summarize


def test_metrics_ignore_empty_rooms_and_measure_class_start():
    records = []
    for t in range(4):
        records.append({"occupancy": np.array([10, 0]), "violation": np.array([2., 100.]),
                        "class_start": np.array([t == 0, False]), "energy_kwh": 0.5,
                        "switches": int(t == 0), "power_kw": 6., "cap_kw": 6.5})
    metrics = summarize(records, 5)
    assert metrics["energy_kwh"] == 2
    assert metrics["student_degree_hours"] == pytest.approx(2 * 10 * 4 / 12)
    assert metrics["occupied_room_degree_hours"] == pytest.approx(2 * 4 / 12)
    assert metrics["mean_student_discomfort_c"] == 2
    assert metrics["worst_room_mean_discomfort_c"] == 2
    assert metrics["class_start_violation_pct"] == 100
    assert metrics["room_discomfort_gap_c"] == 0
