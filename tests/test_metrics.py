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


def sample(occupied=(10, 0), starts=(False, False), violation=(0, 0), arrival=(0, 0)):
    return {"occupancy": np.array(occupied), "class_start": np.array(starts),
            "violation": np.array(violation), "arrival_violation": np.array(arrival),
            "energy_kwh": 0.5, "switches": 0, "power_kw": 6., "cap_kw": 6.5}


def test_arrivals_count_explicit_simultaneous_and_back_to_back_starts_not_cancellations():
    records = [sample((10, 20), (True, True), arrival=(1, 0)),
               sample((10, 20), (True, False), arrival=(0, 0)),
               sample((0, 20), (True, False), arrival=(1, 0))]
    m = summarize(records, 5)
    assert m['class_arrivals'] == 3
    assert m['arrival_measurements'] == 3
    assert m['uncomfortable_arrivals'] == 1
    assert m['arrival_discomfort_pct'] == pytest.approx(100 / 3)


def test_early_class_window_weights_only_fifteen_minutes():
    records = [sample(starts=(True, False)), sample(violation=(1, 0)), sample(violation=(1, 0))]
    m = summarize(records, 10)
    assert m['early_class_occupied_minutes'] == 15
    assert m['early_class_uncomfortable_minutes'] == 5
    assert m['class_start_violation_pct'] == pytest.approx(100 / 3)
    assert m['occupied_room_minutes'] == 30
    assert m['comfortable_room_minutes'] == 10


def test_early_windows_stop_at_empty_time_and_union_overlapping_starts():
    records = [sample(starts=(True, False), violation=(1, 0)),
               sample(starts=(True, False)), sample(), sample(occupied=(0, 0)),
               sample(violation=(1, 0))]
    m = summarize(records, 5)
    assert m['early_class_occupied_minutes'] == 15
    assert m['early_class_uncomfortable_minutes'] == 5


def test_empty_gap_ends_class_window_even_if_occupancy_returns_without_start():
    m = summarize([sample(starts=(True, False)), sample(occupied=(0, 0)), sample(violation=(1, 0))], 5)
    assert m['early_class_occupied_minutes'] == 5
    assert m['class_start_violation_pct'] == 0


def test_no_arrivals_and_empty_precooling_have_explicit_denominators():
    m = summarize([sample(occupied=(0, 0))], 5)
    assert m['class_arrivals'] == m['uncomfortable_arrivals'] == 0
    assert m['arrival_discomfort_pct'] is None
    assert m['early_class_occupied_minutes'] == m['occupied_room_minutes'] == 0
    assert m['energy_kwh'] == 0.5


def test_old_records_never_invent_arrival_conditions():
    record = sample(starts=(True, False))
    del record['arrival_violation']
    m = summarize([record], 5)
    assert m['class_arrivals'] == 1 and m['arrival_measurements'] == 0
    assert m['arrival_discomfort_pct'] is None
    assert m['uncomfortable_arrivals'] is None
