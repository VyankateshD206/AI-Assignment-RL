import pytest
from scripts.summarize_comfort import aggregate_rows


def row(controller, arrivals, bad, early=15, early_bad=5, occupied=60, comfortable=30):
    return dict(controller=controller, scenario='restricted', class_arrivals=arrivals,
                arrival_measurements=arrivals, uncomfortable_arrivals=bad,
                arrival_discomfort_pct=100 * bad / arrivals if arrivals else None,
                early_class_occupied_minutes=early, early_class_uncomfortable_minutes=early_bad,
                occupied_room_minutes=occupied, comfortable_room_minutes=comfortable,
                class_start_violation_pct=100 * early_bad / early if early else 0,
                comfort_pct=100 * comfortable / occupied if occupied else 0,
                energy_kwh=20, cap_violations=0)


def test_report_pools_events_before_aggregating_independent_training_runs():
    rows = [row('dqn_comfort_seed0', 1, 1), row('dqn_comfort_seed0', 3, 1),
            row('dqn_comfort_seed1', 3, 0), row('dqn_comfort_seed1', 1, 0)]
    report = aggregate_rows(rows, {'DQN': ['dqn_comfort_seed0', 'dqn_comfort_seed1']})[0]
    assert report['arrival_discomfort_pct_mean'] == 25  # Per-run pooled values: 50%, 0%.
    assert report['arrival_discomfort_pct_seed_std'] == pytest.approx(35.355339)
    assert report['class_start_violation_pct_mean'] == pytest.approx(100 / 3)
    assert report['comfort_pct_mean'] == 50
    assert report['days_per_run'] == 2
    assert report['arrival_discomfort_pct_day_std_mean'] > 0


def test_report_no_arrivals_is_not_perfect_comfort():
    report = aggregate_rows([row('timetable', 0, 0, 0, 0, 0, 0)], {'Timetable': ['timetable']})[0]
    assert report['arrival_discomfort_pct_mean'] is None
    assert report['class_start_violation_pct_mean'] is None
    assert report['comfort_pct_mean'] is None
    assert report['energy_kwh_mean'] == 20
