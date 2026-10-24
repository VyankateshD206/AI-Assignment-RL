from dataclasses import replace
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from thermorl.config import Config
from thermorl.env import ACTIONS, ClassroomEnv
from thermorl.experiment import rollout
from thermorl.baselines import RuleController


def test_gym_contract():
    check_env(ClassroomEnv(), skip_render_check=True)


def test_seeded_days_and_transitions_reproduce():
    a, b = ClassroomEnv(), ClassroomEnv()
    np.testing.assert_array_equal(a.reset(seed=42)[0], b.reset(seed=42)[0])
    for _ in range(a.horizon):
        action = int(np.flatnonzero(a.action_mask())[-1])
        x, y = a.step(action), b.step(action)
        np.testing.assert_array_equal(x[0], y[0])
        assert x[1:4] == y[1:4]


def test_power_cap_and_rejected_action_does_not_advance():
    env = ClassroomEnv()
    env.reset(seed=0)
    env.t = 12 * 60 // env.config.step_minutes
    before = env.temperature.copy()
    assert not env.action_mask()[15]
    with pytest.raises(ValueError, match="power cap"):
        env.step(15)
    np.testing.assert_array_equal(env.temperature, before)
    assert env.t == 144
    for action in np.flatnonzero(env.action_mask()):
        assert env.action_power()[action] <= env.day.cap[env.t]


def test_ac_cools_and_energy_has_correct_units():
    off, on = ClassroomEnv(), ClassroomEnv()
    for env in (off, on):
        env.reset(seed=9, options={"temperature": [30, 30]})
    off.step(0)
    _, _, _, _, info = on.step(10)  # AC only in both rooms.
    assert np.all(on.temperature < off.temperature)
    assert info["energy_kwh"] == pytest.approx(6 * 5 / 60)
    assert info["switches"] == 2


def test_fans_add_heat_but_improve_comfort_proxy():
    off, fan = ClassroomEnv(), ClassroomEnv()
    off.reset(seed=0)
    fan.reset(seed=0)
    off.step(0)
    _, _, _, _, info = fan.step(5)
    assert np.all(fan.temperature > off.temperature)
    np.testing.assert_allclose(info["effective_temperature"], fan.temperature - 1.5)


def test_empty_room_has_no_discomfort_reward():
    env = ClassroomEnv()
    env.reset(seed=1, options={"temperature": [40, 40]})
    _, reward, _, _, info = env.step(0)
    assert info["reward_discomfort"] == info["reward_fairness"] == 0
    assert reward == 0


def test_reward_terms_sum_and_episode_boundary():
    env = ClassroomEnv()
    env.reset(seed=2)
    for _ in range(env.horizon):
        _, reward, terminated, truncated, info = env.step(0)
        assert reward == pytest.approx(-sum(info[k] for k in (
            "reward_energy", "reward_discomfort", "reward_fairness", "reward_switching")))
        assert not terminated
    assert truncated
    with pytest.raises(RuntimeError):
        env.step(0)


@pytest.mark.parametrize("name", ["thermostat", "timetable", "rotation"])
def test_baseline_completes_restricted_day_with_no_cap_violations(name):
    metrics, records = rollout(ClassroomEnv(), RuleController(name), seed=2_000_000)
    assert len(records) == 288
    assert metrics["cap_violations"] == 0
    assert metrics["energy_kwh"] > 0
    assert np.isfinite(list(metrics.values())).all()


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        replace(Config(), step_minutes=7)
    with pytest.raises(ValueError):
        replace(Config(), thermal_capacity_kwh_per_c=(0, 1))
