import numpy as np
import pytest
from thermorl.env import ClassroomEnv
from thermorl.q_learning import QLearning, discretize


def test_q_learning_masks_target_and_greedy_action(tmp_path):
    env = ClassroomEnv()
    obs, _ = env.reset(seed=1)
    next_obs = obs.copy()
    next_obs[0] = 0.9
    mask = np.zeros(16, dtype=bool)
    mask[0] = True
    agent = QLearning(learning_rate=1, gamma=0.5)
    values = np.full(16, 100.0)
    values[0] = 2.0
    agent.table[discretize(next_obs)] = values
    agent.learn(obs, 0, -1, next_obs, False, mask)
    assert agent.table[discretize(obs)][0] == 0
    assert agent.act(next_obs, mask) == 0
    agent.learn(obs, 0, -3, next_obs, True, mask)
    assert agent.table[discretize(obs)][0] == -3
    path = tmp_path / "q.json"
    agent.save(path)
    loaded = QLearning.load(path)
    np.testing.assert_array_equal(loaded.table[discretize(obs)], agent.table[discretize(obs)])


def test_timetable_ablation_does_not_use_future_class_inputs():
    env = ClassroomEnv()
    obs, _ = env.reset(seed=1)
    changed = obs.copy()
    changed[4:8] = [0, 0, 1, 1]
    assert discretize(obs, False) == discretize(changed, False)


def test_dqn_masked_bellman_target_terminal_and_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from thermorl.dqn import DQN
    agent = DQN(batch_size=1, warmup=1, learning_rate=0.0, gamma=0.5)
    for parameter in agent.online.parameters():
        parameter.data.zero_()
    for parameter in agent.target.parameters():
        parameter.data.zero_()
    agent.target[-1].bias.data.fill_(100)
    agent.target[-1].bias.data[0] = 2
    obs = np.zeros(17, dtype=np.float32)
    mask = np.zeros(16, dtype=bool)
    mask[0] = True
    # Feasible next Q=2, reward=-1 => zero Bellman residual.
    assert agent.learn(obs, 0, -1, obs, False, mask) == pytest.approx(0)
    assert agent.act(obs, mask) == 0
    terminal = DQN(batch_size=1, warmup=1, replay_capacity=1, learning_rate=0.0)
    for parameter in terminal.online.parameters():
        parameter.data.zero_()
    assert terminal.learn(obs, 0, -2, obs, True, mask) == pytest.approx(1.5)
    path = tmp_path / "dqn.pt"
    terminal.save(path)
    loaded = DQN.load(path)
    with torch.no_grad():
        torch.testing.assert_close(loaded.online(torch.from_numpy(obs)), terminal.online(torch.from_numpy(obs)))


def test_dqn_timetable_ablation():
    pytest.importorskip("torch")
    from thermorl.dqn import DQN
    agent = DQN(use_timetable=False)
    obs = np.ones(17, dtype=np.float32)
    features = agent.features(obs)
    assert not features[4:8].any()
    assert obs.all()  # Features do not mutate simulator observations.
