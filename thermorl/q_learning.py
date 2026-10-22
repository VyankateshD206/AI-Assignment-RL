"""Sparse tabular Q-learning, implemented directly with NumPy."""
import json
from pathlib import Path
import numpy as np


def discretize(o, use_timetable=True):
    """Deliberately coarse abstraction; continuous observations are not exactly Markov."""
    temperatures = np.digitize(o[:2] * 50, [23, 26, 29, 32, 35])
    occupancy = np.digitize(o[2:4] * 60, [1, 35])
    arrival = np.digitize(o[4:6] * 240, [15, 45]) if use_timetable else [0, 0]
    next_count = (o[6:8] > 0).astype(int) if use_timetable else [0, 0]
    hour = (np.arctan2(o[10], o[11]) % (2 * np.pi)) * 12 / np.pi
    values = [*temperatures, *occupancy, *arrival, *next_count,
              int(np.digitize(o[8] * 50, [30, 38])), int(o[9] > 0.4),
              int(hour // 3), int(o[12] * 8 < 5), *np.rint(o[13:15]).astype(int),
              int(o[15] * 240 <= 30), int(o[16] * 8 < 5)]
    return ":".join(str(int(v)) for v in values)


class QLearning:
    def __init__(self, seed=0, learning_rate=0.15, gamma=0.99, use_timetable=True):
        self.rng = np.random.default_rng(seed)
        self.learning_rate, self.gamma = learning_rate, gamma
        self.use_timetable = use_timetable
        self.table = {}
        self.visits = 0

    def act(self, observation, mask, epsilon=0.0):
        valid = np.flatnonzero(mask)
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        values = self.table.get(discretize(observation, self.use_timetable), np.zeros(16))
        # Deterministic greedy evaluation; random tie-breaking only while exploring.
        best = valid[values[valid] == np.max(values[valid])]
        return int(self.rng.choice(best)) if epsilon > 0 else int(best[0])

    def learn(self, observation, action, reward, next_observation, done, next_mask):
        key = discretize(observation, self.use_timetable)
        values = self.table.setdefault(key, np.zeros(16, dtype=np.float64))
        future = self.table.get(discretize(next_observation, self.use_timetable), np.zeros(16))
        target = reward + (0 if done else self.gamma * float(np.max(future[next_mask])))
        error = target - values[action]
        values[action] += self.learning_rate * error
        self.visits += 1
        return float(error ** 2)

    def save(self, path):
        payload = {"algorithm": "q_learning", "learning_rate": self.learning_rate,
                   "gamma": self.gamma, "use_timetable": self.use_timetable,
                   "visits": self.visits, "table": {k: v.tolist() for k, v in self.table.items()}}
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        agent = cls(learning_rate=data["learning_rate"], gamma=data["gamma"], use_timetable=data["use_timetable"])
        agent.table = {k: np.array(v) for k, v in data["table"].items()}
        agent.visits = data["visits"]
        return agent
