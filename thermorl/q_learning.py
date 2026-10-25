"""Sparse tabular Q-learning, implemented directly with NumPy."""
import json
from pathlib import Path
import numpy as np


def discretize(o, use_timetable=True, state_mode="legacy", comfort_low=23.0, comfort_high=26.0):
    """Deliberately coarse abstraction; continuous observations are not exactly Markov."""
    if state_mode in ("compact_v1", "comfort_v2"):
        edges = [comfort_low - 2, comfort_low, comfort_high, comfort_high + 2, comfort_high + 5]
        if state_mode == 'comfort_v2':
            edges.insert(2, (comfort_low + comfort_high) / 2)
        temperatures = np.digitize(o[:2] * 50, edges)
        occupied = o[2:4] > 0
        minutes = o[4:6] * 240
        soon = ((minutes <= 30) & (o[6:8] > 0)).astype(int) if use_timetable else np.zeros(2, dtype=int)
        if state_mode == 'comfort_v2' and use_timetable:
            soon += ((minutes <= 60) & (o[6:8] > 0)).astype(int)
        values = [*temperatures, *occupied, *soon, int(o[12] * 8 < 5), *np.rint(o[13:15]).astype(int)]
        return ":".join(str(int(v)) for v in values)
    if state_mode != "legacy":
        raise ValueError("Unknown tabular state mode")
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
    def __init__(self, seed=0, learning_rate=0.15, gamma=0.99, use_timetable=True,
                 state_mode="legacy", comfort_low=23.0, comfort_high=26.0,
                 visit_decay=0.0, count_exploration=False):
        # Keep the tabular update a weighted average of the old value and target.
        if not 0 < learning_rate <= 1 or not 0 <= gamma <= 1:
            raise ValueError("learning_rate must be in (0, 1] and gamma in [0, 1]")
        if state_mode not in ("legacy", "compact_v1", "comfort_v2") or comfort_low >= comfort_high:
            raise ValueError("Invalid tabular state mode or comfort limits")
        if not np.isfinite(visit_decay) or visit_decay < 0:
            raise ValueError('visit_decay must be finite and nonnegative')
        self.visit_decay, self.count_exploration = visit_decay, count_exploration
        self.action_visits = {}
        self.state_mode, self.comfort_low, self.comfort_high = state_mode, comfort_low, comfort_high
        self.rng = np.random.default_rng(seed)
        self.learning_rate, self.gamma = learning_rate, gamma
        self.use_timetable = use_timetable
        self.table = {}
        self.visits = 0

    def state_key(self, observation):
        return discretize(observation, self.use_timetable, self.state_mode, self.comfort_low, self.comfort_high)

    def act(self, observation, mask, epsilon=0.0):
        # At least one action must be allowed by the power cap.
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (16,) or not mask.any():
            raise ValueError("Action mask must contain 16 entries and a feasible action")
        valid = np.flatnonzero(mask)
        if epsilon > 0 and self.rng.random() < epsilon:
            if self.count_exploration:
                counts = self.action_visits.get(self.state_key(observation), np.zeros(16, dtype=int))
                valid = valid[counts[valid] == counts[valid].min()]
            return int(self.rng.choice(valid))
        values = self.table.get(self.state_key(observation), np.zeros(16))
        # Deterministic greedy evaluation; random tie-breaking only while exploring.
        best = valid[values[valid] == np.max(values[valid])]
        return int(self.rng.choice(best)) if epsilon > 0 else int(best[0])

    def learn(self, observation, action, reward, next_observation, done, next_mask):
        target = reward
        if not done:
            # Convert 0/1 flags before NumPy indexing; terminal days do not bootstrap.
            next_mask = np.asarray(next_mask, dtype=bool)
            if next_mask.shape != (16,) or not next_mask.any():
                raise ValueError("Next mask must contain 16 entries and a feasible action")
            future = self.table.get(self.state_key(next_observation), np.zeros(16))
            target += self.gamma * float(np.max(future[next_mask]))
        key = self.state_key(observation)
        values = self.table.setdefault(key, np.zeros(16, dtype=np.float64))
        counts = self.action_visits.setdefault(key, np.zeros(16, dtype=np.int64))
        alpha = self.learning_rate / np.sqrt(1 + counts[action] / self.visit_decay) if self.visit_decay else self.learning_rate
        error = target - values[action]
        values[action] += alpha * error
        counts[action] += 1
        self.visits += 1
        return float(error ** 2)

    def save(self, path):
        payload = {"algorithm": "q_learning", "learning_rate": self.learning_rate,
                   "gamma": self.gamma, "use_timetable": self.use_timetable,
                   "state_mode": self.state_mode, "comfort_low": self.comfort_low, "comfort_high": self.comfort_high,
                   "visit_decay": self.visit_decay, "count_exploration": self.count_exploration,
                   "action_visits": {k: v.tolist() for k, v in self.action_visits.items()},
                   "visits": self.visits, "table": {k: v.tolist() for k, v in self.table.items()}}
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        agent = cls(learning_rate=data["learning_rate"], gamma=data["gamma"], use_timetable=data["use_timetable"],
                    state_mode=data.get("state_mode", "legacy"), comfort_low=data.get("comfort_low", 23.0),
                    comfort_high=data.get("comfort_high", 26.0),
                    visit_decay=data.get('visit_decay', 0.0), count_exploration=data.get('count_exploration', False))
        agent.table = {k: np.array(v) for k, v in data["table"].items()}
        agent.action_visits = {k: np.array(v, dtype=np.int64) for k, v in data.get('action_visits', {}).items()}
        agent.visits = data["visits"]
        return agent
