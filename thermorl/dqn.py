"""Original CPU DQN: replay, target network, Huber loss and feasible targets."""
import numpy as np
import torch
from torch import nn


def network(inputs=17):
    return nn.Sequential(nn.Linear(inputs, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 16))


class DQN:
    def __init__(self, seed=0, learning_rate=0.001, gamma=0.99, batch_size=64,
                 replay_capacity=50000, warmup=500, target_interval=250, use_timetable=True,
                 feature_mode="legacy", comfort_low=23.0, comfort_high=26.0, reward_scale=1.0,
                 train_frequency=1):
        # Reject settings that prevent learning or cause invalid replay sampling.
        if not np.isfinite(learning_rate) or learning_rate <= 0 or not 0 <= gamma <= 1:
            raise ValueError("learning_rate must be finite and positive; gamma must be in [0, 1]")
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in
               (batch_size, replay_capacity, warmup, target_interval, train_frequency)):
            raise ValueError("Replay sizes, warmup, and target_interval must be integers")
        if batch_size < 1 or warmup < 0 or replay_capacity < max(batch_size, warmup) or target_interval < 1 or train_frequency < 1:
            raise ValueError("Replay capacity must cover batch_size and warmup; batch and target_interval must be positive")
        if feature_mode not in ("legacy", "comfort_v1"):
            raise ValueError("Unknown DQN feature mode")
        if not np.isfinite(reward_scale) or reward_scale <= 0 or comfort_low >= comfort_high:
            raise ValueError("Reward scale must be positive and comfort limits ordered")
        self.feature_mode, self.reward_scale = feature_mode, reward_scale
        self.comfort_low, self.comfort_high = comfort_low, comfort_high
        self.input_size = 21 if feature_mode == "comfort_v1" else 17
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        self.rng = np.random.default_rng(seed)
        self.online, self.target = network(self.input_size), network(self.input_size)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.gamma, self.learning_rate = gamma, learning_rate
        self.batch_size, self.warmup = batch_size, warmup
        self.target_interval, self.use_timetable = target_interval, use_timetable
        self.train_frequency, self.transitions = train_frequency, 0
        self.capacity, self.position, self.size, self.updates = replay_capacity, 0, 0, 0
        self.states = np.empty((replay_capacity, self.input_size), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(replay_capacity, dtype=np.int64)
        self.rewards = np.empty(replay_capacity, dtype=np.float32)
        self.dones = np.empty(replay_capacity, dtype=np.float32)
        self.masks = np.empty((replay_capacity, 16), dtype=bool)
        self.scenarios = np.full(replay_capacity, 'unknown', dtype='<U32')
        self.last_learning_stats = {}

    def features(self, observation):
        result = np.array(observation, dtype=np.float32, copy=True)
        if not self.use_timetable:
            result[..., 4:8] = 0
        if self.feature_mode == "comfort_v1":
            temperatures = result[..., :2] * 50
            centre = (self.comfort_low + self.comfort_high) / 2
            cold = np.maximum(self.comfort_low - temperatures, 0) / 5
            hot = np.maximum(temperatures - self.comfort_high, 0) / 5
            result[..., :2] = (temperatures - centre) / 5
            result[..., 8] = (result[..., 8] * 50 - centre) / 10
            result = np.concatenate([result, cold, hot], axis=-1).astype(np.float32)
        return result

    def act(self, observation, mask, epsilon=0.0):
        # An all-false mask must never silently select action zero.
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (16,) or not mask.any():
            raise ValueError("Action mask must contain 16 entries and a feasible action")
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))
        with torch.no_grad():
            values = self.online(torch.from_numpy(self.features(observation))).numpy()
        return int(np.argmax(np.where(mask, values, -np.inf)))

    def learn(self, observation, action, reward, next_observation, done, next_mask, *, scenario=None):
        next_mask = np.asarray(next_mask, dtype=bool)
        if next_mask.shape != (16,) or (not done and not next_mask.any()):
            raise ValueError("Next mask needs 16 entries and a feasible action unless done")
        i = self.position
        self.states[i], self.next_states[i] = self.features(observation), self.features(next_observation)
        self.actions[i], self.rewards[i], self.dones[i], self.masks[i] = action, reward / self.reward_scale, done, next_mask
        self.scenarios[i] = scenario or 'unknown'
        self.position, self.size = (i + 1) % self.capacity, min(self.size + 1, self.capacity)
        self.transitions += 1
        if self.size < max(self.warmup, self.batch_size) or self.transitions % self.train_frequency:
            return None
        batch = self.rng.choice(self.size, self.batch_size, replace=False)
        states, next_states = torch.from_numpy(self.states[batch]), torch.from_numpy(self.next_states[batch])
        actions = torch.from_numpy(self.actions[batch])
        values = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            future = self.target(next_states).masked_fill(~torch.from_numpy(self.masks[batch]), -torch.inf).max(1).values
            # Ended days have no future value. Set it to zero before arithmetic:
            # multiplying an empty-mask maximum (-inf) by zero would produce NaN.
            future = future.masked_fill(torch.from_numpy(self.dones[batch]).bool(), 0.0)
            targets = torch.from_numpy(self.rewards[batch]) + self.gamma * future
        loss = nn.functional.smooth_l1_loss(values, targets)
        self.last_learning_stats = {'td_target_mean': float(targets.mean()),
            'td_target_abs_max': float(targets.abs().max()), 'q_mean': float(values.detach().mean()),
            'q_abs_max': float(values.detach().abs().max())}
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def training_diagnostics(self):
        """Read replay composition and last minibatch statistics without sampling."""
        from .scenarios import SCENARIOS
        occupied = (self.states[:self.size, 2:4] > 0).any(axis=1)
        soon = ((self.states[:self.size, 4:6] <= 30 / 240) &
                (self.states[:self.size, 6:8] > 0)).any(axis=1)
        return {'replay_size': self.size,
            'replay_occupied_transition_pct': float(100 * occupied.mean()) if self.size else 0.,
            'replay_arrival_soon_pct': float(100 * soon.mean()) if self.size else 0.,
            **{'replay_' + s + '_pct': float(100 * (self.scenarios[:self.size] == s).mean()) if self.size else 0.
               for s in SCENARIOS},
            **{k: self.last_learning_stats.get(k) for k in ('td_target_mean', 'td_target_abs_max', 'q_mean', 'q_abs_max')}}

    def save(self, path):
        torch.save({"state_dict": self.online.state_dict(), "gamma": self.gamma,
                    "learning_rate": self.learning_rate, "use_timetable": self.use_timetable,
                    "feature_mode": self.feature_mode, "reward_scale": self.reward_scale,
                    "batch_size": self.batch_size, "warmup": self.warmup, "replay_capacity": self.capacity,
                    "target_interval": self.target_interval, "train_frequency": self.train_frequency,
                    "comfort_low": self.comfort_low, "comfort_high": self.comfort_high}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(learning_rate=data["learning_rate"], gamma=data["gamma"], use_timetable=data["use_timetable"],
                    feature_mode=data.get("feature_mode", "legacy"), reward_scale=data.get("reward_scale", 1.0),
                    comfort_low=data.get("comfort_low", 23.0), comfort_high=data.get("comfort_high", 26.0),
                    batch_size=data.get('batch_size', 64), warmup=data.get('warmup', 500),
                    replay_capacity=data.get('replay_capacity', 50000), target_interval=data.get('target_interval', 250),
                    train_frequency=data.get('train_frequency', 1))
        agent.online.load_state_dict(data["state_dict"])
        agent.target.load_state_dict(data["state_dict"])
        agent.online.eval()
        return agent
