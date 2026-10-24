"""Original CPU DQN: replay, target network, Huber loss and feasible targets."""
import numpy as np
import torch
from torch import nn


def network():
    return nn.Sequential(nn.Linear(17, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 16))


class DQN:
    def __init__(self, seed=0, learning_rate=0.001, gamma=0.99, batch_size=64,
                 replay_capacity=50000, warmup=500, target_interval=250, use_timetable=True):
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        self.rng = np.random.default_rng(seed)
        self.online, self.target = network(), network()
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.gamma, self.learning_rate = gamma, learning_rate
        self.batch_size, self.warmup = batch_size, warmup
        self.target_interval, self.use_timetable = target_interval, use_timetable
        self.capacity, self.position, self.size, self.updates = replay_capacity, 0, 0, 0
        self.states = np.empty((replay_capacity, 17), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(replay_capacity, dtype=np.int64)
        self.rewards = np.empty(replay_capacity, dtype=np.float32)
        self.dones = np.empty(replay_capacity, dtype=np.float32)
        self.masks = np.empty((replay_capacity, 16), dtype=bool)

    def features(self, observation):
        result = np.array(observation, dtype=np.float32, copy=True)
        if not self.use_timetable:
            result[..., 4:8] = 0
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

    def learn(self, observation, action, reward, next_observation, done, next_mask):
        i = self.position
        self.states[i], self.next_states[i] = self.features(observation), self.features(next_observation)
        self.actions[i], self.rewards[i], self.dones[i], self.masks[i] = action, reward, done, next_mask
        self.position, self.size = (i + 1) % self.capacity, min(self.size + 1, self.capacity)
        if self.size < max(self.warmup, self.batch_size):
            return None
        batch = self.rng.choice(self.size, self.batch_size, replace=False)
        states, next_states = torch.from_numpy(self.states[batch]), torch.from_numpy(self.next_states[batch])
        actions = torch.from_numpy(self.actions[batch])
        values = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            future = self.target(next_states).masked_fill(~torch.from_numpy(self.masks[batch]), -torch.inf).max(1).values
            targets = torch.from_numpy(self.rewards[batch]) + self.gamma * (1 - torch.from_numpy(self.dones[batch])) * future
        loss = nn.functional.smooth_l1_loss(values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save({"state_dict": self.online.state_dict(), "gamma": self.gamma,
                    "learning_rate": self.learning_rate, "use_timetable": self.use_timetable}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(learning_rate=data["learning_rate"], gamma=data["gamma"], use_timetable=data["use_timetable"])
        agent.online.load_state_dict(data["state_dict"])
        agent.target.load_state_dict(data["state_dict"])
        agent.online.eval()
        return agent
