"""Two independent thermal zones coupled through a shared electrical cap."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .config import Config
from .scenarios import make_day

# Per-room modes: off, fan, AC, AC+fan. Joint action = 4*room0 + room1.
MODES = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
ACTIONS = np.array([[MODES[a], MODES[b]] for a in range(4) for b in range(4)])
OBSERVATION_NAMES = (
    "temperature_0", "temperature_1", "occupancy_0", "occupancy_1",
    "minutes_to_class_0", "minutes_to_class_1", "next_class_occupancy_0",
    "next_class_occupancy_1", "outdoor", "solar", "time_sin", "time_cos",
    "power_cap", "previous_ac_0", "previous_ac_1", "minutes_to_cap_change", "next_cap",
)


class ClassroomEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None, scenario="restricted"):
        self.config = config or Config()
        self.scenario = scenario
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32)
        self.horizon = 1440 // self.config.step_minutes
        self.t = 0
        self.day = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.day = make_day(self.np_random, self.config, self.scenario)
        self.t = 0
        self.temperature = self.np_random.uniform(27, 30, 2)
        if options and "temperature" in options:
            supplied = np.asarray(options["temperature"], dtype=float)
            if supplied.shape != (2,) or not np.isfinite(supplied).all():
                raise ValueError("temperature must contain two finite values")
            self.temperature = supplied.copy()
        self.previous_ac = np.zeros(2, dtype=int)
        self.previous_fan = np.zeros(2, dtype=int)
        return self._observation(), {"action_mask": self.action_mask()}

    def action_power(self):
        c = self.config
        return (ACTIONS[:, :, 0] * c.ac_electric_kw +
                ACTIONS[:, :, 1] * c.fan_electric_kw).sum(axis=1)

    def action_mask(self):
        return self.action_power() <= self.day.cap[self.t] + 1e-9

    def next_classes(self):
        minutes, counts = [], []
        for room in range(2):
            future = np.flatnonzero(self.day.class_start[self.t:, room])
            if future.size:
                offset = int(future[0])
                minutes.append(min(offset * self.config.step_minutes, 240))
                counts.append(self.day.occupancy[self.t + offset, room])
            else:
                minutes.append(240)
                counts.append(0)
        return np.array(minutes), np.array(counts)

    def _observation(self):
        minutes, counts = self.next_classes()
        changes = np.flatnonzero(self.day.cap[self.t:] != self.day.cap[self.t])
        offset = int(changes[0]) if changes.size else None
        cap_minutes = min(offset * self.config.step_minutes, 240) if offset else 240
        next_cap = self.day.cap[self.t + offset] if offset else self.day.cap[self.t]
        angle = 2 * np.pi * self.t / self.horizon
        return np.concatenate([
            self.temperature / 50, self.day.occupancy[self.t] / 60,
            minutes / 240, counts / 60,
            [self.day.outdoor[self.t] / 50, self.day.solar[self.t] / 1000,
             np.sin(angle), np.cos(angle), self.day.cap[self.t] / 8],
            self.previous_ac, [cap_minutes / 240, next_cap / 8],
        ]).astype(np.float32)

    def step(self, action):
        if self.day is None or self.t >= self.horizon:
            raise RuntimeError("Call reset before stepping a new episode")
        if not self.action_space.contains(action):
            raise ValueError("Action must be an integer from 0 to 15")
        if not self.action_mask()[action]:
            raise ValueError("Action exceeds the current power cap")
        c = self.config
        dt = c.step_minutes / 60
        temperature_before = self.temperature.copy()
        arrival_proxy = temperature_before - self.previous_fan * c.fan_comfort_offset_c
        arrival_violation = (np.maximum(c.comfort_low_c - arrival_proxy, 0) +
                             np.maximum(arrival_proxy - c.comfort_high_c, 0))
        equipment = ACTIONS[action]
        ac, fan = equipment[:, 0], equipment[:, 1]
        occupants = self.day.occupancy[self.t].copy()
        heat = np.array(c.conductance_kw_per_c) * (self.day.outdoor[self.t] - self.temperature)
        heat += np.array(c.solar_gain_kw_per_wm2) * self.day.solar[self.t]
        heat += occupants * c.person_heat_kw + fan * c.fan_electric_kw
        heat -= ac * c.ac_cooling_kw
        self.temperature += dt * heat / np.array(c.thermal_capacity_kwh_per_c)
        effective = self.temperature - fan * c.fan_comfort_offset_c
        violation = np.maximum(c.comfort_low_c - effective, 0) + np.maximum(effective - c.comfort_high_c, 0)
        occupied = occupants > 0
        energy = float(self.action_power()[action] * dt)
        discomfort = float(np.sum((occupants / 60) * violation ** 2) * dt)
        worst = float(np.max(np.where(occupied, violation ** 2, 0)) * dt)
        switches = int(np.abs(ac - self.previous_ac).sum())
        reward = -(c.energy_weight * energy + c.discomfort_weight * discomfort +
                   c.fairness_weight * worst + c.switching_weight * switches)
        # Metrics use the occupants present during this transition, not next interval's class.
        info = {
            "minute": self.t * c.step_minutes, "temperature": self.temperature.copy(),
            "temperature_before": temperature_before,
            "arrival_comfort_proxy": arrival_proxy, "arrival_violation": arrival_violation,
            "effective_temperature": effective.copy(),
            "overheated": effective > c.comfort_high_c, "overcooled": effective < c.comfort_low_c, "outdoor": float(self.day.outdoor[self.t]),
            "occupancy": occupants, "class_start": self.day.class_start[self.t].copy(),
            "violation": violation, "energy_kwh": energy, "power_kw": float(self.action_power()[action]),
            "cap_kw": float(self.day.cap[self.t]), "switches": switches, "action": int(action),
            "reward_energy": c.energy_weight * energy,
            "reward_discomfort": c.discomfort_weight * discomfort,
            "reward_fairness": c.fairness_weight * worst,
            "reward_switching": c.switching_weight * switches,
        }
        self.previous_ac = ac.copy()
        self.previous_fan = fan.copy()
        self.t += 1
        # Day boundary is a finite task horizon, not a physical absorbing state.
        return self._observation(), float(reward), False, self.t == self.horizon, info | {"action_mask": self.action_mask()}
