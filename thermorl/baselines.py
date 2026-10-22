"""Controllers using only the same normalized observations as the RL agents."""
import numpy as np
from .env import ACTIONS


class RuleController:
    def __init__(self, name):
        if name not in ("thermostat", "timetable", "rotation"):
            raise ValueError(name)
        self.name = name

    def act(self, observation, mask, **kwargs):
        o = observation
        temperature, occupancy = o[:2] * 50, o[2:4] * 60
        minutes, next_count = o[4:6] * 240, o[6:8] * 60
        previous = o[13:15] > 0.5
        occupied = occupancy > 0
        fan = occupied & (temperature > 25)
        # Hysteresis prevents the rule from chattering around a single threshold.
        desired_ac = occupied & (temperature > np.where(previous, 24.5, 26))
        priority = np.maximum(temperature - 24.5, 0) * np.maximum(occupancy, 1) / 60
        if self.name == "timetable":
            upcoming = (~occupied) & (next_count > 0) & (minutes <= 30)
            desired_ac |= upcoming & (temperature > 24.5)
            priority += upcoming * np.maximum(temperature - 24.5, 0) * (next_count / 60)
        if self.name == "rotation":
            angle = np.arctan2(o[10], o[11]) % (2 * np.pi)
            minute = round(float(angle * 1440 / (2 * np.pi))) % 1440
            preferred = (minute // 15) % 2
            priority = np.where(np.arange(2) == preferred, 2.0, 1.0)
        # Select the feasible combination with the fewest weighted missed requests.
        scores = []
        for action in np.flatnonzero(mask):
            ac, fans = ACTIONS[action, :, 0], ACTIONS[action, :, 1]
            missed = np.sum(desired_ac * (1 - ac) * (1 + priority))
            unwanted = 10 * np.sum(ac * (~desired_ac))
            fan_mismatch = 0.1 * np.sum(np.abs(fans - fan.astype(int)))
            scores.append((missed + unwanted + fan_mismatch, int(action)))
        return min(scores)[1]
