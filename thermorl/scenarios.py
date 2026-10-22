"""Seeded synthetic days. Timetables and power restrictions are announced."""
from dataclasses import dataclass
import numpy as np
from .config import Config

SCENARIOS = ("normal", "restricted", "unequal", "schedule_changes", "heatwave")


@dataclass
class Day:
    outdoor: np.ndarray
    solar: np.ndarray
    occupancy: np.ndarray
    class_start: np.ndarray
    cap: np.ndarray


def make_day(rng, config: Config, scenario: str) -> Day:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    n = 1440 // config.step_minutes
    hour = np.arange(n + 1) * config.step_minutes / 60
    # Independent short-term disturbances around a smooth daily cycle.
    mean = rng.uniform(30, 33) + (4 if scenario == "heatwave" else 0)
    outdoor = mean + rng.uniform(6, 8) * np.cos((hour - 15) * np.pi / 12)
    outdoor += rng.normal(0, 0.25, n + 1)
    solar = np.maximum(0, np.sin((hour - 6) * np.pi / 12)) * rng.uniform(700, 950)
    solar *= rng.uniform(0.95, 1.05, n + 1)
    occupancy = np.zeros((n + 1, 2), dtype=np.float64)
    class_start = np.zeros((n + 1, 2), dtype=bool)
    slots = [[(8.5, 10), (10.5, 12), (14, 15.5), (16, 17)],
             [(9, 10.5), (11, 12.5), (13.5, 15), (15.5, 17)]]
    for room, schedule in enumerate(slots):
        for start, end in schedule:
            jitter = int(rng.integers(-2, 3)) * config.step_minutes / 60
            start, end = start + jitter, end + jitter
            if scenario == "schedule_changes":
                if rng.random() < 0.25:
                    continue  # Cancellation is known at the start of this simulated day.
                delay = float(rng.choice([0, 0.25, 0.5]))
                start, end = start + delay, end + delay
            first = int(round(start * 60 / config.step_minutes))
            last = int(round(end * 60 / config.step_minutes))
            occupancy[first:last, room] = rng.integers(20, 51)
            class_start[first, room] = True
    cap = np.full(n + 1, config.normal_cap_kw)
    if scenario != "normal":
        cap[(hour >= 11) & (hour < 16)] = config.restricted_cap_kw
    if scenario == "unequal":
        solar *= 1.2
    return Day(outdoor, solar, occupancy, class_start, cap)
