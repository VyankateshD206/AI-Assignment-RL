"""Explicit simulator assumptions; these are not calibrated building measurements."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    step_minutes: int = 5
    thermal_capacity_kwh_per_c: tuple[float, float] = (1.5, 1.8)
    conductance_kw_per_c: tuple[float, float] = (0.16, 0.14)
    solar_gain_kw_per_wm2: tuple[float, float] = (0.003, 0.0015)
    person_heat_kw: float = 0.075
    ac_cooling_kw: float = 12.0
    ac_electric_kw: float = 3.0
    fan_electric_kw: float = 0.12
    fan_comfort_offset_c: float = 1.5
    comfort_low_c: float = 23.0
    comfort_high_c: float = 26.0
    normal_cap_kw: float = 6.5
    restricted_cap_kw: float = 3.3
    energy_weight: float = 0.3
    discomfort_weight: float = 2.0
    fairness_weight: float = 1.0
    switching_weight: float = 0.02

    def __post_init__(self):
        if self.step_minutes <= 0 or 1440 % self.step_minutes:
            raise ValueError("step_minutes must be positive and divide a day")
        if self.step_minutes > 15:
            raise ValueError("Use steps of at most 15 minutes for the thermal integrator")
        for name in ("thermal_capacity_kwh_per_c", "conductance_kw_per_c", "solar_gain_kw_per_wm2"):
            values = tuple(getattr(self, name))
            object.__setattr__(self, name, values)
            if len(values) != 2 or any(v <= 0 for v in values):
                raise ValueError(f"{name} must contain two positive values")
        if self.comfort_low_c >= self.comfort_high_c:
            raise ValueError("Invalid comfort band")
        if min(self.ac_cooling_kw, self.ac_electric_kw, self.fan_electric_kw) <= 0:
            raise ValueError("Equipment powers must be positive")
        if min(self.normal_cap_kw, self.restricted_cap_kw, self.person_heat_kw,
               self.fan_comfort_offset_c, self.energy_weight, self.discomfort_weight,
               self.fairness_weight, self.switching_weight) < 0:
            raise ValueError("Caps, gains, and reward weights must be nonnegative")

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path=None):
        return cls(**json.loads(Path(path).read_text())) if path else cls()
