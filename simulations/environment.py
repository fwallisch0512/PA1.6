import math
from dataclasses import dataclass

@dataclass
class Environment:
    base: float = 5.0
    amplitude: float = 2.0
    period_s: float = 7200.0
    door_drop_C: float = 5.0
    door_start_s: float = 1200.0
    door_duration_s: float = 120.0


    def T_out(self, t: float) -> float:
        temperature = self.base + self.amplitude * math.sin(
            2 * math.pi * t / self.period_s
        )

        door_is_open = self.door_start_s <= t < self.door_start_s + self.door_duration_s
        if door_is_open:
            temperature -= self.door_drop_C

        return temperature
