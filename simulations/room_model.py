from typing import Optional
from utils.rng import RNG

def step_room(
    T: float,
    heater_on: int,
    T_out: float,
    R: float,
    C: float,
    P: float,
    dt: float,
    process_sigma: float = 0.0,
    rng: Optional[RNG] = None,
) -> float:
    dTdt = (T_out - T) / (R * C) + heater_on * P / C
    next_T = T + dTdt * dt

    if process_sigma > 0:
        if rng is None:
            raise ValueError("An RNG is required when process_sigma is positive")
        next_T += rng.gauss(0.0, process_sigma)

    return next_T
