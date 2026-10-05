
from dataclasses import dataclass
from typing import Optional
from utils.rng import RNG

@dataclass
class TempSensor:
    sigma: float = 0.2
    bias: float = 0.3
    dropout_prob: float = 0.02
    rng: Optional[RNG] = None

    def read(self, true_temp: float) -> Optional[float]:
        if self.rng is None:
            raise ValueError("TempSensor requires a RNG")

        if self.rng.bernoulli(self.dropout_prob):
            return None

        noisy = true_temp + self.bias + self.rng.gauss(0.0, self.sigma)
        return noisy
