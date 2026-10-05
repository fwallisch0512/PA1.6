from numpy.random import default_rng

class RNG:
    def __init__(self, seed: int):
        self.rng = default_rng(seed)

    def gauss(self, mu: float, sigma: float) -> float:
        return float(self.rng.normal(mu, sigma))

    def uniform(self, a: float, b: float) -> float:
        return float(self.rng.uniform(a, b))

    def bernoulli(self, p: float) -> bool:
        return bool(self.rng.random() < p)

