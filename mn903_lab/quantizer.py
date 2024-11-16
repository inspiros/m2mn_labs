import numpy as np


__all__ = ['Quantizer', 'distortion']

class Quantizer:
    def __init__(self, delta: float = 1.0):
        if delta <= 0:
            raise ValueError('delta must be > 0')
        self.delta = delta

    def quant_midrise(self, x, m=None):
        idx = np.ceil(x / self.delta)
        if m:
            idx = np.clip(idx, -m / 2, m / 2 + 1)
        qx = (idx - .5) * self.delta
        return qx, idx

    def quant_midtread(self, x, m=None):
        idx = np.ceil(x / self.delta - .5)
        if m:
            idx = np.clip(idx, -m / 2 + 1, m / 2)
        qx = idx * self.delta
        return qx, idx


def quant_midrise(x, delta: float = 1.0, m=None):
    return Quantizer(delta).quant_midrise(x, m)


def quant_midtread(x, delta: float = 1.0, m=None):
    return Quantizer(delta).quant_midtread(x, m)


def distortion(x, qx) -> float:
    return np.var(x - qx)
