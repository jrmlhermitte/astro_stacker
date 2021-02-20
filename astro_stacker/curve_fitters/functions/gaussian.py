import numpy as np


def gaussian_func(
        x: np.ndarray,
        amplitude: float,
        baseline: float,
        center: float,
        sigma: float
) -> np.ndarray:
    return amplitude * np.exp(-(x - center) ** 2 / 2. / sigma**2) + baseline