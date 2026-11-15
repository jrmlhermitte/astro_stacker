from typing import Optional

import numpy as np


def center_of_mass(x: np.ndarray, y: np.ndarray) -> float:
    return _nth_moment_of(x=x, y=y, n=1) / np.sum(y)


def sigma(x: np.ndarray,
          y: np.ndarray,
          com: Optional[float] = None) -> float:
    if com:
        x_1 = com
    else:
        x_1 = center_of_mass(x, y)
    x_2 = _nth_moment_of(x=x, y=y, n=2)
    return np.sqrt((x_2 - x_1**2) / np.sum(y))


def _nth_moment_of(x: np.ndarray, y: np.ndarray, n: int = 1) -> float:
    return np.sum((x ** n) * y)