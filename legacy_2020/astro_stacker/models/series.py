from dataclasses import dataclass
import numpy as np


# TODO (jrmlhermitte): See if pandas Series is okay enough to use here.
@dataclass
class Series:
    x: np.ndarray
    y: np.ndarray
