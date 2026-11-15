from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np


@dataclass
class AnalyzedImage:
    average: np.ndarray
    variance: np.ndarray

    @staticmethod
    def from_filename(filename: str) -> AnalyzedImage:
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def to_filename(self, filename: str) -> None:
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
