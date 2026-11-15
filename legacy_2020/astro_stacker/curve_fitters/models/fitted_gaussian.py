from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
from lmfit.minimizer import MinimizerResult

from astro_stacker.curve_fitters.functions.gaussian import gaussian_func
from astro_stacker.models.series import Series


@dataclass
class FittedGaussian:
    series: Series
    minimizer_result: MinimizerResult

    def generate(self, x: np.ndarray) -> Series:
        params_dict = self.get_params_dict()
        y = gaussian_func(x=x, **params_dict)
        return Series(x=x, y=y)

    def get_params_dict(self) -> Dict:
        data = {}
        for param in self.minimizer_result.params.values():
            data[param.name] = param.value
        return data