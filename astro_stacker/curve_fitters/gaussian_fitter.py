from lmfit import Parameters, minimize
import numpy as np

from astro_stacker.curve_fitters.functions.gaussian import gaussian_func
from astro_stacker.curve_fitters.functions.stats import center_of_mass, sigma
from astro_stacker.models.series import Series

from astro_stacker.curve_fitters.models.fitted_gaussian import FittedGaussian


class GaussianCurveFitter:
    def __init__(self):
        pass

    def fit_data(self, series: Series) -> FittedGaussian:
        params = self._guess_parameters(series)
        result = minimize(
            gaussian_residual,
            params,
            args=(series.x, series.y))
        fitted_gaussian = FittedGaussian(
            series=series,
            minimizer_result=result,
        )
        return fitted_gaussian

    def _guess_parameters(self, series: Series) -> Parameters:
        guessed_baseline = np.min(series.y)
        guessed_amplitude = np.max(series.y) - guessed_baseline
        guessed_center = center_of_mass(x=series.x, y=series.y)
        # TODO (jrmlhermitte): This recomputes the average. Cache the value for later
        guessed_sigma = sigma(x=series.x, y=series.y, com=guessed_center)

        params = Parameters()
        params.add('amplitude', value=guessed_amplitude)
        params.add('baseline', value=guessed_baseline)
        params.add('center', value=guessed_center)
        params.add('sigma', value=guessed_sigma, min=0)
        return params


def gaussian_residual(params: Parameters,
                      x: np.ndarray,
                      data: np.ndarray) -> np.ndarray:
    model = gaussian_func(
        x=x,
        amplitude=params['amplitude'],
        baseline=params['baseline'],
        center=params['center'],
        sigma=params['sigma'])
    return data - model
