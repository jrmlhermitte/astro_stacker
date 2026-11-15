from astro_stacker.models.series import Series
from astro_stacker.curve_fitters.gaussian_fitter import GaussianCurveFitter
from astro_stacker.curve_fitters.functions.gaussian import gaussian_func
import numpy as np
from lmfit import Parameters


def _make_test_params() -> Parameters:
    params_dict = {
        'center': 300,
        'sigma': 100,
        'baseline': -2,
        'amplitude': 100,
    }
    params = Parameters()
    for key, value in params_dict.items():
        params.add(key, value=value)
    return params


def _make_test_series_data(params: Parameters) -> Series:
    x_data = np.linspace(0, 1000, 1000)
    y_data = gaussian_func(x=x_data, **params)
    y_data = y_data + np.random.uniform(-.01, .01, len(y_data))
    return Series(x=x_data, y=y_data)


def test_gaussian_fit():
    fitter = GaussianCurveFitter()
    test_params = _make_test_params()
    series = _make_test_series_data(test_params)

    result = fitter.fit_data(series)
    for param_name, param in result.params.items():
        assert np.isclose(param.value, test_params[param_name].value,
                          rtol=.01)
