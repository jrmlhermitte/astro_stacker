import numpy as np
from scipy.signal import fftconvolve

from astro_stacker.models.image import Image


class SpatialCorrelator:
    def correlate(self,
                  image_1: Image,
                  image_2: Image,
                  symmetric_averaging: bool = False,
                  bgsub: bool = True
                  ) -> np.ndarray:
        data_1 = image_1.data
        data_1_grey = np.average(data_1, axis=2)
        if bgsub:
            data_1_grey -= np.average(data_1_grey)

        data_2 = image_2.data
        data_2_grey = np.average(data_2, axis=2)
        if bgsub:
            data_2_grey -= np.average(data_2_grey)

        # rare_value_threshold = np.quantile(data_1_grey.reshape(-1), .99)
        # data_1_grey *= data_1_grey > rare_value_threshold
        # data_2_grey *= data_2_grey > rare_value_threshold

        convolution_grey= cross_correlate_2d(
            data_1_grey, data_2_grey)

        mask_1 = np.average(image_1.mask, axis=2) > 0
        mask_2 = np.average(image_2.mask, axis=2) > 0
        convolution_mask = cross_correlate_2d(
            mask_1.astype(float), mask_2.astype(float)
        )

        if symmetric_averaging is False:
            normalized_convolution_grey = convolution_grey / convolution_mask
        else:
            left_mask = cross_correlate_2d(data_1_grey * mask_1, mask_2)
            right_mask = cross_correlate_2d(mask_1, data_2_grey * mask_2)
            normalized_convolution_grey = (
                    convolution_grey * convolution_mask / left_mask / right_mask)
        return normalized_convolution_grey


def cross_correlate_2d(
        data_1: np.ndarray,
        data_2: np.ndarray) -> np.ndarray:
    assert data_1.shape == data_2.shape

    assert data_1.ndim == 2
    assert data_2.ndim == 2

    return fftconvolve(data_1, data_2[::-1, ::-1], mode="full")
