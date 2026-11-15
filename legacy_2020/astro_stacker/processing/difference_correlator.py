from __future__ import annotations
import numpy as np

from astro_stacker.models.image import Image
from dataclasses import dataclass

# TODO: this whole module needs a lot of restructuring and tests.
#  It is currently experimental


@dataclass
class DifferenceWindow:
    dx: int
    dy: int

    def x_range(self) -> np.ndarray:
        return np.arange(-self.dx, self.dx + 1)

    def y_range(self) -> np.ndarray:
        return np.arange(-self.dy, self.dy + 1)


@dataclass
class CorrelatedImage:
    data: np.ndarray
    x: np.ndarray
    y: np.ndarray


class DifferenceCorrelator:
    """Just computes the different of images."""
    def __init__(self, window: DifferenceWindow):
        self.difference_window = window
        dx = window.dx
        dy = window.dy
        x_coords = np.arange(-dx, dx + 1)
        y_coords = np.arange(-dy, dy + 1)
        XY = np.meshgrid(x_coords, y_coords)
        self.coordinates_x = XY[0]
        self.coordinates_y = XY[1]
        # TODO : wrap into their own class
        self.max_x = np.max(self.coordinates_x)
        self.max_y = np.max(self.coordinates_y)
        self.min_x = np.min(self.coordinates_x)
        self.min_y = np.min(self.coordinates_y)

    @staticmethod
    def make_symmetric(dx: int = 40, dy: int = 40) -> DifferenceCorrelator:
        window = DifferenceWindow(dx=dx, dy=dy)
        return DifferenceCorrelator(window=window)

    def correlate(self,
                  image_1: Image,
                  image_2: Image,
                  ) -> np.ndarray:
        data_1 = image_1.data.astype(float)
        data_1_grey = np.average(data_1, axis=2)
        mask_1_grey = np.sum(image_1.mask, axis=2) > 0

        data_2 = image_2.data.astype(float)
        data_2_grey = np.average(data_2, axis=2)
        mask_2_grey = np.sum(image_2.mask, axis=2) > 0

        image_1_grey = Image(data=data_1_grey, mask=mask_1_grey)
        image_2_grey = Image(data=data_2_grey, mask=mask_2_grey)
        difference_correlation = self.correlate_difference_2d(
            image_1_grey,
            image_2_grey)

        return difference_correlation

    def _make_blank_correlated_image(self) -> np.ndarray:
        """Initialize a new correlated image when computing a
        difference correlation."""
        return np.zeros((2 * self.difference_window.dy + 1,
                         2 * self.difference_window.dx + 1))

    def _get_difference_mask_for_shift(self,
                                       image_X: np.ndarray,
                                       image_Y: np.ndarray,
                                       shiftx: int,
                                       shifty: int) -> np.ndarray:
        """When computing the difference, we roll axes. However, when rolling,
        the numbers on the edges get wrapped to the other edge of the array. We should mask
        them out as it does not make sense to include these points
        from a geometrical standpoint.
        """
        # TODO : written very quickly to try out thoughts, reorganize later
        absolute_x_shift = np.abs(shiftx)
        absolute_y_shift = np.abs(shifty)
        Nx = image_X.shape[1]
        Ny = image_Y.shape[0]
        upper_x_mask = image_X <= (Nx - 1) - absolute_x_shift
        lower_x_mask = image_X >= absolute_x_shift
        upper_y_mask = image_Y <= (Ny - 1) - absolute_y_shift
        lower_y_mask = image_Y >= absolute_y_shift
        return (upper_x_mask * lower_x_mask) * (upper_y_mask * lower_y_mask)

    # TODO: Add a mask to this in the future.
    def correlate_difference_2d(
            self,
            image_1: Image,
            image_2: Image,
    ) -> CorrelatedImage:
        assert image_1.data.shape == image_2.data.shape

        assert image_1.data.ndim == 2
        assert image_2.data.ndim == 2
        if image_1.mask is None:
            mask_1 = np.ones_like(image_1.data)
        else:
            mask_1 = image_1.mask

        if image_2.mask is None:
            mask_2 = np.ones_like(image_2.data)
        else:
            mask_2 = image_1.mask

        # TODO (jrmlhermitte): Maybe these image coordinates should be
        #  part of the reader
        image_coord_y = np.arange(image_1.data.shape[0])
        image_coord_x = np.arange(image_1.data.shape[1])

        # x is the fastest moving dimension
        image_X, image_Y = np.meshgrid(image_coord_x, image_coord_y)

        correlated_image = self._make_blank_correlated_image()
        dx = self.difference_window.dx
        dy = self.difference_window.dy

        for xi in range(-dx, dx+1):
            for yi in range(-dy, dy + 1):
                difference_mask = self._get_difference_mask_for_shift(
                    image_X, image_Y,
                    shiftx=xi, shifty=yi)
                mask_overlap = mask_1.astype(float) * np.roll(mask_2, shift=(yi, xi), axis=(0, 1))
                mask_this_iteration = mask_overlap  # * difference_mask
                data_1 = image_1.get_data()
                data_2 = np.roll(image_2.get_data(), shift=(yi, xi), axis=(0, 1))
                counts = np.sum(mask_this_iteration)
                image_difference = (data_1 - data_2) * mask_this_iteration
                delta_2 = np.max(np.abs(image_difference))
                correlated_image[yi, xi] = delta_2

        return CorrelatedImage(
            data=correlated_image,
            x=self.difference_window.x_range(),
            y=self.difference_window.y_range(),
        )
