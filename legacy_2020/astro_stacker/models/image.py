from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from skimage.measure import block_reduce

from astro_stacker.models.region_of_interest import RegionOfInterest
from astro_stacker.models.series import Series
from astro_stacker.algorithms.savitzky_golay import sgolay2d


@dataclass
class Image:
    data: np.ndarray
    mask: Optional[np.ndarray] = None
    dark: Optional[np.ndarray] = None
    # In the future: flatfield

    def region_of_interest(
            self,
            region_of_interest: RegionOfInterest) -> Image:
        xmin = region_of_interest.xmin
        xmax = region_of_interest.xmax
        ymin = region_of_interest.ymin
        ymax = region_of_interest.ymax
        data = self.data[ymin:ymax, xmin:xmax, :]
        if self.mask is not None:
            mask = self.mask[ymin:ymax, xmin:xmax]
        else:
            mask = None
        if self.dark is not None:
            dark = self.dark[ymin:ymax, xmin:xmax]
        else:
            dark = None
        return Image(data=data, mask=mask, dark=dark)

    def downsample(self, shape: Tuple[int, int, int]) -> Image:
        mask = self.mask if self.mask is not None else 1
        data = block_reduce(self.data * mask, shape, func=np.mean)
        if self.mask is not None:
            mask = block_reduce(self.mask, shape, func=np.max).astype(bool)
        else:
            mask = None
        if self.dark is not None:
            dark = block_reduce(self.dark * self.mask, shape, func=np.mean)
        else:
            dark = None
        return Image(data=data, mask=mask, dark=dark)

    def histogram(self, nbins=1000) -> Series:
        data_flattened = (self.data * self.mask).reshape(-1)
        counts, edges = np.histogram(data_flattened, bins=nbins)
        centers = _edges_to_centers(edges)
        return Series(x=centers, y=counts)

    def smooth(self, window_size: int = 9, order: int = 3) -> Image:
        mask = self.mask if self.mask is not None else 1
        data_s = _sgsmooth_3d_image(
            self.data * mask, window_size=window_size, order=order, derivative=None)
        if self.mask is not None:
            mask_s = _sgsmooth_3d_image(
                self.mask.astype(int),
                window_size=window_size,
                order=order,
                derivative=None).astype(bool)
        else:
            mask_s = None
        if self.dark is not None:
            dark_s = _sgsmooth_3d_image(
                self.dark, window_size=window_size, order=order, derivative=None)
        else:
            dark_s = None
        return Image(data=data_s, mask=mask_s, dark=dark_s)

    def get_data(self) -> np.ndarray:
        data = self.data
        if self.dark is not None:
            data = data - self.dark
        if self.mask is not None:
            data = data * self.mask
        return data


def _sgsmooth_3d_image(images: np.ndarray, window_size: int,
                       order: int, derivative: Optional[str]) -> np.ndarray:
    # assumed axis is the third axis. Still experimentation
    output_images = []
    third_dimension_size = images.shape[2]
    for i in range(third_dimension_size):
        output_images.append(sgolay2d(images[:, :, i], window_size=window_size,
                                      order=order, derivative=derivative))
    return np.stack(output_images, axis=2)


def _edges_to_centers(edges: np.ndarray) -> np.ndarray:
    return (edges[1:] + edges[:-1]) * .5
