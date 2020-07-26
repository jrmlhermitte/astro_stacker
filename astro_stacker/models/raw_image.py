from dataclasses import dataclass
import numpy as np

# An image always has two dimensions in this case
from astro_stacker.exceptions import Invalid2DImageException
from astro_stacker.models.constants import IMAGE_DIMENSIONS


# Very thin wrapper but emphasizes that this is an array meant to be treated as an image
@dataclass
class RawImage:
    data: np.ndarray

    def __post_init__(self):
        if self.data.ndim != IMAGE_DIMENSIONS:
            raise Invalid2DImageException(num_dimensions=self.data.ndim)
