from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rawpy

from astro_stacker.models.image import Image
from astro_stacker.models.region_of_interest import RegionOfInterest

CANON_RAW_EXTENSION = '.cr2'


# NOT sure where to put this yet
# . Keeping isolated here for now.
@dataclass
class CannonImageReader:
    filepaths: List[str]

    dark: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    region_of_interest: Optional[RegionOfInterest] = None

    @staticmethod
    def from_directory(directory: str) -> CannonImageReader:
        filepaths = [directory + '/' + filename
                     for filename in os.listdir(directory)
                     if _is_raw_cannon_filename(filename)]
        return CannonImageReader(filepaths=filepaths)

    def number_images(self) -> int:
        return len(self.filepaths)

    def get_image(self, n) -> Image:
        filepath = self.filepaths[n]
        return Image(
            data=rawpy.imread(filepath).postprocess(),
            mask=self.mask,
            dark=self.dark
        )


def _is_raw_cannon_filename(filepath) -> bool:
    return filepath.lower().endswith(CANON_RAW_EXTENSION)
