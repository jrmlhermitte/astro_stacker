from dataclasses import dataclass
from typing import Optional

from astro_stacker.models.raw_image import RawImage


@dataclass
class Image:
    data: RawImage
    mask: Optional[RawImage]
    dark: Optional[RawImage]