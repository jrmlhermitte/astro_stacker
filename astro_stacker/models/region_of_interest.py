from dataclasses import dataclass


@dataclass
class RegionOfInterest:
    xmin: int
    xmax: int
    ymin: int
    ymax: int