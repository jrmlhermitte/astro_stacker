from dataclasses import dataclass


# TODO(jrmlhermitte): We should save the scale here
#  (as well as in the original image).
@dataclass
class Shift:
    xshift: int
    yshift: int


class ShiftAnalyzer:
    def analyze(self) -> Shift:
        pass


