class Invalid2DImageException(Exception):
    def __init__(self, num_dimensions: int):
        self.num_dimensions = num_dimensions

    def __str__(self) -> str:
        return f'Expected 2 dimensions, but got {self.num_dimensions}'