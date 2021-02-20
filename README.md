# Astro Stacker
Small project experimenting with the manual stacking of images.


# Example
```py3
from astro_stacker.config import Config
from astro_stacker.models.cannon_raw_file_group import CannonRawFileGroup
from astro_stacker.greyscale_image_processor import CannonGreyScaleImageProcessor

image_filenames = CannonRawFileGroup.from_directory(Config.STACK_1_DIR)
darks = CannonRawFileGroup.from_directory(Config.DARK_DIR)

processor = CannonGreyScaleImageProcessor(darks)
average_image = processor.compute_average()


histogram = average_image.histogram()

# plotting
import plotly.express as px
px.bar(x=histogram.centers, y=histogram.counts)
```