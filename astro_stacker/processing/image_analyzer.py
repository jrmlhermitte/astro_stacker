from astro_stacker.models.analyzed_image import AnalyzedImage
from astro_stacker.image_readers.cannon_image_reader import CannonImageReader
from tqdm import tqdm
import numpy as np


class ImageAnalyzer:
    def analyze(self, image_reader: CannonImageReader) -> AnalyzedImage:
        # TODO (jrmlhermitte): replace with running average and running stddev
        final_image = None
        final_image_squared = None
        number_images = image_reader.number_images()
        for n in tqdm(range(number_images)):
            image = image_reader.get_image(n)
            if final_image is None:
                final_image = image.data
                final_image_squared = image.data ** 2
            else:
                final_image += image.data
                final_image_squared += image.data ** 2
        data_average = final_image / number_images
        data_stddev = final_image_squared/(number_images + 1) - data_average * number_images/(number_images + 1)
        return AnalyzedImage(
            average=np.array(data_average),
            variance=np.array(data_stddev)
        )
