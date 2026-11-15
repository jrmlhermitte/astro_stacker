import numpy as np

from astro_stacker.models.analyzed_image import AnalyzedImage

QUANTILE_VARIANCE_THRESHOLD = .9999
# Move into image
NUMBER_COLORS = 3


# TODO (jrmlhermitte): This is a placeholder for now. Later on,
#  will add more in-depth analysis to find anomalous points.
class VarianceAnalyzer:
    def generate_mask(self, image: AnalyzedImage) -> np.ndarray:
        return _find_anomalous_points(image)


def _find_anomalous_points(image: AnalyzedImage) -> np.ndarray:
    image_flattened = image.variance.reshape((-1, 3))
    anomaly_threshold = np.quantile(image_flattened, QUANTILE_VARIANCE_THRESHOLD, axis=0)
    pixel_mask = image.variance.data < anomaly_threshold
    return pixel_mask
