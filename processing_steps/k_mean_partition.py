import cv2
import numpy as np
from .pipeline import ProcessingStep


class KMeansPartitionStep(ProcessingStep):
    """
    A pipeline step that partitions the 'current_image' into k sets based on k-means clustering.
    """

    def __init__(self, k: int = 5):
        """
        :param k: the number of clusters
        """
        self.k = k

    def process(self, context: dict) -> dict:
        if 'current_image' not in context:
            raise ValueError("Context is missing 'current_image'.")

        image = context['current_image']

        # Change color to RGB (from BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
        pixel_vals = image_rgb.reshape((-1, 3))

        # Convert to float type only for supporting cv2.kmeans
        pixel_vals = np.float32(pixel_vals)

        # criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

        retval, labels, centers = cv2.kmeans(pixel_vals, self.k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)

        segmented_data = centers[labels.flatten()]

        segmented_image_rgb = segmented_data.reshape(image_rgb.shape)

        segmented_image_bgr = cv2.cvtColor(segmented_image_rgb, cv2.COLOR_RGB2BGR)

        context['current_image'] = segmented_image_bgr

        return context