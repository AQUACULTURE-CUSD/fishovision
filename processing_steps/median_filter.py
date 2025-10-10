import cv2
import numpy as np
from .pipeline import ProcessingStep


class MedianFilter(ProcessingStep):
    """
    Applies a median blur to an image to reduce noise. This is particularly
    effective against "salt-and-pepper" noise.

    Parameters:
    - kernel_size (int): The size of the median filter kernel. Must be an
                         odd integer greater than 1 (e.g., 3, 5, 7).

    Inputs (from context):
    - 'current_image' (np.ndarray): The input image to be filtered.

    Outputs (to context):
    - 'current_image': The filtered image, which overwrites the current
                            in the context.
    """
    def __init__(self, kernel_size: int = 5):
        # Ensure the kernel size is an odd number
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer.")
        self.kernel_size = kernel_size

    def process(self, context: dict) -> dict:
        print("Applying Median Filter...")
        image = context.get('current_frame')
        if image is None:
            raise KeyError("'current_frame' not found in context. Cannot apply median filter.")

        # Apply the median blur operation
        filtered_image = cv2.medianBlur(image, self.kernel_size)

        # Update the context with the processed image
        context['current_frame'] = filtered_image
        return context
