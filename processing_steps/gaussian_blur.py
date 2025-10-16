import cv2
import numpy as np
from .pipeline import ProcessingStep


class GaussianBlur(ProcessingStep):
    """
    Applies a gaussian blur to an image to reduce noise. 

    Parameters:
    - kernel_size (tuple): The size of the gaussian blur kernel. Both arg must be an
                         odd integer greater than 1 (e.g., 3, 5, 7).
    - sigmaX (float): Standard deviation in the x direction. 

    Inputs (from context):
    - 'current_image' (np.ndarray): The input image to be filtered.

    Outputs (to context):
    - 'current_image': The filtered image, which overwrites the current
                            in the context.
    """
    def __init__(self, kernel_size: (5,5), sigmaX:1):
        # Ensure kernel size has two odd numbers
        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            raise ValueError("each dimension in kernel_size must be odd.")
        self.kernel_size = kernel_size
        self.sigmaX= sigmaX

    def process(self, context: dict) -> dict:
        print("Applying Gaussian Blur...")
        image = context.get('current_frame')
        if image is None:
            raise KeyError("'current_frame' not found in context. Cannot apply gaussian blur.")

        # Apply gaussian blur
        filtered_image = cv2.GaussianBlur(image, self.kernel_size, self.sigmaX)

        # Update the context with the processed image
        context['current_frame'] = filtered_image
        return context
    

