from .pipeline import ProcessingStep
import numpy as np
import cv2


class CropLine(ProcessingStep):
    """
    This crops the image along a particular line (useful for chopping off a bad chunk in the corner).

    Initialized parameters:
        slope -> the slope of the line (in pixel coordinates with (0,0) in the top left corner)
        intercept -> the position on the y-axis that the line should cross
            (y-axis positive goes down in pixel coordinates)

    Inputs:
        The current frame (grayscaled or not) @ context['current_frame']

    Outputs:
        The current frame with the cropping applied (with black pixels in the cropped area)
    """
    def __init__(self, slope: float, intercept: float, reverse: bool=False):
        self.normal = np.array([-slope, 1])
        self.reverse = reverse
        self.bias = -intercept

    def process(self, context: dict) -> dict:
        a,b = self.normal
        c = self.bias
        image = context['current_frame']
        h, w = image.shape[:2]

        # Create a grid of (x, y) coordinates for every pixel
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # Calculate the line equation for all pixels at once (vectorized)
        side_check = a * x_coords + b * y_coords + c

        # Create a boolean mask where the condition is met (e.g., > 0)
        if not self.reverse:
            mask_numpy = (side_check >= 0).astype(np.uint8) * 255
        else:
            mask_numpy = (side_check <= 0).astype(np.uint8) * 255

        # Apply the mask
        context['current_frame'] = cv2.bitwise_and(image, image, mask=mask_numpy)
        return context
