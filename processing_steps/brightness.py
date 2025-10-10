import numpy as np
import cv2
from .pipeline import ProcessingStep


class BrightnessAdjuster(ProcessingStep):
    """
    Adjusts the brightness of the input image.

    Initialized Value: brightness value between -255 and 255.
    Input: Grayscale image @ context['current_frame']
    Output: Image with brightness adjusted @ context['current_frame']
    """
    def __init__(self, brightness: int = 0):
        """
        Initializes the step with a brightness adjustment value.
        - brightness: A value from -255 to 255.
                      Positive values brighten, negative values darken.
        """
        self.brightness = brightness

    def process(self, context: dict) -> dict:
        if self.brightness == 0:
            return context  # No change needed

        frame = context.get('current_frame')
        if frame is None:
            return context

        # Create a matrix of the same size as the image, filled with the brightness value
        matrix = np.ones(frame.shape, dtype="uint8") * abs(self.brightness)

        # Add or subtract the matrix to adjust brightness
        if self.brightness > 0:
            adjusted_frame = cv2.add(frame, matrix)
        else:
            adjusted_frame = cv2.subtract(frame, matrix)

        context['current_frame'] = adjusted_frame
        return context
