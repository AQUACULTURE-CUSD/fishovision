import cv2
from .pipeline import ProcessingStep


class MidToneThresholdMask(ProcessingStep):
    """
    Thresholds an image to keep only mid-toned areas, setting pixels
    below a low threshold and above a high threshold to black.

    Parameters:
    - low_threshold (int): Pixel intensities below this value will be set to 0.
    - high_threshold (int): Pixel intensities above this value will be set to 0.

    Context Input:
        'current_frame' A grayscale image.

    Context Output:
        'current_frame' (np.ndarray): The thresholded image, with original pixel
                                   values preserved in the mid-tone range.
    """

    def __init__(self, low_threshold: int, high_threshold: int):
        if not 0 <= low_threshold < high_threshold <= 255:
            raise ValueError("Thresholds must be in the range 0-255 and low < high.")
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def process(self, context: dict) -> dict:
        image = context.get('current_frame')
        if image is None:
            return context  # Or raise an error

        # cv2.inRange is the most efficient way to create a mask for a specific range.
        # It creates a white mask (255) for pixels within the range and black (0) for those outside.
        mask = cv2.inRange(image, self.low_threshold, self.high_threshold)
        if context.get('mask') is not None:
            context['mask'] = cv2.bitwise_and(context['mask'], mask)
        else:
            context['mask'] = None

        return context
