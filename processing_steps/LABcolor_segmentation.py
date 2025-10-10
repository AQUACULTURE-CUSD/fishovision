import cv2
import numpy as np
from .pipeline import ProcessingStep


class LabColorSegmentationMask(ProcessingStep):
    """
    Segments an image based on a color range in the L*a*b* color space.
    This step is configured with default values to isolate achromatic
    (grayish) regions from chromatic (brownish) ones.

    Parameters:
    - lower_bound (np.ndarray): The lower L*a*b* threshold.
    - upper_bound (np.ndarray): The upper L*a*b* threshold.

    Inputs (from context):
    - 'image' (np.ndarray): The BGR image to segment.

    Outputs (to context):
    - 'mask' (np.ndarray): A new binary mask where white pixels correspond
                           to the segmented region.
    """
    def __init__(self, lower_bound=np.array([0, 120, 120]), upper_bound=np.array([220, 138, 138])):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def process(self, context: dict) -> dict:
        image = context.get('current_frame')
        if image is None:
            raise KeyError("'current_image' not found in context. Cannot perform segmentation.")

        # Convert the image from BGR to the L*a*b* color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Create a mask by thresholding the L*a*b* image
        # Pixels within the bounds become white (255), others become black (0)
        mask = cv2.inRange(lab_image, self.lower_bound, self.upper_bound)

        # Add the new mask to the context (combining it with earlier masks if they exist
        if context.get('mask') is not None:
            mask = cv2.bitwise_and(context['mask'], mask)
        context['mask'] = mask

        return context
