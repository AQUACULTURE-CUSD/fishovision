import cv2
import numpy as np
from .pipeline import ProcessingStep


class CircleCrop(ProcessingStep):
    """
    Crops the image so that only a circle is visible. Can set hyperparameters to determine
    circle location (relative to the center of the image) and the radius.

    Context Input: context['current_frame'] holding the current frame.
    Context Output: context['mask'] holding the circle mask (combined with any previous mask steps).
    """

    def __init__(self, center=(0,0), r=0):
        self.center = center
        self.r = r

    def process(self, context: dict) -> dict:
        frame = context.get('current_frame')
        if frame is None:
            return context  # Or raise an error

        """ Applies circular mask to frame """

        hh, ww, ch = frame.shape

        center = ((ww // 2) + self.center[0], (hh // 2) + self.center[1])

        if self.r == 0:
            r = self.center[1]
        else:
            r = self.r

        # Create a mask with a filled white circle
        mask = np.zeros((hh, ww), dtype=np.uint8)
        cv2.circle(mask, center, r, 255, thickness=-1)

        # Add the new mask to the context (combining it with earlier masks if they exist
        if context.get('mask') is not None:
            mask = cv2.bitwise_and(context['mask'], mask)
        context['mask'] = mask

        context['current_frame'] = frame
        return context
