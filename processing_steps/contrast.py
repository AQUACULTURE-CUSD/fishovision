import numpy as np
import cv2
from .pipeline import ProcessingStep


class HistogramContrastAdjuster(ProcessingStep):
    """
    Adjusts the contrast of the input image.

    Input: Grayscale image @ context['current_frame']
    Output: Image with brightness adjusted @ context['current_frame']
    """
    def __init__(self):
        pass

    def process(self, context: dict) -> dict:
        frame = context.get('current_frame')
        if frame is None:
            return context

        context['current_frame'] = cv2.equalizeHist(frame)
        return context


class LinearContrastAdjuster(ProcessingStep):
    def __init__(self, alpha:float):
        self.alpha = alpha

    def process(self, context: dict) -> dict:
        frame = context.get('current_frame')
        if frame is None:
            return context

        adjusted_image = cv2.convertScaleAbs(frame, alpha=self.alpha)

        context['current_frame'] = adjusted_image
        return context