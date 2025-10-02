import cv2
from processing_steps.pipeline import ProcessingStep


class GrayscaleConverter(ProcessingStep):
    """
    Converts an input image to grayscale.

    Input: Any image @ context['current_frame']
    Output: Grayscaled image @ context['current_frame']
    """
    def process(self, context: dict) -> dict:
        frame = context.get('current_frame')
        if frame is None:
            return context  # Or raise an error

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # We can either replace the frame or add a new key
        context['current_frame'] = gray_frame
        return context
