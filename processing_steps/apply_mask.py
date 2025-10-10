import cv2
from .pipeline import ProcessingStep


class ApplyMaskDenoised(ProcessingStep):
    """
    Performs a morphological opening operation on a binary mask to remove
    small, isolated noise. This is useful for cleaning up a mask after
    thresholding.

    Parameters:
    - kernel_size (tuple): The (width, height) of the kernel for the operation.

    Inputs (from context):
    - 'mask' (np.ndarray): The binary mask to be cleaned.

    Outputs (to context):
    - 'mask' (np.ndarray): The cleaned mask, which overwrites the original.
    """
    def __init__(self, kernel_size: tuple = (5, 5)):
        self.kernel_size = kernel_size

    def process(self, context: dict) -> dict:
        mask = context.get('mask')

        if mask is None:
            raise KeyError("'mask' not found in context. Cannot apply opening.")

        # Create a structuring element (kernel). An ellipse is often good for natural shapes.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)

        # Apply the morphological opening operation
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Update the context with the cleaned mask
        context['mask'] = cleaned_mask
        image = context['current_frame']
        context['current_frame'] = cv2.bitwise_and(image, image, mask=cleaned_mask)
        return context
