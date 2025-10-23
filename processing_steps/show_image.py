from .pipeline import ProcessingStep
import cv2


class ShowCurrentImage(ProcessingStep):
    """
    Just shows what the "current_image" context item looks like when this gets
    executed. Press any key to close the window and continue execution.
    """
    def process(self, context: dict) -> dict:
        print("In Current Image")
        frame = context.get('current_frame')
        if frame is None:
            return context

        # --- 2. Display, wait, and destroy ---
        window_name = 'Debug Show'

        # Display the image
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, frame)
        # Pause execution until a key is pressed
        cv2.waitKey(0)
        # After a key is pressed, destroy the window
        cv2.destroyWindow(window_name)
        return context
