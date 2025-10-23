from .pipeline import ProcessingStep
import numpy as np
import cv2


class Visualize(ProcessingStep):
    """
    Step to visualize the LK optical flow (to get an idea of what we need to adjust).

    Input:
        The original frame (to draw on) @ context['original_frame']
        The tracked Optical Flow point pairs @ context['tracks']
    Output:
        Shows the image with the vectors drawn on, waits for a key press to continue.
        Visualized image @ context['visualized']
    """
    def process(self, context: dict) -> dict:
        # --- 1. Get data and prepare the image (same as before) ---
        frame = context.get('original_frame')
        if frame is None:
            return context

        output_image = frame.copy()
        tracks = context.get('tracks')

        if tracks is not None:
            good_old, good_new = tracks
            # Loop and draw logic is identical
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                # Flattens the points into a 1D array (in case they aren't already)
                # Then, unpacks the point coordinates into a,b,c,d and makes them all int32 typed
                a, b = new.ravel()
                c, d = old.ravel()
                a, b, c, d = np.int32(a), np.int32(b), np.int32(c), np.int32(d)

                # Adds a line from the old point to the new point and a red point at the new
                cv2.line(output_image, (c, d), (a, b), (0, 255, 0), 2)
                cv2.circle(output_image, (a, b), 4, (0, 0, 255), -1)

        # --- 2. Display, wait, and destroy ---
        window_name = 'Debug Step: Optical Flow'

        # Display the image
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, output_image)
        # Pause execution until a key is pressed
        cv2.waitKey(0)
        # After a key is pressed, destroy the window
        cv2.destroyWindow(window_name)

        # --- 3. Update the context and return ---
        context['visualized'] = output_image
        return context
