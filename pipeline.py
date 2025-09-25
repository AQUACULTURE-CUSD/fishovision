from abc import abstractmethod, ABC
import cv2
import numpy as np
import LucasKanade


# The interface for any step in our pipeline
class ProcessingStep(ABC):
    @abstractmethod
    def process(self, context: dict) -> dict:
        """
        Processes data from the context dictionary and returns the updated context.
        'context' is a dictionary used to pass data between steps.
        """
        pass


class Pipeline:
    def __init__(self, steps: list[ProcessingStep]):
        self.steps = steps

    def run(self, context: dict) -> dict:
        """Runs the data through all registered steps."""
        for step in self.steps:
            context = step.process(context)
        return context


class GrayscaleConverter(ProcessingStep):
    def process(self, context: dict) -> dict:
        frame = context.get('current_frame')
        if frame is None:
            return context  # Or raise an error

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # We can either replace the frame or add a new key
        context['current_frame'] = gray_frame
        return context


class BrightnessAdjuster(ProcessingStep):
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


class OpticalFlowCalculator(ProcessingStep):
    def __init__(self, min_feature_quality: float, feature_threshold: int = 100):
        self.prev_gray = None
        self.feature_points = None  # <-- RENAMED for clarity

        # Hyperparameters
        self.min_feature_quality = min_feature_quality
        self.feature_threshold = feature_threshold  # <-- NEW: Min points before re-detection

        # LK parameters
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def process(self, context: dict) -> dict:
        current_gray = context.get('current_frame')
        if current_gray is None:
            return context

        # If we don't have points yet, or too few points, detect new ones
        if self.feature_points is None or len(self.feature_points) < self.feature_threshold:
            # Use the current frame to detect features
            self.feature_points = cv2.goodFeaturesToTrack(
                current_gray,
                maxCorners=500,
                qualityLevel=self.min_feature_quality,
                minDistance=7
            )
            # After detecting, we set the current frame as the previous for the *next* iteration
            self.prev_gray = current_gray
            return context  # <-- Skip tracking on the detection frame

        # If we have feature points, track them
        if self.feature_points is not None:
            # Calculate optical flow
            new_points, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                current_gray,
                self.feature_points,
                None,
                **self.lk_params
            )

            # Filter and keep only the successfully tracked points
            if new_points is not None:
                good_new = new_points[status == 1]
                good_old = self.feature_points[status == 1]

                # <-- CRITICAL: Update the feature list with the new positions
                self.feature_points = good_new.reshape(-1, 1, 2)

                # Add the valid tracks to the context for visualization
                context['tracks'] = (good_old, good_new)

        # Remember the current frame for the next iteration
        self.prev_gray = current_gray

        return context


class Visualize(ProcessingStep):
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
                a, b = new.ravel()
                c, d = old.ravel()
                a, b, c, d = np.int32(a), np.int32(b), np.int32(c), np.int32(d)

                cv2.line(output_image, (c, d), (a, b), (0, 255, 0), 2)
                cv2.circle(output_image, (a, b), 4, (0, 0, 255), -1)

        # --- 2. Display, wait, and destroy ---
        window_name = 'Debug Step: Optical Flow'

        # Display the image
        cv2.imshow(window_name, output_image)
        # Pause execution until a key is pressed
        cv2.waitKey(0)
        # After a key is pressed, destroy the window
        cv2.destroyWindow(window_name)

        # --- 3. Update the context and return ---
        context['visualized'] = output_image
        return context


class ShowCurrentImage(ProcessingStep):
    """
    Just shows what the "current_image" context item looks like when this gets
    executed. Press any key to close the window and continue execution.
    """
    def process(self, context: dict) -> dict:
        frame = context.get('curent_frame')
        if frame is None:
            return context

        # --- 2. Display, wait, and destroy ---
        window_name = 'Debug Step: Optical Flow'

        # Display the image
        cv2.imshow(window_name, frame)
        # Pause execution until a key is pressed
        cv2.waitKey(0)
        # After a key is pressed, destroy the window
        cv2.destroyWindow(window_name)
        return context

