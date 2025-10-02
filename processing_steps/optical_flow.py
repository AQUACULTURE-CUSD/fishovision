from processing_steps.pipeline import ProcessingStep
import cv2


class OpticalFlowCalculator(ProcessingStep):
    """
    Calculates the optical flow between two images.

    Initialized Values:
        minimum feature quality (defines the minimum acceptable quality of feature matches),
        feature_threshold (the number of features that we want. The algorithm will try to continue tracking the same
            features throughout the process, but gets rid of bad ones. If the total number of tracked features falls
            below this value, it will trigger another full feature search).

    Input: Previous image @ self.prev_gray, current image @ context['current_frame']
    Output: (Old, New) lists of matching point pairs. (Old[i], New[i]) are matched point pairs.
        Found @ context['tracks']
    """
    def __init__(self, min_feature_quality: float, feature_threshold: int = 100):
        self.prev_gray = None
        self.feature_points = None  # <-- RENAMED for clarity

        # Hyperparameters
        self.min_feature_quality = min_feature_quality
        self.feature_threshold = feature_threshold  # <-- NEW: Min points before re-detection

        # LK parameters, most of these should stay at these values (but we can think about changing them if necessary)
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
            return context  # Skip tracking on the detection frame

        # If we have feature points, track them
        if self.feature_points is not None:
            # Calculate optical flow
            new_points, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                current_gray,
                self.feature_points,
                None,
                # This expands the lk_params dictionary as a set of labelled function arguments
                **self.lk_params
            )

            # Filter and keep only the successfully tracked points
            if new_points is not None:
                good_new = new_points[status == 1]
                good_old = self.feature_points[status == 1]

                # CRITICAL: Update the feature list with the new positions
                self.feature_points = good_new.reshape(-1, 1, 2)

                # Add the valid tracks to the context for visualization
                context['tracks'] = (good_old, good_new)

        # Remember the current frame for the next iteration
        self.prev_gray = current_gray

        return context
