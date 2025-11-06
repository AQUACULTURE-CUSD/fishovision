from .pipeline import ProcessingStep
import numpy as np
import cv2


class OpticalFlowCalculator(ProcessingStep):
    """
    Calculates the optical flow between two images.

    Initialized Values:
        minimum feature quality (defines the minimum acceptable quality of feature matches),
        feature_threshold (the number of features that we want. The algorithm will try to continue tracking the same
            features throughout the process, but gets rid of bad ones. If the total number of tracked features falls
            below this value, it will trigger another full feature search).

    Input: Previous image @ self.prev_gray, current image @ context['current_frame'], previous points @ self.prev_points,
    Output: (Old, New) lists of matching point pairs. (Old[i], New[i]) are matched point pairs.
        Found @ context['tracks']
    """
    def __init__(self, min_feature_quality: float, feature_threshold: int = 100):
        self.prev_gray = None
        self.prev_features = None  # <-- RENAMED for clarity

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
        if self.prev_gray is None:
            # Find initial features in the first frame
            self.prev_features = cv2.goodFeaturesToTrack(
                current_gray,
                maxCorners=500,
                qualityLevel=self.min_feature_quality,
                minDistance=15
            )
            # Store the current frame as the 'previous' for the next iteration
            self.prev_gray = current_gray
            return context

        # SUBSEQUENT FRAMES: We have a previous frame and points, so we can track.
        if self.prev_features is not None:
            # Ensure points are float32, a common requirement for this function
            p0 = self.prev_features.astype(np.float32)

            # Calculate optical flow
            new_points, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,  # Previous frame
                current_gray,  # Current frame
                p0,  # Points from the PREVIOUS frame
                None,  # Let OpenCV find the new points
                **self.lk_params
            )

            # Filter and keep only the successfully tracked points
            if new_points is not None:
                good_new = new_points[status == 1]
                good_old = p0[status == 1]  # <-- CORRECTED: Use the points we tracked FROM
                context['tracks'] = (good_old, good_new)

        self.prev_features = cv2.goodFeaturesToTrack(
            current_gray,
            maxCorners=500,
            qualityLevel=self.min_feature_quality,
            minDistance=15
        )

        # Remember the current frame for the next iteration
        self.prev_gray = current_gray

        return context

