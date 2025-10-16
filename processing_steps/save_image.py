from .pipeline import ProcessingStep
import cv2

class SaveCurrentImage:
    """ Saves image """
    def __init__(self, output_folder="data/blurtc"):
        import os
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        self.counter = 0

    def process(self, frame):
        import cv2, os
        filename = os.path.join(self.output_folder, f"frame_{self.counter:04d}.jpg")
        cv2.imwrite(filename, frame)
        self.counter += 1
        return frame
