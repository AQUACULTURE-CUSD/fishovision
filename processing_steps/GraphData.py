import cv2
import numpy as np
from .pipeline import ProcessingStep


class GraphData(ProcessingStep):
    """
    Saves the average length of a vector in context['tracks'] to a text file
    Context Input:
    Context Output:
    """

    def __init__(self, outfile):
        self.outfile = outfile

    def process(self, context: dict) -> dict:
        tracks = context.get('tracks')
        if tracks is None or tracks == []:
            return context  # Or raise an error
        avg_len = 0.0
        count = 0
        for old_point, new_point in zip(tracks[0], tracks[1]):
            l = cv2.norm(new_point - old_point, normType=cv2.NORM_L2)
            if l <= 130:
                avg_len += l
                count += 1
        avg_len /= count
        with open(self.outfile, 'a') as f:
            f.write(f'{context['frame_number']}, {avg_len}\n')
        return context
