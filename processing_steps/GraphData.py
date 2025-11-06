import cv2
import numpy as np
from .pipeline import ProcessingStep
import csv


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
        for old_point, new_point in zip(tracks[0], tracks[1]):
            avg_len += cv2.norm(new_point - old_point, normType=cv2.NORM_L2)
            pass
        avg_len /= len(tracks)
        with open(self.outfile, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame_number', 'avg_length'])
            writer.writerow([context.get('frame_number', -1), avg_len])
        return context
