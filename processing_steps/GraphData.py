import cv2
import numpy as np
from .pipeline import ProcessingStep
from collections import deque


class GraphData(ProcessingStep):
    """
    Saves the average length of a vector in context['tracks'] to a text file
    Context Input:
    Context Output:
    """

    def __init__(self, outfile, fps, windowSize):
        self.most_recent = deque()
        self.fps = fps
        self.outfile = outfile
        self.window = windowSize

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
        if len(self.most_recent) < (self.window - 1):
            self.most_recent.append(avg_len)
            return context
        else:
            self.most_recent.append(avg_len)
            with open(self.outfile, 'a') as f:
                f.write(f'{context['frame_number']/float(self.fps)}, {sum(self.most_recent)/len(self.most_recent)}\n')
            self.most_recent.popleft()
        return context
