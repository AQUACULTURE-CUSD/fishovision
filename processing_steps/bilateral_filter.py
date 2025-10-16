import cv2
import numpy as np
from .pipeline import ProcessingStep


class BilateralFilter(ProcessingStep):
    """
    Applies a bilateral filter to an image to reduce noise. 

    Parameters:
    - diameter (int):  The diameter of each pixel neighborhood. Must be a 
                        positive integer. 
    - sigmaColor (int): The size of the bilateral filter sigma in the color space. 
                        Increasing the value will mix colors far from each other.
    - sigmaSpace (int): The size of the bilateral filter sigma in the coordinate space. 
                        Increasing the value will mix pixels far from each other. 

    Inputs (from context):
    - 'current_image' (np.ndarray): The input image to be filtered.

    Outputs (to context):
    - 'current_image': The filtered image, which overwrites the current
                            in the context.
    """
    def __init__(self, diameter: int=9, sigmaColor: int=150, sigmaSpace: int=150):
        # Ensure the kernel size is a positive number
        if diameter < 0:
            raise ValueError("diameter must be a positive integer.")
        self.diameter= diameter
        self.sigmaColor= sigmaColor
        self.sigmaSpace= sigmaSpace

    def process(self, context: dict) -> dict:
        print("Applying Bilateral Filter...")
        image = context.get('current_frame')
        if image is None:
            raise KeyError("'current_frame' not found in context. Cannot apply bilateral filter.")

        # Set region of interest
        #x=0
        #y=0
        #w=150
        #h=150

        hh, ww, _ = image.shape  # height, width, channels

        w, h = 500, 500  # size of ROI
        x = ww//2 - w//2  # center horizontally
        y = hh//2 - h//2   

        roi=image[x: x + w, y: y + h]
        
        print(f"Applying Bilateral Filter to ROI at (x={x}, y={y}, w={w}, h={h})...")
        # x=710, y=930, w=200, h=150

        # Apply bilateral filter
        filtered_image = cv2.bilateralFilter(image, self.diameter, self.sigmaColor, self.sigmaSpace)

        # Apply bilateral filter to ROI
        filtered_roi = cv2.bilateralFilter(roi, self.diameter, self.sigmaColor, self.sigmaSpace)

        image[x: x + w, y: y + h] = filtered_roi

        # Update the context with the processed image
        # context['current_frame'] = filtered_image
        
        # Update the context with the processed ROI image
        context['current_frame'] = image

        return context
    
