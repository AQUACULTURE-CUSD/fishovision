import cv2 as cv2
import numpy as np
import os


def crop_center_circle(frame):
    """ Applies circular mask to frame """

    hh, ww = frame.shape[:2]

    # Center points
    xc = ww // 2
    yc = hh // 2
    
    radius= yc
    
    # Create a mask with a filled white circle
    mask = np.zeros((hh, ww), dtype=np.uint8)
    cv2.circle(mask, (xc, yc), radius, 255, thickness=-1)

    # Assuming 3 channels for test case ***
    MASK = cv2.merge([mask, mask, mask])

    # Apply mask
    masked_frame = cv2.bitwise_and(frame, MASK)

    # Crop to the square area of the circle
    cropped_frame = masked_frame[yc - radius:yc + radius, xc - radius:xc + radius]

    return cropped_frame


def run_test(image_path, n):
    # Load your image
    frame = cv2.imread(image_path)

    cropped = crop_center_circle(frame)

    # Create folder
    output_folder = "cropped_frames"
    os.makedirs(output_folder, exist_ok=True)

    filename = f"cropped_frame_{n}.jpg"
    cv2.imwrite(os.path.join(output_folder, filename), cropped)

    print(f"Saved cropped image to {os.path.join(output_folder, filename)}")
