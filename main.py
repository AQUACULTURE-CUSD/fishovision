# importing the module 
import cv2
import numpy as np
import os


# split up a video into a collection of images 
# video_path = "/Users/lieselwong/Documents/fishovision/data/10_1-Vid2.mp4"
# output_path = "/Users/lieselwong/Documents/fishovision/data/output"
def get_images(video_path, output_folder, frame_interval=5):
    """
    Extracts frames from a video and saves them as images.

    Parameters:
        video_path (str): Path to the input video file. Defaults to a file in the current directory if not provided.
        output_folder (str): Folder to save the extracted images. Defaults to a folder in the current directory if not provided.
        frame_interval (int): Save every 'n'-th frame (default is 1, meaning every frame).
    """
    # Set default paths based on the current working directory
    if video_path is None:
        video_path = os.path.join(os.getcwd(), "10_1-Vid2.mp4")
    if output_folder is None:
        output_folder = os.path.join(os.getcwd(), "frames")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video has ended
        
        # Save frame at the specified interval
        if frame_count % frame_interval == 0:
            image_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(image_path, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames and saved to {output_folder}")

# Example usage:
# get_images(frame_interval=5)

source = cv2.VideoCapture('data/10_1-Vid2.mp4')

# We need to set resolutions.
# so, convert them from float to integer. 
frame_width = int(source.get(3))
frame_height = int(source.get(4))

size = (frame_width, frame_height)

video_name = input()
result = cv2.VideoWriter('data/output/'+video_name+".mp4",
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         30, size, 0)

# running the loop to convert frames to grayscale 

while True:

    # extracting the frames 
    ret, img = source.read()
    if img is None:
        break
    # converting to gray-scale
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = 60  # whatever value you want to add
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], value)
    gray = hsv[:, :, 2]

    # write to gray-scale 
    result.write(gray)

    # displaying the video 
    cv2.imshow("Live", gray)

    # exiting the loop 
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# closing the window 
cv2.destroyAllWindows()
source.release()

get_images(video_path="data/10_1-Vid2.mp4", output_folder="data/frames", frame_interval=5)

