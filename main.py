# importing the module 
import cv2
import numpy as np
import os


def assess_paths(video, folder):
    """
    Checks whether the video and output folder paths are good, and adjusts them if not.
    :param video: String, The given video input file name.
    :param folder: String, The given output folder name
    :return: String, String; The (potentially adjusted) video input file and output folder.
    """
    # Set default paths based on the current working directory
    if video is None:
        video = os.path.join(os.getcwd(), "10_1-Vid2.mp4")
    if folder is None:
        folder = os.path.join(os.getcwd(), "frames")

    # Create output directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    return video, folder


def video_io(video, output):
    """
    Creates the I/O objects for the video conversion from a given input video and output folder.
    :param video: the path to the source video
    :param output: the path to the output folder
    :return: the CV2 videoCapture object for the source video and the CV2 VideoWriter object for the dest video
    """
    video_path, output_folder = assess_paths(video, output)
    s = cv2.VideoCapture(video_path)
    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(s.get(3))
    frame_height = int(s.get(4))

    # Opening video output file
    size = (frame_width, frame_height)
    video_name = input("Please input the output name for the video file (different than the frames folder):\n")
    r = cv2.VideoWriter('data/output/' + video_name + ".mp4", cv2.VideoWriter_fourcc(*'MP4V'), 30, size, 0)
    return s, r


# split up a video into a collection of images 
# video_path = "/Users/lieselwong/Documents/fishovision/data/10_1-Vid2.mp4"
# output_path = "/Users/lieselwong/Documents/fishovision/data/output"
def get_image_set(video_path, output_folder, frame_interval, brightness_adjust=0, show_video=False):
    """
    Extracts frames from a video and saves them as images.

    Parameters:
        video_path (str): Path to the input video file. Defaults to a file in the current directory if not provided.
        output_folder (str): Folder to save the extracted images. Defaults to a folder in the current directory if not provided.
        frame_interval (int): Save every 'n'-th frame (default is 1, meaning every frame).
        brightness_adjust (int): Adjusts the "value" parameter of the grayscale HSV image frames (which is the brightness of the image).
        show_video (boolean): Whether to show the video on screen while it is processing.
    """

    # Opens the input video file
    source, result = video_io(video_path, output_folder)

    frame_count, saved_count = 0, 0
    while True:
        # extracting the frames
        ret, img = source.read()
        if img is None:
            break
        # converting to gray-scale
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], brightness_adjust)
        gray = hsv[:, :, 2]

        # Save every 5th frame
        if frame_count % frame_interval == 0:
            image_path = os.path.join('data/frames', f"frame_{saved_count: 04d}.jpg")
            cv2.imwrite(image_path, gray)
            saved_count += 1

        # write to gray-scale
        result.write(gray)

        # displaying the video
        if show_video:
            cv2.imshow("Live", gray)

        # exiting the loop
        # key = cv2.waitKey(1)
        # if key == ord("q"):
        #     break
        frame_count += 1

    print(f"Extracted {saved_count} frames and saved to data/frames.")
    # closing the I/O to clean up
    cv2.destroyAllWindows()
    source.release()


# get_image_set(video_path="data/10_1-Vid2.mp4", output_folder="data/frames", frame_interval=5, brightness_adjust=60)

