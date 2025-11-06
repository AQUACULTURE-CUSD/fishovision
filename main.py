# importing the module
from __future__ import annotations
import time
from processing_steps import *

import cv2
import os
import circle_tc
import LucasKanade
from processing_steps import grayscale


def assess_paths(video, folder, folder1):
    """
    Checks whether the video and output folder paths are good, and adjusts them if not.
    :param video: String, The given video input file name.
    :param folder: String, The given output folder name
    :return: String, String; The (potentially adjusted) video input file and output folder.
    """
    # Set default paths based on the current working directory
    if video is None:
        video = os.path.join(os.getcwd(), "11_18-vid11(1).mp4")
    if folder is None:
        folder = os.path.join(os.getcwd(), "frames")
    if folder1 is None:
        folder1 = os.path.join(os.getcwd(), "cropped_frames")

    # Create output directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    return video, folder, folder1


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

def cropped_circles_test():
    # Define input and output folders
    input_folder = os.path.join(os.getcwd(), 'data/frames')
    output_folder = os.path.join(os.getcwd(), 'data/cropped_frames')
    os.makedirs(output_folder, exist_ok=True)

    # Establish frames folder
    frame_folder = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    saved_count = 0
    start_t = time.time()
    for file_name in frame_folder:
        # Establish image path
        saved_count += 1
        image_path = os.path.join(input_folder, file_name)

        # Read the image
        frame = cv2.imread(image_path)

        # Implement function
        cropped = circle_tc.crop_center_circle(frame)

        # Save image to new folder
        output_path = os.path.join(output_folder, f"cropped_frame_{saved_count:04d}.jpg")
        cv2.imwrite(output_path, cropped)

    length = time.time() - start_t
    print(length, "seconds to crop all frames.")
    print(length / saved_count, "seconds per frame.")


def lucas_kanade_test():
    frame_folder = [f for f in os.listdir('data/cropped_frames') if f.endswith('.jpg')]
    output_folder = os.path.join(os.getcwd(), 'data/lucas_kanade_frames')
    os.makedirs(output_folder, exist_ok=True)
    saved_count = 0
    start_t = time.time()
    for file_index in range(len(frame_folder) - 1):
        image_path1 = os.path.join('data/cropped_frames', frame_folder[file_index])
        image_path2 = os.path.join('data/cropped_frames', frame_folder[file_index + 1])
        # Read the image
        frame1 = cv2.imread(image_path1)
        frame2 = cv2.imread(image_path2)

        # displacements = LucasKanade.displacements(frame1, frame2)
        displacements = LucasKanade.drawOnFrameWrapper(frame1, frame2)

        # Save image to new folder
        output_path = os.path.join(output_folder, f"displacement_frame{saved_count:04d}.jpg")
        cv2.imwrite(output_path, displacements)
        saved_count += 1

    length = time.time() - start_t
    print(length, "seconds to displace all frames.")
    print(length / saved_count, "seconds per displacement frame.")


def images_to_video():
    image_folder = 'data/lucas_kanade_frames'
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def run_main(pipeline_steps):
    cv_pipeline = Pipeline(pipeline_steps)
    # CHANGE THIS LINE TO RUN DIFFERENT VIDEO FILE
    cap = cv2.VideoCapture('fishovision/data/11_18-Vid11(1).mp4')  # Or 0 for webcam
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_num % 10) != 0:
            frame_num += 1
            continue
        # 4. Prepare the context for this frame
        process_context = {
            'original_frame': frame.copy(),
            'current_frame': frame.copy(),
            'frame_number': frame_num
        }

        # 5. Run the pipeline
        _ = cv_pipeline.run(process_context)

        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()


def test_one_image(pipeline_steps):
    frames = [cv2.imread('data/frames/frame_ 000.jpg'), cv2.imread('data/frames/frame_ 005.jpg')]
    cv_pipeline = Pipeline(pipeline_steps)
    for frame in frames:
        # 4. Prepare the context for this frame
        process_context = {
            'original_frame': frame.copy(),
            'current_frame': frame.copy(),
            'frame_number': 0
        }

        # 5. Run the pipeline
        _ = cv_pipeline.run(process_context)


if __name__ == "__main__":
    pipeline_steps = [
        # MedianFilter(),
        CircleCrop(center=(-50, -30), r=470),
        LabColorSegmentationMask(),
        ApplyMaskDenoised((7,7)),
        GrayscaleConverter(),
        LinearContrastAdjuster(1.4),
        # MidToneThresholdMask(20, 190),
        # ApplyMaskDenoised((7, 7)),
        CropLine(-0.5, 1250, reverse=True),

        # MidToneThresholdDenoised(10, 200, 7),
        # BrightnessAdjuster(30),
        # ShowCurrentImage(),


        OpticalFlowCalculator(0.2),
        # Visualize(),
        # TODO: Change it so it outputs to csv
        GraphData("output.csv")

    ]
    run_main(pipeline_steps)

    # test_one_image(pipeline_steps)
    # images_to_video()
    # lucas_kanade_test()
