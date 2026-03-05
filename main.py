# importing the module
from __future__ import annotations
from processing_steps import *
import cv2
from processing_steps import grayscale

def run_main(pipeline_steps):
    # load in the pipeline/analysis steps
    cv_pipeline = Pipeline(pipeline_steps)
    # load in the video you want to analyze
    cap = cv2.VideoCapture('data/11_18-Vid11.mov')  # Or 0 for webcam
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    print(cap.get(cv2.CAP_PROP_FPS))
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # skips frames and reads only 0, 10, 20, etc 
        if (frame_num % 10) != 0:
            frame_num += 1
            continue
        # Prepare the context for this frame
        process_context = {
            'original_frame': frame.copy(),
            'current_frame': frame.copy(),
            'frame_number': frame_num
        }
        # Run the pipeline
        context = cv_pipeline.run(process_context)
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
        context = cv_pipeline.run(process_context)


'''
 pipeline steps, associated with files. 
 Refer to comments or https://docs.google.com/document/d/1XnaYBOWkOAfA_JPGirxE1JfnxZRbWAsDbrj2eBJtMsM/edit?usp=sharing '''
if __name__ == "__main__":
    pipeline_steps = [
        # MedianFilter(),
        CircleCrop(center=(-50, -30), r=470),
        LabColorSegmentationMask(),
        ApplyMaskDenoised((7,7)),
        # ShowCurrentImage(),
        GrayscaleConverter(),
        LinearContrastAdjuster(1.4),
        # MidToneThresholdMask(20, 190),
        # ApplyMaskDenoised((7, 7)),
        CropLine(-0.5, 1250, reverse=True),
        # MidToneThresholdDenoised(10, 200, 7),
        # BrightnessAdjuster(30),
        OpticalFlowCalculator(0.2),
        # Visualize(1),
        GraphData("output.csv", 30, 20)
    ]
    run_main(pipeline_steps)

    # test_one_image(pipeline_steps)
