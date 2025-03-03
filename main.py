# importing the module 
import cv2
import numpy as np


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


# running the loop 
while True:

    # extracting the frames 
    ret, img = source.read()
    if img is None:
        break
    # converting to gray-scale
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = 42  # whatever value you want to add
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

#comment