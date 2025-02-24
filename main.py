import numpy as np
import cv2 as cv


cap = cv.VideoCapture('10_1-Vid2.mp4')
fourcc = cv.VideoWriter_fourcc(*'MP4V')
out = cv.VideoWriter('output.mp4', fourcc, 30.0, (1620, 1080))


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # Display the resulting frame
    cv.imshow('frame', gray)

    # write the flipped frame
    out.write(frame)

    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
out.release()
cv.destroyAllWindows()