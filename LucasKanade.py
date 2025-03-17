# taken from https://github.com/Utkal97/Object-Tracking/tree/main

import numpy as np
import cv2
import matplotlib.pyplot as plt

''' 
    inRange checks whether the given coordinates line in the given image limits
 coordinates, limits are tuples i.e., (X,Y) 
'''


def inRange(coordinates, limits):
    x, y = coordinates
    x_limit, y_limit = limits
    return 0 <= x < x_limit and 0 <= y < y_limit


'''
opticalFlow calculates the displacements in X and Y directions i.e., 
(u,v) given two consecutive images varying with time
'''


def optical_flow(old_frame, new_frame, window_size, min_quality=0.01):

    max_corners = 10000
    min_distance = 0.1
    feature_list = cv2.goodFeaturesToTrack(old_frame, max_corners, min_quality, min_distance)

    w = int(window_size/2)

    old_frame = old_frame / 255
    new_frame = new_frame / 255

    # Convolve to get gradients w.r.to X, Y and T dimensions
    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])
    kernel_t = np.array([[1, 1], [1, 1]])

    fx = cv2.filter2D(old_frame, -1, kernel_x)              # Gradient over X
    fy = cv2.filter2D(old_frame, -1, kernel_y)              # Gradient over Y
    ft = cv2.filter2D(new_frame, -1, kernel_t) - cv2.filter2D(old_frame, -1, kernel_t)  # Gradient over Time

    u = np.zeros(old_frame.shape)
    v = np.zeros(old_frame.shape)

    for feature in feature_list:        # for every corner
            j, i = feature.ravel()		# get coordinates of the corners (i,j). They are stored in the order j, i
            i, j = int(i), int(j)		# i,j are floats initially

            i_x = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            i_y = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            i_t = ft[i-w:i+w+1, j-w:j+w+1].flatten()

            b = np.reshape(i_t, (i_t.shape[0], 1))
            A = np.vstack((i_x, i_y)).T

            U = np.matmul(np.linalg.pinv(A), b)     # Solving for (u,v) i.e., U

            u[i, j] = U[0][0]
            v[i, j] = U[1][0]
 
    return u, v


'''
Draw the displacement vectors on the image, given (u,v) and save it to the output filepath provided
'''


def drawOnFrame(frame, U, V, output_file):

    line_color = (0, 255, 0)  # Green

    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            u, v = U[i][j], V[i][j]

            if u and v:
                frame = cv2.arrowedLine(frame, (i, j), (int(round(i+u)), int(round(j+v))),
                                        (0, 255, 0),
                                        thickness=1)
    cv2.imwrite(output_file, frame)


'''
Create a plot of the displacement vectors given (u,v) and plot the two images and displacement in a row.
Save the plot to given output filepath
'''


def drawSeparately(old_frame, new_frame, U, V, output_file):

    displacement = np.ones_like(new_frame)
    displacement.fill(255.)             # Fill the displacement plot with White background
    line_color = (0, 0, 0)
    # draw the displacement vectors
    for i in range(new_frame.shape[0]):
        for j in range(new_frame.shape[1]):

            start_pixel = (i, j)
            end_pixel = (int(i+U[i][j]), int(j+V[i][j]))
            # check if there is displacement for the corner and endpoint is in range
            if U[i][j] and V[i][j] and inRange(end_pixel, old_frame.shape):
                displacement = cv2.arrowedLine(displacement, start_pixel, end_pixel, line_color, thickness=2)

    figure, axes = plt.subplots(1,3)
    axes[0].imshow(old_frame, cmap="gray")
    axes[0].set_title("first image")
    axes[1].imshow(new_frame, cmap="gray")
    axes[1].set_title("second image")
    axes[2].imshow(displacement, cmap="gray")
    axes[2].set_title("displacements")
    figure.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", dpi=200)


# Read Input (two images)
'''
img1 = cv2.imread("data/frames/frame_0000.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread("data/frames/frame_0001.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
'''

# Obtain (u,v) from Lucas Kanade's optical flow approach
''' 
Use this format to get the optical flow results in the other file:
U, V = optical_flow(img1, img2, 3, 0.05) 
'''

# Save results
'''
img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
drawSeparately(img1, img2, U, V, "videoFrameOutSeparate.png")
drawOnFrame(img2, U, V, 'VideoFrameOut.png')
'''