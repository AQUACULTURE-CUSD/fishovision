import cv2
import numpy as np


def kmeans_partition(image, k):
    """
    Partitions the image into k sets based on k-means clustering.
    :param image: the image to be partitioned
    :param k: the number of clusters
    :return: a partitioned image
    """
    # Change color to RGB (from BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = image.reshape((-1, 3))  # numpy reshape operation -1 unspecified

    # Convert to float type only for supporting cv2.kmean
    pixel_vals = np.float32(pixel_vals)

    # criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert data into 8-bit values
    centers = np.uint8(centers)

    segmented_data = centers[labels.flatten()]  # Mapping labels to center points( RGB Value)

    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape(image.shape)
    return segmented_image

# load image from images directory
img = cv2.imread('data/frames/frame_ 000.jpg')
simg = kmeans_partition(img, 10)
cv2.namedWindow("Image")
cv2.imshow('Image',img)
cv2.waitKey(0)
