# importing the module 
import cv2 as cv2
import numpy as np
import os

msg= "rool"
print(msg)
# read image
img = cv2.imread('/Users/rachelpyeon/Desktop/fish.jpg')
hh, ww = img.shape[:2]


# Calculate center
start_x = (width - min_dim) // 2
start_y = (height - min_dim) // 2

# define circles
xc = hh // 2
yc = ww // 2

radius= xc

# draw filled circles in white on black background as masks
mask = np.zeros_like(img)
mask = cv2.circle(mask, (xc,yc), radius, (255,255,255), -1)

# put mask into alpha channel of input
result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
result[:, :, 3] = mask[:,:,0]

# save results
cv2.imwrite('fish.png', mask)
cv2.imwrite('fish.png', result)

cv2.imshow('image', img)
cv2.imshow('mask', mask)
cv2.imshow('masked image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()


