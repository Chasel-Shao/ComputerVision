import cv2
import numpy as np
from convolution_function import *


filepath = '../../resource/lena_gray_512.png'
raw_image = cv2.imread(filepath, cv2.CV_8U)
row, col = raw_image.shape

Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)


Ix = image_convolution(raw_image, Kx)
image_x = np.zeros(Ix.shape)
for i in range(row):
    for j in range(col):
        if Ix[i, j] > 255:
            image_x[i, j] = 255
        elif Ix[i, j] < 0:
            image_x[i, j] = 0
        else:
            image_x[i, j] = Ix[i, j]
cv2.imshow("Ix-image", np.uint8(image_x))


Iy = image_convolution(raw_image, Ky)
image_y = np.zeros(Iy.shape)
for i in range(row):
    for j in range(col):
        if Iy[i, j] > 255:
            image_y[i, j] = 255
        elif Iy[i, j] < 0:
            image_y[i, j] = 0
        else:
            image_y[i, j] = Iy[i, j]
cv2.imshow("Iy_image", np.uint8(image_y))

sobel_image = np.zeros_like(Ix, dtype=float)

for i in range(row):
    for j in range(col):
        sobel_image[i][j] = np.sqrt(Ix[i][j]**2 + Iy[i][j]**2)
        if sobel_image[i][j] > 255:
            sobel_image[i][j] = 255

cv2.imshow("sobel_image", np.uint8(sobel_image))

cv2.waitKey(0)
cv2.destroyAllWindows()

