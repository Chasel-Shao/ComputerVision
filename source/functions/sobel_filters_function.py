import numpy as np
from convolution_function import *

def sobel_filters(image):

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    # Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = image_convolution(image, Kx)
    Iy = image_convolution(image, Ky)

    sobel_image = np.zeros_like(image)
    row, col = sobel_image.shape
    for i in range(row):
        for j in range(col):
            sobel_image[i][j] = np.sqrt(Ix[i][j] ** 2 + Iy[i][j] ** 2)
            if sobel_image[i][j] > 255:
                sobel_image[i][j] = 255

    theta = np.arctan2(Iy, Ix)
    return sobel_image, theta
