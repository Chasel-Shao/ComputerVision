# coding=gbk
import matplotlib.pyplot as plt
from convolution_function import *
from noise_function import *
from PIL import Image
import cv2



filepath = '../resource/lena_gray_512.png'
raw_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
Kx = np.array(([1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]))

Ky = np.array(([1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]))

Ky2 = np.array(([-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]))

def test_sobel_filter(Kx, Ky):
    Ix = image_convolution(Kx, raw_image, opt=False)
    Iy = image_convolution(Ky, raw_image, opt=True)

    sobel_image = np.zeros_like(raw_image)
    row, col = sobel_image.shape
    for i in range(row):
        for j in range(col):
            sobel_image[i][j] = np.sqrt(Ix[i][j] ** 2 + Iy[i][j] ** 2)
    sobel_image = sobel_image / sobel_image.max() * 255
    theta = np.arctan2(Iy, Ix)
    return sobel_image, theta

image1, theta1 = test_sobel_filter(Kx, Ky)
image2, theta2 = test_sobel_filter(Kx, Ky2)


# image1 = image_convolution(Ky, raw_image)
# image2 = image_convolution(Ky2, raw_image)

calculate_number_of_diff_pixel(image1, image2, 3)