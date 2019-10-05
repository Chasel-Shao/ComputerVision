import cv2
import numpy as np
from convolution_function import *
from noise_function import *


def is_diff_sign(a, b):
    if (a > 0 and b < 0) or (a < 0 and b > 0):
        return True
    return False


def is_zero_crossing(image, i, j, thres):
    if i >= 1 and j >= 1:
        v1 = is_diff_sign(image[i - 1][j - 1], image[i + 1][j + 1]) \
             and abs(image[i - 1][j - 1] - image[i + 1][j + 1]) > thres
        v2 = is_diff_sign(image[i - 1][j], image[i + 1][j]) \
             and abs(image[i - 1][j] - image[i + 1][j]) > thres
        v3 = is_diff_sign(image[i + 1][j - 1], image[i - 1][j + 1]) \
             and abs(image[i + 1][j - 1] - image[i - 1][j + 1]) > thres
        v4 = is_diff_sign(image[i][j - 1], image[i][j + 1]) \
             and abs(image[i][j - 1] - image[i][j + 1]) > thres
        if v1 or v2 or v3 or v4: return True
    return False


def log_kernel(size, sigma):
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    kernel = np.zeros(shape=(size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            r = i - size // 2
            c = j - size // 2
            coeffi2 = (r ** 2 + c ** 2 - 2.0 * sigma ** 2) / (sigma ** 4)
            coeffi3 = np.exp(-1 * (r ** 2 + c ** 2) / (2.0 * sigma ** 2))
            kernel[i][j] = normal * coeffi2 * coeffi3
    return kernel


def marr_hildreth(image, sigma):
    size = int(2 * (np.ceil(3 * sigma)) + 1)
    kernel = log_kernel(size, sigma)
    log_image = np.zeros(image.shape, dtype=float)

    row, col = image.shape
    m, n = kernel.shape
    wr = m // 2
    wc = n // 2
    for i in range(wr, row - wr):
        for j in range(wc, col - wc):
            window = image[i-wr:i+wr+1, j-wc:j+wc+1] * kernel
            log_image[i, j] = np.sum(window)

    log_image = log_image.astype(int, copy=False)
    ret_image = np.zeros_like(log_image)
    slop = np.abs(log_image).mean()
    for i in range(row - wr):
        for j in range(col - wc):
            if is_zero_crossing(log_image, i, j, slop):
                ret_image[i][j] = 255
    return log_image, ret_image


filepath = '../../resource/peppers_gray.png'
raw_image = cv2.imread(filepath, cv2.CV_8U)
r, c = raw_image.shape
sigma = 2
size = int(2 * (np.ceil(3 * sigma)) + 1)

gaussian_kernel = gaussian_kernel(size, sigma)
gaussian_image = image_convolution(raw_image, gaussian_kernel)
cv2.imshow("gaussian_image", np.uint8(gaussian_image))

LoG_image, zero_crossing_image = marr_hildreth(gaussian_image, sigma)
cv2.imshow("LoG_image", np.uint8(LoG_image))
cv2.imshow("zero_crossing_image", np.uint8(zero_crossing_image))

cv2.waitKey(0)
cv2.destroyAllWindows()
