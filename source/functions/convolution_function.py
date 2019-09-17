import numpy as np
import math


# calculate the mask by for-loop
def image_convolution(kernel, image):
    dst = np.copy(image)
    row, col = image.shape
    m, n = kernel.shape
    wr = math.floor(m / 2)
    wc = math.floor(n / 2)
    for i in range(wr, row - wr):
        for j in range(wc, col - wc):
            sum = 0
            for k in range(m):
                for g in range(n):
                    a = i - wr + k
                    b = j - wc + g
                    sum += image[a, b] * kernel[k, g]
            dst[i][j] = sum
    return dst


# calculate mask by the mean method
def image_convolution_mean(kernel, image):
    dst = np.copy(image)
    row, col = image.shape
    m, n = kernel.shape
    wr = math.floor(m / 2)
    wc = math.floor(n / 2)
    for i in range(wr, row - wr):
        for j in range(wc, col - wc):
            mean = np.mean(image[i - wr:i + wr + 1, j - wc: j + wc + 1])
            dst[i][j] = mean
    return dst


def calculate_number_of_diff_pixel(image1, image2, margin):
    n = 0
    width, height = image1.shape
    for i in range(margin, height - margin):
        for j in range(margin, width - margin):
            if abs(image1[i][j] - image2[i][j]) > 0.0000001:
                n += 1
    print("Number of pixel differences: ", n)
