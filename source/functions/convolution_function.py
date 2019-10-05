import numpy as np
import math
import cv2


# calculate the mask by for-loop
def image_correlation(image, kernel):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dst = np.zeros(image.shape)
    M, N = image.shape
    kernel_m, kernel_n = kernel.shape
    wr = kernel_m // 2
    wc = kernel_n // 2
    padded_image = np.zeros((M + kernel_n - 1, N + kernel_m - 1))
    padded_image[wc:M + wc, wr:N + wr] = image

    for i in range(wr, M - wr):
        for j in range(wc, N - wc):
            dst[i][j] = np.sum(image[i - wr:i + wr + 1, j - wc:j + wc + 1] * kernel)
            # sum = 0
            # for k in range(m):
            #     for g in range(n):
            #         a = i - wr + k
            #         b = j - wc + g
            #         sum += image[a, b] * kernel[k, g]
            # if sum < 0:
            #     sum = 0
            # elif sum > 255:
            #     sum = 255
            # dst[i][j] = sum
    return dst


def image_convolution(image, kernel):
    kernel = np.flip(kernel)
    return image_convolution(image, kernel)


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
                print(image1[i][j], " ", image2[i][j])
                n += 1
    print("Number of pixel differences: ", n)
