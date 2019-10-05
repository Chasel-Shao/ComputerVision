import cv2
import numpy as np
from noise_function import *
from convolution_function import *
from sobel_filters_function import *


def nonmax_supression(image, angle):
    M, N = image.shape
    max_image = np.copy(image)
    angle = angle * 180.0 / np.pi
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            degree = angle[i][j]
            # 0°
            if (22.5 > degree > -22.5) or (degree < -157.5 or degree > 157.5):
                if image[i][j] < image[i][j - 1] or image[i][j] < image[i][j + 1]:
                    max_image[i][j] = 0
                # 45°
            elif (22.5 <= degree < 67.5) or (-67.5 <= degree < -22.5):
                if image[i][j] < image[i + 1][j - 1] or image[i][j] < image[i - 1][j + 1]:
                    max_image[i][j] = 0
                # 90°
            elif (67.5 <= degree < 112.5) or (-112.5 <= degree < -67.5):
                if image[i][j] < image[i + 1][j] or image[i][j] < image[i - 1][j]:
                    max_image[i][j] = 0
                # 135°
            elif (112.5 <= degree < 157.5) or (-157.5 <= degree < -112.5):
                if image[i][j] < image[i - 1][j - 1] or image[i][j] < image[i + 1][j + 1]:
                    max_image[i][j] = 0
            else:
                pass

            # if angle[i][j] < 0:
            #     angle[i][j] += 180.0
            # degree = angle[i][j]
            # # 0°
            # if (0 <= degree < 22.5) or (157.5 <= degree < 180.):
            #     if image[i][j] < image[i][j-1] or image[i][j] < image[i][j+1]:
            #         max_image[i][j] = 0
            # # 45°
            # elif 22.5 <= degree < 67.5:
            #     if image[i][j] < image[i+1][j-1] or image[i][j] < image[i-1][j+1]:
            #         max_image[i][j] = 0
            # # 90°
            # elif 67.5 <= degree < 112.5:
            #     if image[i][j] < image[i + 1][j] or image[i][j] < image[i - 1][j]:
            #         max_image[i][j] = 0
            # # 135°
            # elif 112.5 <= degree < 157.5:
            #     if image[i][j] < image[i-1][j-1] or image[i][j] < image[i+1][j+1]:
            #         max_image[i][j] = 0
            # else:
            #     pass
    return max_image


def double_threshold(img, lowThreshold=20, highThreshold=50):
    weak = 75
    strong = 255
    res = np.zeros_like(img, dtype=float)
    strong_i, strong_j = np.where(img >= highThreshold)
    # zeros_i, zeros_j = np.where(img < lowThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, weak, strong


def hysteresis(image, weak, strong=255):
    M, N = image.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if image[i, j] == weak:
                if ((image[i + 1, j - 1] == strong) or (image[i + 1, j] == strong) or (image[i + 1, j + 1] == strong)
                        or (image[i, j - 1] == strong) or (image[i, j + 1] == strong) or (image[i - 1, j - 1] == strong)
                        or (image[i - 1, j] == strong) or (image[i - 1, j + 1] == strong)):
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image


filepath = '../../resource/lena_gray_512.png'
# filepath = '../../resource/peppers_gray.png'
raw_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
sigma = 1
size = int(2 * (np.ceil(3 * sigma)) + 1)

# step 1
gaussian_kernel = gaussian_kernel(size, sigma)
smoothed_image = image_convolution(raw_image, gaussian_kernel)
cv2.imshow("smoothed_image", np.uint8(smoothed_image))

# step 2
derivative_image, theta = sobel_filters(smoothed_image)
cv2.imshow("derivative_image", np.uint8(derivative_image))

# step 3
nonmax_supression_image = nonmax_supression(derivative_image, theta)
cv2.imshow("nonmax_supression_image", np.uint8(nonmax_supression_image))

# step 4
threshold_image, weak, strong = double_threshold(nonmax_supression_image)
cv2.imshow("threshold_image", np.uint8(threshold_image))

# step 5
canny_image = hysteresis(threshold_image, weak, strong)
cv2.imshow("canny_image", np.uint8(canny_image))

cv2.waitKey(0)
cv2.destroyAllWindows()
