
import cv2
import numpy as np
from noise_function import *
from convolution_function import *
from sobel_filters_function import *


# IMAGE FILTERS METHODS

# Guassian filter
# def gaussian():
#     img = cv2.imread("noise_img.jpg", cv2.IMREAD_ANYCOLOR)
#
#     # Creating the kernel with opencv
#     ksize = 5
#     kradi = np.uint8(ksize / 2)
#     sigma = np.float64(kradi) / 2
#     kernel = cv2.getGaussianKernel(ksize, sigma)
#     kernel = np.repeat(kernel, ksize, axis=1)
#     kernel = kernel * kernel.transpose()
#     kernel = kernel / kernel.sum()
#
#     # Create a copy with black padding
#     imgpadding = np.pad(img, pad_width=2, mode='constant', constant_values=0)
#     filtered = np.zeros(img.shape)
#     gradient = np.zeros(img.shape)
#
#     for i in range(0, img.shape[0]):
#         for j in range(0, img.shape[1]):
#             sub_mat = imgpadding[i - kradi + 2:i + kradi + 3, j - kradi + 2:j + kradi + 3]
#             res_gauss = (np.multiply(kernel, sub_mat)).sum()
#             filtered[i, j] = res_gauss
#
#     cv2.imshow("Original img", np.uint8(img))
#     cv2.imshow("gaussian img", np.uint8(filtered))
#
#     k = cv2.waitKey(0)
#
#     if k == ord('s'):
#         cv2.destroyAllWindows()

# Canny edge detector
def canny():
    # Load an image
    filepath = '../resource/lena_gray_512.png'
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    gx_base = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    gy_base = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Creating the kernel with opencv
    # ksize = 5
    # kradi = np.uint8(ksize / 2)
    # sigma = np.float64(kradi) / 2
    # kernel = cv2.getGaussianKernel(ksize, sigma)
    # kernel = np.repeat(kernel, ksize, axis=1)
    # kernel = kernel * kernel.transpose()
    # kernel = kernel / kernel.sum()
    #
    # # Create a copy with black padding
    # imgpadding = np.pad(img, pad_width=2, mode='constant', constant_values=0)
    # filtered = np.zeros(img.shape)
    # gradient = np.zeros(img.shape)
    #
    # for i in range(0, img.shape[0]):
    #     for j in range(0, img.shape[1]):
    #         sub_mat = imgpadding[i - kradi + 2:i + kradi + 3, j - kradi + 2:j + kradi + 3]
    #         res_gauss = (np.multiply(kernel, sub_mat)).sum()
    #         filtered[i, j] = res_gauss
    #


    filepath = '../resource/lena_gray_512.png'
    # filepath = '../../resource/peppers_gray.png'
    raw_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    sigma = 1
    size = int(2 * (np.ceil(3 * sigma)) + 1)

    # step 1
    kernel = gaussian_kernel(5,1)
    filtered = image_convolution(raw_image, kernel)
    cv2.imshow("smoothed_image", np.uint8(filtered))

    imgpadding_sobel = np.pad(filtered, pad_width=1, mode='constant', constant_values=0)
    kradi = 1
    gradient = np.zeros(img.shape)

    # Sobel
    for i in range(0, filtered.shape[0]):
        for j in range(0, filtered.shape[1]):
            sub_mat = imgpadding_sobel[i - kradi + 1:i + kradi + 2, j - kradi + 1:j + kradi + 2]
            new_gx = np.multiply(gx_base, sub_mat)
            new_gx = np.sum(new_gx)
            new_gy = np.multiply(gy_base, sub_mat)
            new_gy = np.sum(new_gy)
            g = np.sqrt(new_gx * new_gx + new_gy * new_gy)
            filtered[i, j] = g

            # curr_angle = np.abs(np.arctan2(new_gy, new_gx) * 180 / np.pi)
            curr_angle = np.arctan2(new_gy, new_gx) * 180 / np.pi
            if curr_angle < 0: curr_angle += 180.0

            if (curr_angle >= 0 and curr_angle < 22.5) or (curr_angle >= 158.5 and curr_angle <= 180):
                gradient[i, j] = 0
            elif curr_angle >= 22.5 and curr_angle < 112.5:
                gradient[i, j] = 45
            elif curr_angle >= 167.5 and curr_angle < 112.5:
                gradient[i, j] = 90
            elif curr_angle >= 112.5 and curr_angle < 158.5:
                gradient[i, j] = 135

    n_right = -1
    n_left = -1
    imgpadding_supress = np.pad(filtered, pad_width=1, mode='constant', constant_values=0)
    img_copy = np.copy(filtered)

    # sobel_image, theta = sobel_filters(filtered)


    for i in range(0, filtered.shape[0]):
        for j in range(0, filtered.shape[1]):

            if gradient[i, j] == 0:
                n_right = imgpadding_supress[i + 1, j]
                n_left = imgpadding_supress[i - 1, j]
            elif gradient[i, j] == 45:
                n_right = imgpadding_supress[i + 1, j + 1]
                n_left = imgpadding_supress[i - 1, j - 1]
            elif gradient[i, j] == 90:
                n_right = imgpadding_supress[i, j + 1]
                n_left = imgpadding_supress[i, j - 1]
            elif gradient[i, j] == 135:
                n_right = imgpadding_supress[i - 1, j + 1]
                n_left = imgpadding_supress[i + 1, j - 1]

            if n_right > img_copy[i, j] or n_left > img_copy[i, j]:
                filtered[i, j] = 0

    max_th = 50
    min_th = 20
    img_copy = np.copy(filtered)
    cv2.imshow("no-max suppression", np.uint8(img_copy))


    for i in range(0, filtered.shape[0]):
        for j in range(0, filtered.shape[1]):
            if img_copy[i, j] >= max_th:
                filtered[i, j] = 255
            elif img_copy[i, j] >= min_th:
                filtered[i, j] = 0

    imgpadding_final = np.pad(filtered, pad_width=1, mode='constant', constant_values=0)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            sub_mat = imgpadding_final[i - kradi + 1:i + kradi + 2, j - kradi + 1:j + kradi + 2]
            for k in range(0, sub_mat.shape[0]):
                for l in range(0, sub_mat.shape[1]):
                    if sub_mat[k, l] >= max_th:
                        filtered[i, j] = 255
                    elif sub_mat[k, l] <= min_th:
                        filtered[i, j] = 0

    cv2.imshow("Original img", np.uint8(img))
    cv2.imshow("Filtered img", np.uint8(filtered))
    edges = cv2.Canny(img, 125, 200)
    cv2.imshow("CV2 canny img", edges)

    k = cv2.waitKey(0)

    if k == ord('s'):
        cv2.destroyAllWindows()

canny()