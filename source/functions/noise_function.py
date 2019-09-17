import numpy as np
import cv2


def add_uniform_noise(image, scale):
    ret_image = image + scale * np.random.random(image.shape)
    return ret_image


def add_salt_pepper_noise(image, prob):
    salt_peper_noise_image = np.zeros(image.shape, np.float)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                salt_peper_noise_image[i][j] = 0
            elif rdn > thres:
                salt_peper_noise_image[i][j] = 255
            else:
                salt_peper_noise_image[i][j] = image[i][j]
    return salt_peper_noise_image


def add_gaussian_noise(image):
    return cv2.GaussianBlur(image, (9, 9), 5)
