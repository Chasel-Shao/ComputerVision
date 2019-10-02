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


def gaussian_kernel(size, sigma):
    constant = 1/(2 * np.pi * sigma * sigma)
    scale = -1 / (2 * sigma * sigma)
    kernel_1d = np.linspace(-(size // 2), size // 2, size)
    kernel_2d = np.zeros(shape=(size,size),dtype=float)
    sum = 0
    for i in range(size):
        for j in range(size):
            a = kernel_1d[i]
            b = kernel_1d[j]
            v = constant * np.e ** ((a*a + b*b) * scale)
            kernel_2d[i][j] = v
            sum += v
    kernel_2d = kernel_2d/sum
    return kernel_2d


# def gaussian_kernel(size, sigma=1):
#     size = int(size) // 2
#     x, y = np.mgrid[-size:size+1, -size:size+1]
#     normal = 1 / (2.0 * np.pi * sigma**2)
#     g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
#     return g