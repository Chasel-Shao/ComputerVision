import numpy as np


def image_hist(image):
    m, n = image.shape
    h = [0.0] * 256
    for i in range(m):
        for j in range(n):
            value = int(image[i, j])
            h[value] += 1
    return np.array(h) / (m * n)


def cumsum(h):
    return [sum(h[:i + 1]) for i in range(len(h))]


def hist_equalization(image):
    h = image_hist(image)
    cdf = np.array(cumsum(h))
    cdf_normalized = np.uint8(255 * cdf)
    m, n = image.shape
    dst = np.zeros_like(image)
    for i in range(m):
        for j in range(0, n):
            value = int(image[i, j])
            dst[i, j] = cdf_normalized[value]
    return dst
