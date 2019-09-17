# coding=gbk

from convolution_function import *
from PIL import Image
import cv2

size = 7
dataSize = 500
# pic = np.random.randint(0, 256, size=(dataSize, dataSize)).astype(float)
# print(pic)
filename = '../resource/part2#1.png'
raw_image = np.array(Image.open(filename).convert('L')).astype(float)
dataSize, x = raw_image.shape



kernel = np.mat(np.ones(shape=(size, size), dtype=float) / (size * size))
m, n = kernel.shape
mean_smoothed_image = image_convolution_mean(kernel, raw_image)

# print("array1")
# print(array1)
# print("")
# print("")

f1 = np.mat(np.ones(shape=(1, size), dtype=float) / size)
temp_image = image_convolution_mean(f1, raw_image)
# print("array2:")
# print(array2)


f2 = np.mat(np.ones(shape=(size, 1), dtype=float) / size)
separable_smoothed_image = image_convolution_mean(f2, temp_image)
# print("array3:")
# print(array3)

dst = cv2.filter2D(raw_image, -1, kernel)

calculate_number_of_diff_pixel(separable_smoothed_image, dst, size)
