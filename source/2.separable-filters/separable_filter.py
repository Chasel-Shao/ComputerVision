import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from noise_function import add_uniform_noise, add_salt_pepper_noise, add_gaussian_noise
from convolution_function import image_convolution, calculate_number_of_diff_pixel

# load image from local file
filename = '../resource/part2#1.png'
raw_image = np.array(Image.open(filename).convert('L')).astype(float)

size = 7
mean_kernel = np.mat(np.ones(shape=(size, size), dtype=float)) / (size * size)
separable_kernel_1 = np.mat(np.ones(shape=(1, size), dtype=float)) / size
separable_kernel_2 = np.mat(np.ones(shape=(size, 1), dtype=float)) / size


# 1. show the original image
plt.subplot(321)
plt.set_cmap('gray')
plt.title('Original Image')
plt.imshow(raw_image)

# 2. show the uniform noisy image
scale = 100
plt.subplot(322)
plt.set_cmap('gray')
uniform_noise_image = add_uniform_noise(raw_image, scale)
plt.title('Uniform noise Image')

# prob = 0.05
# uniform_noise_image = add_salt_pepper_noise(raw_image, prob)
# plt.title('Salt&Pepper noise Image')

# uniform_noise_image = add_gaussian_noise(raw_image)
# plt.title('Gaussian noise Image')

plt.imshow(uniform_noise_image)

# 3. show the salt&pepper noisy image
prob = 0.05
salt_peper_noise_image = add_salt_pepper_noise(raw_image, prob)
plt.subplot(323)
plt.set_cmap('gray')
plt.title('Salt&Pepper noise Image')
plt.imshow(salt_peper_noise_image)

# 4. show the gaussian noisy image
gaussian_noise_image = add_gaussian_noise(raw_image)
plt.subplot(324)
plt.set_cmap('gray')
plt.title('Gaussian noise Image')
plt.imshow(gaussian_noise_image)


# 5. show the mean filter image
start_time = time.time()
mean_filter_image = image_convolution(mean_kernel, uniform_noise_image)
end_time = time.time() - start_time
print("mean filter cost time: ", end_time)

plt.subplot(325)
plt.set_cmap('gray')
plt.title('Mean Filter')
plt.imshow(mean_filter_image)

# 6. show the separable mean filter image
start_time = time.time()
temp_image = image_convolution(separable_kernel_1, uniform_noise_image)
separable_filter_image = image_convolution(separable_kernel_2, temp_image)
end_time = time.time() - start_time
print("separable filter cost time: ", end_time)

plt.subplot(326)
plt.set_cmap('gray')
plt.title('Separable Mean Filter')
plt.imshow(separable_filter_image)

plt.show()

# calculate the differences between two images
calculate_number_of_diff_pixel(separable_filter_image, mean_filter_image, size)





