
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from histeq_function import *

filename = '../../resource/part1#3.png'
im = np.array(Image.open(filename).convert('L'))
im_grey = im
# image_red = im[:, :, 0]
# image_green = im[:, :, 1]
# image_blue = im[:, :, 2]
# im_grey = 0.21 * image_red + 0.72 * image_green + 0.07 * image_blue;
new_img = hist_equalization(im_grey)


plt.subplot(321)
plt.imshow(im_grey)
plt.title('Original image')
plt.set_cmap('gray')

plt.subplot(322)
plt.imshow(new_img)
plt.title('Equalized image')
plt.set_cmap('gray')


plt.subplot(323)
plt.hist(im_grey.flatten(), bins=256, range=(0, 256), density=True, cumulative=False)
plt.title('Original histogram')

plt.subplot(324)
plt.hist(new_img.flatten(), bins=256, range=(0, 256), density=True, cumulative=False)
plt.title('Equalized histogram')


h = image_hist(im)
cdf = np.array(cumsum(h))
sk = np.uint8(255 * cdf)

plt.subplot(325)
plt.plot(sk)
plt.title('Cumulative histogram')


plt.show()


