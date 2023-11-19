import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage as ski

from natsort import natsorted
from skimage import img_as_float
from skimage import exposure


matplotlib.rcParams['font.size'] = 8


def plot_img(image, axes, bins=256):
    """Plot an image along with its contrast stretching.

    """
    image = img_as_float(image)
    ax_img = axes[0]
    ax_cont = axes[1]

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display contrast stretching
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    ax_cont.imshow(img_rescale, cmap=plt.cm.gray)
    ax_cont.set_axis_off()

    return ax_img, ax_cont


# Load example images
folder_path = os.path.join(os.getcwd(), 'images')
list_files = os.listdir(folder_path)
list_files = natsorted(list_files)
image_list = []
for filename in list_files:
  filename = os.path.join(folder_path, filename)
  image_list.append(ski.io.imread(filename))
num = len(image_list)

# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, num), dtype=object)
axes[0, 0] = fig.add_subplot(2, num, 1)
for i in range(1, num):
    axes[0, i] = fig.add_subplot(2, num, 1+i)
for i in range(0, num):
    axes[1, i] = fig.add_subplot(2, num, num+1+i)

# Display images
counter = 0
for img in image_list:
    ax_img, ax_cont = plot_img(img, axes[:, counter])
    ax_img.set_title('Original image')
    ax_cont.set_title('Contrast stretching')
    counter += 1

# prevent overlap of y-axis labels
fig.tight_layout()
plt.show()