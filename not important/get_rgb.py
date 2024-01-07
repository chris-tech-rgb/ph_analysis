import matplotlib.pyplot as plt
import numpy as np
import os
import skimage as ski

from natsort import natsorted
from skimage import color, filters


def except_of_rybg(img):
  # Convert the image to grayscale
  gray_image = color.rgb2gray(img)
  # Apply a Gaussian filter to smooth the image
  smoothed_image = filters.gaussian(gray_image, sigma=1)
  # Define color thresholds for blue, yellow, and red
  blue_mask = (img[:, :, 2] > 100) & (img[:, :, 1] < 150) & (img[:, :, 0] < 150)
  yellow_mask = (img[:, :, 2] > 150) & (img[:, :, 1] > 150) & (img[:, :, 0] < 150)
  red_mask = (img[:, :, 2] < 150) & (img[:, :, 1] < 150) & (img[:, :, 0] > 150)
  green_mask = (img[:, :, 2] < 100) & (img[:, :, 1] > 50) & (img[:, :, 0] < 150)
  # Combine the masks
  combined_mask = blue_mask | yellow_mask | red_mask | green_mask
  # Create an masked image with white background
  masked_image = np.ones_like(img) * 255
  # Set the blue regions to the original color in the output image
  masked_image[combined_mask] = img[combined_mask]

  # Define a threshold for identifying gray regions (adjust as needed)
  gray_threshold = 0.3
  # Create a mask for gray regions
  gray_mask = smoothed_image < gray_threshold
  # Create an output image with white background
  output_image = np.ones_like(masked_image) * 255
  # Set the gray regions to white in the output image
  output_image[gray_mask] = masked_image[gray_mask]

  return output_image


def average_rgb(img):
    # Create a mask for white regions
    white_mask = np.all(img == [255, 255, 255], axis=-1)
    # Invert the mask to select non-white regions
    non_white_mask = ~white_mask
    # Extract non-white pixels from the image
    non_white_pixels = img[non_white_mask]
    # Calculate the average RGB values for non-white regions
    average_rgb = np.mean(non_white_pixels, axis=0)
    return average_rgb


# Load the example image
folder_path = os.path.join(os.getcwd(), 'images')
list_files = os.listdir(folder_path)
list_files = natsorted(list_files)
image_list = []
for filename in list_files:
  filename = os.path.join(folder_path, filename)
  image_list.append(ski.io.imread(filename))
img1 = image_list[0]
img2 = image_list[1]

# Convert the image
output_image1 = except_of_rybg(img1)
output_image2 = except_of_rybg(img2)

# Define layout
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 2), dtype=object)
axes[0, 0] = fig.add_subplot(2, 2, 1)
axes[0, 1] = fig.add_subplot(2, 2, 2)
axes[1, 0] = fig.add_subplot(2, 2, 3)
axes[1, 1] = fig.add_subplot(2, 2, 4)

# Display the original image1
axes[0, 0].imshow(img1)
axes[0, 0].set_title("pH = 7")
axes[0, 0].axis("off")

# Display the processed image1
axes[1, 0].imshow(output_image1)
axes[1, 0].axis("off")

# Display the original image2
axes[0, 1].imshow(img2)
axes[0, 1].set_title("pH = 4")
axes[0, 1].axis("off")

# Display the processed image2
axes[1, 1].imshow(output_image2)
axes[1, 1].axis("off")

fig.tight_layout()
plt.show()
