import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import os
import re
import scipy.ndimage as nd
import skimage as ski
import statistics


def mask_without_background(img):
  """Get a mask without the background."""
  elevation_map = ski.filters.sobel(img)
  elevation_map = (elevation_map * 255).astype(np.uint8)
  green_mask = (elevation_map[:, :, 2] < 100) & (elevation_map[:, :, 1] > 12) & (elevation_map[:, :, 0] < 150)
  anti_green_mask = (img[:, :, 2] < 100) & (img[:, :, 1] > 70) & (img[:, :, 0] < 100)
  mask = green_mask | anti_green_mask
  masked_image = np.ones_like(img) * 255
  masked_image[mask] = img[mask]
  return masked_image

def remove_background(img, preliminary_mask):
  """Remove the background."""
  # Turn the image to grayscale
  gray_image = ski.color.rgb2gray(preliminary_mask)
  # Denoise the image
  blurred_image = ski.filters.gaussian(gray_image, sigma=4)
  # Apply a threshold of 0.8 to the image
  threshold = 0.8
  binary_mask = blurred_image < threshold
  # Create an all zero values array with the same shape as our binary mask
  binary_mask_image = np.ones_like(img) * 255
  binary_mask_image[binary_mask] = 0
  # Label the objects
  labeled_image, _ = ski.measure.label(binary_mask, return_num=True)
  # Get the areas of the objects to remove small objects
  objects = ski.measure.regionprops(labeled_image)
  object_areas = [obj["area"] for obj in objects]
  if len(object_areas) > 2:
    largest = np.partition(object_areas, -1)[-1]
    # Remove small objects
    small_objects =[obj for obj in objects if obj.area<largest]
    for i in small_objects:
      labeled_image[i.bbox[0]:i.bbox[2], i.bbox[1]:i.bbox[3]]=0
  # Fill holes
  solid_labels = nd.binary_fill_holes(labeled_image).astype(int)
  # Final mask
  final_mask = solid_labels > 0
  # Output
  output_image = np.ones_like(img) * 255
  output_image[final_mask] = img[final_mask]
  return output_image

def load_images(folder_name):
  """Load images in the folder."""
  folder_path = os.path.join(os.getcwd(), folder_name)
  list_files = os.listdir(folder_path)
  list_files = natsorted(list_files)
  image_dict = {}
  for filename in list_files:
    file_path = os.path.join(folder_path, filename)
    image_dict[filename] = ski.io.imread(file_path)
  return image_dict

def get_rgb(img):
  """Get the average RGB of an image."""
  # Create a mask for white regions
  white_mask = np.all(img == [255, 255, 255], axis=-1)
  # Invert the mask to select non-white regions
  non_white_mask = ~white_mask
  # Extract non-white pixels from the image
  non_white_pixels = img[non_white_mask]
  # Calculate the average RGB values for non-white regions
  average_rgb = np.mean(non_white_pixels, axis=0)
  normalized_rgb = [i*100/255 for i in average_rgb]
  # Calculate the standard deviation of each color
  st_dev = [statistics.stdev([p[i]*20/255 for p in non_white_pixels]) for i in range(3)]
  return normalized_rgb, st_dev

def comparison(imgs):
  """Display the result of comparison and the RGB value of each one."""
  # Count number
  number = len(imgs)
  if number < 1:
    return
  # Remove background
  image_names = list(imgs.keys())
  processed_images = {}
  for i in image_names:
    processed_images[i] = remove_background(imgs[i], mask_without_background(imgs[i]))
  # Display images
  fig = plt.figure(figsize=(8, 8))
  axes = np.zeros((2, number), dtype=object)
  for i in range(0, number):
    axes[0, i] = fig.add_subplot(2, number, 1+i)
    axes[0, i].axis("off")
    axes[0, i].imshow(processed_images[image_names[i]])
    axes[0, i].set_title(image_names[i][:-4])
  axes[1, 0] = fig.add_subplot(2, 1, 2)
  # Show RGB values
  pHs = np.array([float(re.findall(r'\d+\.\d+', i)[0]) for i in image_names])
  rgb = []
  sd = []
  for i in image_names:
    color, st_dev = get_rgb(processed_images[i])
    rgb.append(color)
    sd.append(st_dev)
  # Plots and errorbars of R
  red = np.array([i[0] for i in rgb])
  red_sd = [i[0] for i in sd]
  p1 = axes[1, 0].plot(pHs, red, color="lightcoral", marker="o")
  axes[1, 0].errorbar(pHs, red, red_sd, color="lightcoral", capsize=5)
  # Plots and errorbars of G
  green = np.array([i[1] for i in rgb])
  green_sd = [i[1] for i in sd]
  p2 = axes[1, 0].plot(pHs, green, color="yellowgreen", marker="D")
  axes[1, 0].errorbar(pHs, green, green_sd, color="yellowgreen", capsize=5)
  # Plots and errorbars of B
  blue = np.array([i[2] for i in rgb])
  blue_sd = [i[2] for i in sd]
  p3 = axes[1, 0].plot(pHs, blue, color="cornflowerblue", marker="s")
  axes[1, 0].errorbar(pHs, blue, blue_sd, color="cornflowerblue", capsize=5)
  # for a, b in zip(pHs, red): 
  #   axes[1, 0].text(a, b, str("{:.2f}".format(b)), color="lightcoral")
  # for a, b in zip(pHs, green): 
  #   axes[1, 0].text(a, b, str("{:.2f}".format(b)), color="yellowgreen")
  # for a, b in zip(pHs, blue): 
  #  axes[1, 0].text(a, b, str("{:.2f}".format(b)), color="cornflowerblue")
  # Add legends
  axes[1, 0].legend((p1[0], p2[0], p3[0]), ("R", "G", "B"), loc='upper right')
  # axis label
  axes[1, 0].set_ylabel('Percentage of RGB color (%)')
  axes[1, 0].set_xlabel('pH')
  plt.show()

def main():
  image_dict = load_images('ph test 3')
  comparison(image_dict)


if __name__ == "__main__":
    main()
