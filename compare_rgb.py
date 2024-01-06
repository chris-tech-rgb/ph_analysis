import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import os
import re
import scipy.ndimage as nd
import skimage as ski


def mask_without_background(img):
  """Get a mask of the background."""
  elevation_map = ski.filters.sobel(img)
  elevation_map = (elevation_map * 255).astype(np.uint8)
  green_mask = (elevation_map[:, :, 2] < 100) & (elevation_map[:, :, 1] > 12) & (elevation_map[:, :, 0] < 150)
  masked_image = np.ones_like(img) * 255
  masked_image[green_mask] = img[green_mask]
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
    third_largest = np.partition(object_areas, -3)[-3]
    # Remove small objects
    small_objects =[obj for obj in objects if obj.area<third_largest]
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

def average_rgb(img):
    """Get the average RGB of an image."""
    # Create a mask for white regions
    white_mask = np.all(img == [255, 255, 255], axis=-1)
    # Invert the mask to select non-white regions
    non_white_mask = ~white_mask
    # Extract non-white pixels from the image
    non_white_pixels = img[non_white_mask]
    # Calculate the average RGB values for non-white regions
    average_rgb = np.mean(non_white_pixels, axis=0)
    return average_rgb

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
    axes[0, i].set_title(image_names[i])
  axes[1, 0] = fig.add_subplot(2, 1, 2)
  # Show RGB values
  number = np.array([float(re.findall(r'\d+\.\d+', i)[0]) for i in image_names])
  rgb = []
  for i in image_names:
    rgb.append(average_rgb(processed_images[i]))
  # Show values of R
  red = np.array([i[0] for i in rgb])
  p1 = axes[1, 0].plot(number, red, color="red", marker="o")
  for a,b in zip(number, red): 
    axes[1, 0].text(a, b, str("{:.2f}".format(b)), color="red")
  # Show values of G
  green = np.array([i[1] for i in rgb])
  p2 = axes[1, 0].plot(number, green, color="green", marker="D")
  for a,b in zip(number, green): 
    axes[1, 0].text(a, b, str("{:.2f}".format(b)), color="green")
  # Show values of B
  blue = np.array([i[2] for i in rgb])
  p3 = axes[1, 0].plot(number, blue, color="blue", marker="s")
  for a,b in zip(number, blue): 
    axes[1, 0].text(a, b, str("{:.2f}".format(b)), color="blue")
  # Add legends
  axes[1, 0].legend((p1[0], p2[0], p3[0]), ("R", "G", "B"), loc='center left', bbox_to_anchor=(1, 0.5))
  plt.show()

def main():
  image_dict = load_images('ph test')
  comparison(image_dict)


if __name__ == "__main__":
    main()
