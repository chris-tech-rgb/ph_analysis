import matplotlib.pyplot as plt
import numpy as np
import os
import skimage as ski


def mask_without_background(img):
  elevation_map = ski.filters.sobel(img)
  elevation_map = (elevation_map * 255).astype(np.uint8)
  green_mask = (elevation_map[:, :, 2] < 100) & (elevation_map[:, :, 1] > 20) & (elevation_map[:, :, 0] < 150)
  masked_image = np.ones_like(img) * 255
  masked_image[green_mask] = img[green_mask]
  return masked_image


def remove_all_background(img, preliminary_mask):
  # Turn the image to grayscale
  gray_image = ski.color.rgb2gray(preliminary_mask)
  # Denoise the image
  blurred_image = ski.filters.gaussian(gray_image, sigma=4)
  # Apply a threshold of 0.7 to the image
  threshold = 0.7
  binary_mask = blurred_image < threshold
  # Create an all zero values array with the same shape as our binary mask
  binary_mask_image = np.ones_like(img) * 255
  binary_mask_image[binary_mask] = 0
  # Label the objects
  labeled_image, _ = ski.measure.label(binary_mask, return_num=True)
  # Find the areas of the objects
  objects = ski.measure.regionprops(labeled_image)
  object_areas = [obj["area"] for obj in objects]
  fourth_largest = np.partition(object_areas, -4)[-4]
  # Remove small objects
  small_objects =[obj for obj in objects if obj.area<fourth_largest]
  for i in small_objects:
    labeled_image[i.bbox[0]:i.bbox[2], i.bbox[1]:i.bbox[3]]=0
  # Final mask
  final_mask = labeled_image > 0
  # Output
  output_image = np.ones_like(img) * 255
  output_image[final_mask] = img[final_mask]
  return output_image


# Load the images
folder_path = os.path.join(os.getcwd(), 'images')
filename = os.path.join(folder_path, '2.jpg')
image = ski.io.imread(filename)

output_image = remove_all_background(image, mask_without_background(image))

# Display the image
plt.figure(figsize=(8, 5))
plt.imshow(output_image)
plt.axis("off")
plt.show()