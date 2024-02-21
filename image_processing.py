import numpy as np
import scipy.ndimage as nd
import skimage as ski


def preprocess(img):
  """Remove most of the background."""
  elevation_map = ski.filters.sobel(img)
  elevation_map = (elevation_map * 255).astype(np.uint8)
  mask_from_elevation_map = (elevation_map[:, :, 2] < 100) & (elevation_map[:, :, 1] > 17) & (elevation_map[:, :, 0] < 150)
  mask_from_image = (img[:, :, 2] < 100) & (img[:, :, 1] > 70) & (img[:, :, 0] < 100)
  mask = mask_from_elevation_map | mask_from_image
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
