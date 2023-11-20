import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import os
import skimage as ski


def mask_without_background(img):
  """Get a mask of the background."""
  elevation_map = ski.filters.sobel(img)
  elevation_map = (elevation_map * 255).astype(np.uint8)
  green_mask = (elevation_map[:, :, 2] < 100) & (elevation_map[:, :, 1] > 20) & (elevation_map[:, :, 0] < 150)
  masked_image = np.ones_like(img) * 255
  masked_image[green_mask] = img[green_mask]
  return masked_image

def remove_background(img, preliminary_mask):
  """Remove the background."""
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

def first_grid(img):
  """Get the first grid."""
  # Turn the image to grayscale
  gray_image = ski.color.rgb2gray(img)
  # Apply a threshold of 0.7 to the image
  threshold = 0.7
  binary_mask = gray_image < threshold
  # Create an all zero values array with the same shape as our binary mask
  binary_mask_image = np.ones_like(img) * 255
  binary_mask_image[binary_mask] = 0
  # Label the objects
  labeled_img, _ = ski.measure.label(binary_mask, return_num=True)
  # Get all objects
  objects = ski.measure.regionprops(labeled_img)
  # Sort the objects
  average_x = np.mean([object['centroid'][0] for object in objects])
  average_y = np.mean([object['centroid'][1] for object in objects])
  sorted_objects = [None] * 4
  for i in objects:
    if i['centroid'][0] < average_x and i['centroid'][1] < average_y:
      sorted_objects[0] = i
    elif i['centroid'][0] < average_x and i['centroid'][1] > average_y:
      sorted_objects[1] = i
    elif i['centroid'][0] > average_x and i['centroid'][1] < average_y:
      sorted_objects[2] = i
    else:
      sorted_objects[3] = i
  # Remove other objects
  other_objects = [obj for obj in sorted_objects[1:]]
  for i in other_objects:
    labeled_img[i.bbox[0]:i.bbox[2], i.bbox[1]:i.bbox[3]]=0
  # Final mask
  final_mask = labeled_img > 0
  final_grid = np.ones_like(img) * 255
  final_grid[final_mask] = img[final_mask]
  return final_grid

def second_grid(img):
  """Get the second grid."""
  # Turn the image to grayscale
  gray_image = ski.color.rgb2gray(img)
  # Apply a threshold of 0.7 to the image
  threshold = 0.7
  binary_mask = gray_image < threshold
  # Create an all zero values array with the same shape as our binary mask
  binary_mask_image = np.ones_like(img) * 255
  binary_mask_image[binary_mask] = 0
  # Label the objects
  labeled_img, _ = ski.measure.label(binary_mask, return_num=True)
  # Get all objects
  objects = ski.measure.regionprops(labeled_img)
  # Sort the objects
  average_x = np.mean([object['centroid'][0] for object in objects])
  average_y = np.mean([object['centroid'][1] for object in objects])
  sorted_objects = [None] * 4
  for i in objects:
    if i['centroid'][0] < average_x and i['centroid'][1] < average_y:
      sorted_objects[0] = i
    elif i['centroid'][0] < average_x and i['centroid'][1] > average_y:
      sorted_objects[1] = i
    elif i['centroid'][0] > average_x and i['centroid'][1] < average_y:
      sorted_objects[2] = i
    else:
      sorted_objects[3] = i
  # Remove other objects
  other_objects = [sorted_objects[0]] + [obj for obj in sorted_objects[2:]]
  for i in other_objects:
    labeled_img[i.bbox[0]:i.bbox[2], i.bbox[1]:i.bbox[3]]=0
  # Final mask
  final_mask = labeled_img > 0
  final_grid = np.ones_like(img) * 255
  final_grid[final_mask] = img[final_mask]
  return final_grid

def third_grid(img):
  """Get the third grid."""
  # Turn the image to grayscale
  gray_image = ski.color.rgb2gray(img)
  # Apply a threshold of 0.7 to the image
  threshold = 0.7
  binary_mask = gray_image < threshold
  # Create an all zero values array with the same shape as our binary mask
  binary_mask_image = np.ones_like(img) * 255
  binary_mask_image[binary_mask] = 0
  # Label the objects
  labeled_img, _ = ski.measure.label(binary_mask, return_num=True)
  # Get all objects
  objects = ski.measure.regionprops(labeled_img)
  # Sort the objects
  average_x = np.mean([object['centroid'][0] for object in objects])
  average_y = np.mean([object['centroid'][1] for object in objects])
  sorted_objects = [None] * 4
  for i in objects:
    if i['centroid'][0] < average_x and i['centroid'][1] < average_y:
      sorted_objects[0] = i
    elif i['centroid'][0] < average_x and i['centroid'][1] > average_y:
      sorted_objects[1] = i
    elif i['centroid'][0] > average_x and i['centroid'][1] < average_y:
      sorted_objects[2] = i
    else:
      sorted_objects[3] = i
  # Remove other objects
  other_objects = [sorted_objects[3]] + [obj for obj in sorted_objects[:2]]
  for i in other_objects:
    labeled_img[i.bbox[0]:i.bbox[2], i.bbox[1]:i.bbox[3]]=0
  # Final mask
  final_mask = labeled_img > 0
  final_grid = np.ones_like(img) * 255
  final_grid[final_mask] = img[final_mask]
  return final_grid

def fourth_grid(img):
  """Get the fourth grid."""
  # Turn the image to grayscale
  gray_image = ski.color.rgb2gray(img)
  # Apply a threshold of 0.7 to the image
  threshold = 0.7
  binary_mask = gray_image < threshold
  # Create an all zero values array with the same shape as our binary mask
  binary_mask_image = np.ones_like(img) * 255
  binary_mask_image[binary_mask] = 0
  # Label the objects
  labeled_img, _ = ski.measure.label(binary_mask, return_num=True)
  # Get all objects
  objects = ski.measure.regionprops(labeled_img)
  # Sort the objects
  average_x = np.mean([object['centroid'][0] for object in objects])
  average_y = np.mean([object['centroid'][1] for object in objects])
  sorted_objects = [None] * 4
  for i in objects:
    if i['centroid'][0] < average_x and i['centroid'][1] < average_y:
      sorted_objects[0] = i
    elif i['centroid'][0] < average_x and i['centroid'][1] > average_y:
      sorted_objects[1] = i
    elif i['centroid'][0] > average_x and i['centroid'][1] < average_y:
      sorted_objects[2] = i
    else:
      sorted_objects[3] = i
  # Remove other objects
  other_objects = [obj for obj in sorted_objects[:3]]
  for i in other_objects:
    labeled_img[i.bbox[0]:i.bbox[2], i.bbox[1]:i.bbox[3]]=0
  # Final mask
  final_mask = labeled_img > 0
  final_grid = np.ones_like(img) * 255
  final_grid[final_mask] = img[final_mask]
  return final_grid

def display_segmentation(img, image_name):
  """Display the result of segmentation and the RGB value of each segment."""
  # Remove background
  processed_image = remove_background(img, mask_without_background(img))
  # Segmentation
  segments = [None] * 4
  segments[0] = first_grid(processed_image)
  segments[1] = second_grid(processed_image)
  segments[2] = third_grid(processed_image)
  segments[3] = fourth_grid(processed_image)
  # Display the image
  fig = plt.figure(figsize=(8, 8))
  axes = np.zeros((2, 5), dtype=object)
  for i in range(0, 5):
    axes[0, i] = fig.add_subplot(2, 5, 1+i)
    axes[0, i].axis("off")
  axes[1, 0] = fig.add_subplot(2, 1, 2)
  # Show processed image
  axes[0, 0].imshow(processed_image)
  axes[0, 0].set_title(image_name)
  # Show segmentation 1
  axes[0, 1].imshow(segments[0])
  axes[0, 1].set_title(image_name+" (1)")
  # Show segmentation 2
  axes[0, 2].imshow(segments[1])
  axes[0, 2].set_title(image_name+" (2)")
  # Show segmentation 3
  axes[0, 3].imshow(segments[2])
  axes[0, 3].set_title(image_name+" (3)")
  # Show segmentation 4
  axes[0, 4].imshow(segments[3])
  axes[0, 4].set_title(image_name+" (4)")
  # Show RGB values
  number = np.array([1, 2, 3, 4, 5])
  rgb = [average_rgb(i) for i in [processed_image]+segments]
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

def main():
  image_dict = load_images('images')
  for i in image_dict:
    display_segmentation(image_dict[i], i)


if __name__ == "__main__":
    main()
