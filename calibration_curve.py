"""Calibration Curve

Draw a calibration curve.
"""
import image_processing as ip
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import skimage as ski
import statistics
from matplotlib.ticker import (MaxNLocator, MultipleLocator)
from natsort import natsorted


def load_images(folder_name):
  """Load images in the folder."""
  folder_path = os.path.join(os.getcwd(), folder_name)
  list_files = os.listdir(folder_path)
  list_files = natsorted(list_files)
  images = []
  for filename in list_files:
    file_path = os.path.join(folder_path, filename)
    images.append(ski.io.imread(file_path))
  return images

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
  return normalized_rgb

def rgb_stdev(imgs):
  """Get the average RGB and standard deviation of images."""
  rgbs = [get_rgb(i) for i in imgs]
  rs = [i[0] for i in rgbs]
  gs = [i[1] for i in rgbs]
  bs = [i[2] for i in rgbs]
  average_rgb = [statistics.mean(rs), statistics.mean(gs), statistics.mean(bs)]
  st_dev = [statistics.stdev(rs), statistics.stdev(gs), statistics.stdev(bs)]
  return average_rgb, st_dev

def comparison(images, ph_range):
  """Display the result of comparison and the RGB value of each one."""
  # Remove background
  processed_images = []
  for i in range(0, len(ph_range)):
    processed_images.append([ip.remove_background(j, ip.preprocess(j)) for j in images[i]])
  # Show RGB values
  pHs = ph_range
  rgb = []
  sd = []
  for i in processed_images:
    color, st_dev = rgb_stdev(i)
    rgb.append(color)
    sd.append(st_dev)
  # Export as excel
  df = pd.DataFrame([[a] + b + c + [d] for a, b, c, d in zip(pHs, rgb, sd, [sum(triplet) / len(triplet) for triplet in sd])],
  columns=['pH', 'R', 'G', 'B', 'SD of R', 'SD of G', 'SD of B', 'Average SD'])
  df.to_excel("excel/pH calibration curve.xlsx", index=False)
  # Plots and errorbars of R
  red = np.array([i[0] for i in rgb])
  red_sd = [i[0] for i in sd]
  p1 = plt.plot(pHs, red, color="lightcoral", marker="o")
  plt.errorbar(pHs, red, red_sd, color="lightcoral", capsize=5)
  # Plots and errorbars of G
  green = np.array([i[1] for i in rgb])
  green_sd = [i[1] for i in sd]
  p2 = plt.plot(pHs, green, color="yellowgreen", marker="D")
  plt.errorbar(pHs, green, green_sd, color="yellowgreen", capsize=5)
  # Plots and errorbars of B
  blue = np.array([i[2] for i in rgb])
  blue_sd = [i[2] for i in sd]
  p3 = plt.plot(pHs, blue, color="cornflowerblue", marker="s")
  plt.errorbar(pHs, blue, blue_sd, color="cornflowerblue", capsize=5)
  # Color bars
  # plt.bar(pHs, [100] * 7, width=0.1, color=[(i[0]/100, i[1]/100, i[2]/100, 0.2) for i in rgb])
  # Add legends
  plt.legend((p1[0], p2[0], p3[0]), ("R", "G", "B"), loc='upper right')
  # Axis range
  plt.axis([2.8, 9.2, 0, 100])
  plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
  plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
  plt.gca().yaxis.set_minor_locator(MultipleLocator(10))
  plt.xlabel("pH")
  plt.ylabel("Percentage of RGB color(%)")
  plt.show()

def main():
  images = []
  ph_range = range(3, 10)
  for i in ph_range:
    images.append(load_images('calibration curve//' + str(i)))
  comparison(images, ph_range)


if __name__ == "__main__":
    main()
