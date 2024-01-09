import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
 
# Fitting function
def func(rgb, a1, a2, b1, b2, c1, c2):
    return a1 * rgb[:, 0]**a2 + b1 * rgb[:, 1]**b2 + c1 * rgb[:, 2]**c2
 
# Experimental data points
pH_values = np.array([4.65, 5.50, 5.95, 6.40, 6.70, 7.00, 7.60, 8.00])
rgb_values = np.array([[146.29, 135.73, 50.21],
                       [82.79, 106.04, 54.27],
                       [68.93, 94.55, 51.15],
                       [40.13, 75.40, 48.58],
                       [40.87, 80.64, 66.19],
                       [28.64, 71.22, 72.75],
                       [26.41, 60.29, 85.66],
                       [16.36, 42.82, 73.87]])
r = rgb_values[:, 0]
g = rgb_values[:, 1]
b = rgb_values[:, 2]

# Perform the curve-fit
popt, pcov = curve_fit(func, rgb_values, pH_values, maxfev = 20000)
a1, a2, b1, b2, c1, c2 = popt
print("Calibration Curves:\npH = " + str(a1) + " × R^(" + str(a2) + ") + " + str(b1) + " × G^(" + str(b2) + ") + " + str(c1) + " × B^(" + str(c2) + ")")

# Create figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Fitting
r_fit = np.linspace(min(r), max(r), 16)
g_fit = np.linspace(min(g), max(g), 16)
b_fit = np.linspace(min(b), max(b), 16)
R, G, B = np.meshgrid(r_fit, g_fit, b_fit)
pH = a1 * R**a2 + b1 * G**b2 + c1 * B**c2

# Scatter plot
img = ax.scatter(R, G, B, c=pH, cmap='viridis_r')

# Colorbar
colorbar = fig.colorbar(img, location='left')  # Set position to 'left'
colorbar.set_label('pH Values')

# Set labels
ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')
# Color of labels
ax.xaxis.label.set_color('red')
ax.yaxis.label.set_color('green')
ax.zaxis.label.set_color('blue')
# Color of axes
ax.tick_params(axis='x', colors='red')
ax.tick_params(axis='y', colors='green')
ax.tick_params(axis='z', colors='blue')
# Set title
ax.set_title('RGB to pH')
# Show figure
plt.show()