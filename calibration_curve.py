import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
 
# Fitting function
def func(rgb, a1, a2, b1, b2, c1, c2):
    return a1 * rgb[:, 0]**a2 + b1 * rgb[:, 1]**b2 + c1 * rgb[:, 2]**c2
 
# Experimental data points
with open('ph test data.csv') as f:
    reader = csv.reader(f)
    rows = [row for row in reader]
pH_values = np.array(rows[0])
rgb_values = np.array([list(t) for t in list(zip([float(i) for i in rows[2]], [float(i) for i in rows[4]], [float(i) for i in rows[6]]))])
r = rgb_values[:, 0]
g = rgb_values[:, 1]
b = rgb_values[:, 2]

# Perform the curve-fit
popt, pcov = curve_fit(func, rgb_values, pH_values, maxfev = 20000)
a1, a2, b1, b2, c1, c2 = popt
print("Fitting function:\npH = " + str(a1) + " x R^(" + str(a2) + ") + " + str(b1) + " x G^(" + str(b2) + ") + " + str(c1) + " x B^(" + str(c2) + ")")

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
colorbar.set_label('pH')

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
# Show figure
plt.show()

# Save result
with open('calibration curve.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(popt)