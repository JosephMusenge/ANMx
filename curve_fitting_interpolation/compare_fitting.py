import numpy as np
import matplotlib.pyplot as plt
from lagrange_interpolation import lagrange_interpolate
from splines import generate_cubic_spline
# import linear regression to show difference between fitting and interpolating
from linear_regression import least_squares_linear

# Generate Synthetic "Noisy Sensor Data" 
np.random.seed(42) # for reproducible results

# true underlying function 
def true_function(x):
    return np.sin(x) + 0.5 * np.cos(3*x)

# Create sparse "knot" points 
x_knots = np.linspace(0, 6, 11)
# noise to represent sensor inaccuracy
noise = np.random.normal(0, 0.2, len(x_knots))
y_knots = true_function(x_knots) + noise

# for plotting smooth curves
x_dense = np.linspace(0, 6, 400)


# Apply the Methods
# A) Linear Regression (Trend fitting)
a0, a1, _ = least_squares_linear(x_knots, y_knots)
y_linear = a0 + a1 * x_dense

# B) High-Order Polynomial Interpolation (Lagrange)
# Since we have 11 points, this creates a 10th-degree polynomial
# This will cause Runge's Phenomenon (wild oscillations near edges)
y_lagrange = lagrange_interpolate(x_knots, y_knots, x_dense)

# C) Cubic Spline Interpolation
# Piecewise 3rd-degree polynomials designed to be stable.
spline_func = generate_cubic_spline(x_knots, y_knots)
y_spline = spline_func(x_dense)


# Visualize
plt.figure(figsize=(12, 7))

# Plot the data points
plt.plot(x_knots, y_knots, 'ko', markersize=8, label='Noisy Data Points (Knots)')

# Plot Linear Regression
plt.plot(x_dense, y_linear, 'g--', linewidth=2, label='Linear Regression (Trend)')

# Plot Spline
plt.plot(x_dense, y_spline, 'b-', linewidth=2.5, alpha=0.8, label='Cubic Spline (Stable Interpolation)')

# Plot Lagrange Polynomial
plt.plot(x_dense, y_lagrange, 'r:', linewidth=2, label='10th-Degree Poly (Runge\'s Phenomenon)')


# Formatting and Annotations
plt.title('The Danger of High-Order Polynomials vs. Stability of Splines', fontsize=14)
plt.xlabel('Sensor Input (x)', fontsize=12)
plt.ylabel('Sensor Output (y)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-3, 3) # Limit y-axis to zoom in on the relevant area, ignoring extreme polynomial swings

# annotation pointing out the oscillations
plt.annotate('Runge\'s Phenomenon:\nWild oscillations between points!', 
             xy=(0.5, y_lagrange[30]), 
             xytext=(1, -2.5),
             arrowprops=dict(facecolor='red', shrink=0.05),
             color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('curve_fitting_comparison.png')
plt.show()