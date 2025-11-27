import numpy as np
from scipy.interpolate import CubicSpline

def generate_cubic_spline(x_points, y_points):
    """
    Generates a cubic spline function from data points using SciPy.
    
    In engineering, splines are preferred over high-order polynomials because
    they avoid oscillations (Runge's phenomenon) by using piecewise 
    low-order polynomials connected smoothly.

    Returns:
        A callable function cs(x) that calculates interpolated values.
    """
    # bc_type='natural' means the second derivative is zero at the endpoints.
    # This is a common engineering assumption.
    cs = CubicSpline(x_points, y_points, bc_type='natural')
    return cs