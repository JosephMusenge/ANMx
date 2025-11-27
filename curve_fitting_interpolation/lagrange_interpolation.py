import numpy as np

# Implementation of the Lagrange polynomial formula
def lagrange_interpolate(x_points, y_points, x_query):
    """
    Estimates y values at x_query locations using a Lagrange polynomial 
    that passes exactly through all (x_points, y_points).

    Args:
        x_points: The known data points (knots).
        y_points: The values at the known data points.
        x_query: A single value or array of values where we want to estimate y.
    """
    x_query = np.atleast_1d(x_query)
    n = len(x_points)
    y_estimated = np.zeros_like(x_query, dtype=float)
    
    for k, xq in enumerate(x_query):
        # Evaluate the Lagrange polynomial at this specific query point
        total_sum = 0
        for i in range(n):
            # Calculate the Lagrange basis polynomial L_i(x)
            # L_i is the product term
            product_term = y_points[i]
            for j in range(n):
                if i != j:
                    product_term = product_term * (xq - x_points[j]) / (x_points[i] - x_points[j])
            total_sum += product_term
        y_estimated[k] = total_sum
        
    if len(y_estimated) == 1:
        return y_estimated[0]
    return y_estimated