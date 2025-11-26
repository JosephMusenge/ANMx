import numpy as np

def least_squares_linear(x_data: np.ndarray, y_data: np.ndarray):
    """
    Fits a straight line (y = a0 + a1*x) to data using least-squares regression.
    Implemented from scratch using standard normal equations.

    Args:
        x_data: Numpy array of x coordinates.
        y_data: Numpy array of y coordinates.

    Returns:
        (a0, a1, r2): Intercept, Slope, and Coefficient of Determination (R-squared).
    """
    n = len(x_data)
    
    # Calculate sums required for normal equations
    sum_x = np.sum(x_data)
    sum_y = np.sum(y_data)
    sum_xy = np.sum(x_data * y_data)
    sum_x2 = np.sum(x_data**2)
    
    # Calculate means
    x_mean = sum_x / n
    y_mean = sum_y / n
    
    # Calculate slope (a1) and intercept (a0) using standard formulas
    # (Chapra & Clough, Chapter on Least Squares)
    denominator = n * sum_x2 - sum_x**2
    if denominator == 0:
        raise ValueError("Cannot fit line: all x-values are the same.")
        
    a1 = (n * sum_xy - sum_x * sum_y) / denominator
    a0 = y_mean - a1 * x_mean
    
    # --- Calculate R-squared ---
    # Total sum of squares (variance of data)
    st = np.sum((y_data - y_mean)**2)
    # Residual sum of squares (variance of error)
    sr = np.sum((y_data - (a0 + a1*x_data))**2)
    r2 = (st - sr) / st
    
    return a0, a1, r2

# Example usage for testing
if __name__ == "__main__":
    x_test = np.array([1, 2, 3, 4, 5, 6, 7])
    # Data with a slight trend and noise
    y_test = np.array([0.5, 2.5, 2.0, 4.0, 3.5, 6.0, 5.5])
    
    intercept, slope, r_sq = least_squares_linear(x_test, y_test)
    print(f"Model: y = {intercept:.4f} + {slope:.4f}x")
    print(f"R-squared: {r_sq:.4f}")