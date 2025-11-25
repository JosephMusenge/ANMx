import numpy as np

def parabolic_interpolation(func, x1, x2, x3, tol=1e-5, max_iter=50):
    """
    Finds the minimum using Successive Parabolic Interpolation.
    Requires three initial points: x1 < x2 < x3.
    """
    f1, f2, f3 = func(x1), func(x2), func(x3)
    
    for i in range(max_iter):
        # Calculate the vertex of the parabola (x_opt)
        # Using the formula from Chapra & Clough
        numerator = (x2 - x1)**2 * (f2 - f3) - (x2 - x3)**2 * (f2 - f1)
        denominator = (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)
        
        if denominator == 0:
            print("Error: Division by zero (collinear points).")
            return x2, f2, i
            
        x4 = x2 - 0.5 * (numerator / denominator)
        f4 = func(x4)
        
        # Check convergence
        if abs(x4 - x2) < tol:
            return x4, f4, i + 1
            
        # Update points (simple strategy: replace the point with highest function value)
        # In a robust solver, we would bracket the minimum carefully.
        # This is a simplified update for demonstration:
        if x4 > x2:
            x1, x2 = x2, x4
            f1, f2 = f2, f4
        else:
            x3, x2 = x2, x4
            f3, f2 = f2, f4
            
    print("Warning: Maximum iterations reached.")
    return x2, f2, max_iter