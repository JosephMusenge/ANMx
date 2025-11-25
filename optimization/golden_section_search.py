import math

def golden_section_search(func, xl, xu, tol=1e-5, max_iter=50):
    """
    Finds the minimum of a function using Golden Section Search.
    
    Args:
        func: The objective function to minimize.
        xl: Lower bound of the bracket.
        xu: Upper bound of the bracket.
        tol: Tolerance for stopping criterion.
        max_iter: Maximum number of iterations.
        
    Returns:
        (x_opt, f_opt, iterations): The optimal x, minimum value, and iter count.
    """
    # The Golden Ratio
    phi = (math.sqrt(5) - 1) / 2
    
    # Initial interior points
    d = phi * (xu - xl)
    x1 = xl + d
    x2 = xu - d
    
    f1 = func(x1)
    f2 = func(x2)
    
    for i in range(max_iter):
        # Check if interval is small enough
        if (xu - xl) < tol:
            xmin = (xu + xl) / 2
            return xmin, func(xmin), i + 1
        
        if f1 < f2:
            # The minimum is to the right of x2
            xl = x2
            x2 = x1
            f2 = f1
            # New x1
            d = phi * (xu - xl)
            x1 = xl + d
            f1 = func(x1)
        else:
            # The minimum is to the left of x1
            xu = x1
            x1 = x2
            f1 = f2
            # New x2
            d = phi * (xu - xl)
            x2 = xu - d
            f2 = func(x2)
            
    print("Warning: Maximum iterations reached.")
    return (xu + xl) / 2, func((xu + xl) / 2), max_iter