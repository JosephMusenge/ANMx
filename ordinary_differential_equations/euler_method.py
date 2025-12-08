import numpy as np

def euler_method(dydt, y0, t_span, h):
    """
    Solves ODEs using Euler's Method: y_{i+1} = y_i + f(t_i, y_i) * h
    
    Args:
        dydt: The derivative function f(t, y). Returns dy/dt.
              y can be a scalar or a numpy array (for systems of ODEs).
        y0: Initial condition(s).
        t_span: Tuple (t_start, t_end).
        h: Step size.
        
    Returns:
        t_values, y_values (numpy arrays)
    """
    t_start, t_end = t_span
    
    # Create time array
    t_values = np.arange(t_start, t_end + h, h)
    
    # Initialize solution array
    # Check if y0 is a scalar or a vector
    if np.isscalar(y0):
        y_values = np.zeros(len(t_values))
    else:
        y_values = np.zeros((len(t_values), len(y0)))
        
    y_values[0] = y0
    
    for i in range(1, len(t_values)):
        # Apply Euler's formula
        # y_new = y_old + slope * step
        y_values[i] = y_values[i-1] + dydt(t_values[i-1], y_values[i-1]) * h
        
    return t_values, y_values