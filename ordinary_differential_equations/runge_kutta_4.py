import numpy as np

def rk4_method(dydt, y0, t_span, h):
    """
    Solves ODEs using the 4th-Order Runge-Kutta Method.
    This is much more accurate than Euler for the same step size.
    """
    t_start, t_end = t_span
    t_values = np.arange(t_start, t_end + h, h)
    
    if np.isscalar(y0):
        y_values = np.zeros(len(t_values))
    else:
        y_values = np.zeros((len(t_values), len(y0)))
        
    y_values[0] = y0
    
    for i in range(1, len(t_values)):
        t = t_values[i-1]
        y = y_values[i-1]
        
        # Calculate the four slopes (k1, k2, k3, k4)
        k1 = dydt(t, y)
        k2 = dydt(t + 0.5*h, y + 0.5*k1*h)
        k3 = dydt(t + 0.5*h, y + 0.5*k2*h)
        k4 = dydt(t + h, y + k3*h)
        
        # Weighted average slope
        slope_avg = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        y_values[i] = y + slope_avg * h
        
    return t_values, y_values