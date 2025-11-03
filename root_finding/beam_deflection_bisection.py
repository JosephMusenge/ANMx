"""
beam_deflection_bisection.py
--------------------------------
Numerical Methods Application:
Find the point of maximum deflection of a pinned–fixed beam
under uniform load using the Bisection Root-Finding Method.

Equation for deflection:
    y = (-w / (48 * E * I)) * (2*x**4 - 3*L*x**3 + L**3*x)

Objective:
Find x where dy/dx = 0 (maximum deflection location).

Author: Joseph Musenge
Course: Applied Numerical Methods with Python
Date: November 2025
"""

import numpy as np


# -----------------------------
# 1. Beam Parameters (consistent units)
# -----------------------------
L = 400.0      # cm
E = 52000.0    # kN/cm²
I = 32000.0    # cm⁴
w = 4.0        # kN/cm


# -----------------------------
# 2. Define Mathematical Models
# -----------------------------
def deflection(x):
    """Deflection equation y(x) = (-w / (48EI)) * (2x⁴ - 3Lx³ + L³x)."""
    return (-w / (48 * E * I)) * (2 * x**4 - 3 * L * x**3 + L**3 * x)


def dydx(x):
    """First derivative of deflection (slope). dy/dx = 0 at max deflection."""
    return (-w / (48 * E * I)) * (8 * x**3 - 9 * L * x**2 + L**3)


# -----------------------------
# 3. Generic Bisection Method
# -----------------------------
def bisection(func, xl, xu, es=1e-7, maxit=50):
    """
    Estimate a root of func(x) = 0 using the Bisection Method.
    Parameters:
        func  : callable function
        xl, xu: lower and upper guesses (must bracket the root)
        es    : relative error tolerance
        maxit : maximum iterations
    Returns:
        (root, f(root), ea, iter)
    """
    if func(xl) * func(xu) > 0:
        raise ValueError("Root not bracketed. Choose different xl, xu.")

    xr_old = xl
    for i in range(maxit):
        xr = (xl + xu) / 2.0
        ea = abs((xr - xr_old) / xr)
        if ea < es:
            return xr, func(xr), ea, i + 1
        if func(xl) * func(xr) < 0:
            xu = xr
        else:
            xl = xr
        xr_old = xr
    return xr, func(xr), ea, i + 1


# -----------------------------
# 4. Main Computation
# -----------------------------
if __name__ == "__main__":
    # Find x where dy/dx = 0 using bisection between 0 and 0.9L
    x_lower = 0.0
    x_upper = 0.9 * L

    x_max, _, ea, n = bisection(dydx, x_lower, x_upper)

    y_max = deflection(x_max)

    # -----------------------------
    # 5. Display Results
    # -----------------------------
    print("Bisection Method for Beam Deflection")
    print("-------------------------------------")
    print(f"Maximum deflection location (x_max): {x_max:10.6f} cm")
    print(f"Maximum deflection value (y_max):    {y_max:10.6f} cm")
    print(f"Approx. relative error:             {ea:.2e}")
    print(f"Iterations:                         {n:d}")