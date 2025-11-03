"""
bisection_method.py
--------------------------------
Numerical Methods Implementation:
Bisection Method for Root Finding.

Algorithm Summary:
The Bisection Method locates a root of f(x) = 0 by repeatedly bisecting
an interval [xl, xu] and selecting the subinterval where the sign of f(x)
changes. Convergence is guaranteed if f(x) is continuous and the root is bracketed.

"""

import numpy as np

def f(x):
    # Example function: f(x) = exp(-x) - x."""
    return np.exp(-x) - x


def bisection(func, xl, xu, es=1e-7, maxit=50):
    """
    Finds the root of func(x) = 0 using the Bisection Method.

    Parameters
    ----------
    func : callable
        Function for which the root is sought.
    xl, xu : float
        Lower and upper guesses that bracket the root.
    es : float, optional
        Stopping criterion for relative error (default: 1e-7).
    maxit : int, optional
        Maximum number of iterations (default: 50).

    Returns
    -------
    (xr, f(xr), ea, iter) : tuple
        xr : estimated root
        f(xr) : function value at root
        ea : approximate relative error
        iter : iterations performed
    """
    if func(xl) * func(xu) > 0:
        raise ValueError("Error: Root not bracketed. Choose different xl and xu.")

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


if __name__ == "__main__":
    root, froot, ea, it = bisection(f, 0, 1)
    print("Bisection Method Example")
    print("-------------------------")
    print(f"Root estimate: {root:.6f}")
    print(f"f(root): {froot:.6e}")
    print(f"Relative error: {ea:.2e}")
    print(f"Iterations: {it}")