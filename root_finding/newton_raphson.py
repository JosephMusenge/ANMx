"""
newton_raphson.py
--------------------------------
Numerical Methods Implementation:
Newton-Raphson Method for Root Finding.

Algorithm Summary:
The Newton-Raphson Method uses tangents to approximate the root:
    x_(i+1) = x_i - f(x_i) / f'(x_i)
This method converges quadratically near the root but may diverge
if the initial guess is poor or f'(x) ≈ 0.

"""

import numpy as np

def f(x):
    """Example function: f(x) = exp(-x) - x."""
    return np.exp(-x) - x

def df(x):
    """Derivative of f(x): f'(x) = -exp(-x) - 1."""
    return -np.exp(-x) - 1


def newton_raphson(func, dfunc, x0, es=1e-7, maxit=50):
    """
    Finds the root of func(x) = 0 using the Newton-Raphson Method.

    Parameters
    ----------
    func : callable
        Function whose root is sought.
    dfunc : callable
        Derivative of func(x).
    x0 : float
        Initial guess.
    es : float, optional
        Stopping criterion for relative error (default: 1e-7).
    maxit : int, optional
        Maximum number of iterations (default: 50).

    Returns
    -------
    (x, f(x), ea, iter) : tuple
        x : estimated root
        f(x) : function value at root
        ea : approximate relative error
        iter : iterations performed
    """
    for i in range(maxit):
        x1 = x0 - func(x0) / dfunc(x0)
        ea = abs((x1 - x0) / x1)
        if ea < es:
            return x1, func(x1), ea, i + 1
        x0 = x1
    return x1, func(x1), ea, i + 1


if __name__ == "__main__":
    root, froot, ea, it = newton_raphson(f, df, 0.5)
    print("Newton-Raphson Method Example")
    print("------------------------------")
    print(f"Root estimate: {root:.6f}")
    print(f"f(root): {froot:.6e}")
    print(f"Relative error: {ea:.2e}")
    print(f"Iterations: {it}")