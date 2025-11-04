"""
gauss_seidel.py
--------------------------------
Numerical Methods Implementation:
Gauss-Seidel Iterative Method for Solving Linear Systems.

Algorithm Summary:
Iteratively refines an initial guess for x in A·x = b using:
    x_i^(k+1) = (b_i - Σ_{j≠i} a_ij * x_j^(k+1 or k)) / a_ii
until convergence within a specified tolerance.

"""

import numpy as np

def gauss_seidel(A, b, x0=None, tol=1e-6, maxit=100):
    """
    Solve A·x = b using the Gauss-Seidel iterative method.

    Parameters
    ----------
    A : ndarray
        Coefficient matrix (n x n)
    b : ndarray
        Right-hand side vector (n,)
    x0 : ndarray, optional
        Initial guess (default: zero vector)
    tol : float
        Convergence tolerance
    maxit : int
        Maximum number of iterations

    Returns
    -------
    x : ndarray
        Approximate solution vector
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)

    x = x0.copy()
    for iteration in range(maxit):
        x_old = x.copy()
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - sigma) / A[i, i]

        if np.linalg.norm(x - x_old, np.inf) < tol:
            break
    return x


if __name__ == "__main__":
    A = np.array([[4, -1, 0, 0],
                  [-1, 4, -1, 0],
                  [0, -1, 4, -1],
                  [0, 0, -1, 3]], dtype=float)
    b = np.array([15, 10, 10, 10], dtype=float)

    sol = gauss_seidel(A, b)
    print("Gauss-Seidel Solution")
    print("----------------------")
    print(f"x = {sol}")