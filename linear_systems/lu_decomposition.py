"""
lu_decomposition.py
--------------------------------
Numerical Methods Implementation:
LU Decomposition for Solving Linear Systems.

Algorithm Summary:
Decomposes A into lower (L) and upper (U) triangular matrices such that A = L·U,
then solves A·x = b by forward and backward substitution.

"""

import numpy as np

def lu_decomposition(A):
    """
    Perform LU decomposition using Doolittle’s method.

    Parameters
    ----------
    A : ndarray
        Square matrix (n x n)

    Returns
    -------
    L, U : tuple of ndarray
        Lower and Upper triangular matrices
    """
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for k in range(i, n):
            U[i, k] = A[i, k] - np.dot(L[i, :i], U[:i, k])
        for k in range(i + 1, n):
            L[k, i] = (A[k, i] - np.dot(L[k, :i], U[:i, i])) / U[i, i]
    return L, U


def lu_solve(L, U, b):
    """Solve A·x = b using LU factors."""
    # Forward substitution
    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    # Backward substitution
    x = np.zeros_like(b)
    for i in reversed(range(len(b))):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x


if __name__ == "__main__":
    A = np.array([[4, 3],
                  [6, 3]], dtype=float)
    b = np.array([10, 12], dtype=float)

    L, U = lu_decomposition(A)
    x = lu_solve(L, U, b)

    print("LU Decomposition Solution")
    print("--------------------------")
    print("L =\n", L)
    print("U =\n", U)
    print("x =", x)