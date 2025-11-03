"""
gauss_elimination.py
--------------------------------
Numerical Methods Implementation:
Gaussian Elimination with Partial Pivoting.

Algorithm Summary:
Solves a system of linear equations A·x = b by performing
forward elimination to transform A into upper triangular form,
followed by back substitution to solve for x.

"""

import numpy as np

def gauss_elimination(A, b):
    """
    Solve the system A·x = b using Gaussian Elimination with Partial Pivoting.

    Parameters
    ----------
    A : ndarray
        Coefficient matrix (n x n)
    b : ndarray
        Right-hand side vector (n,)

    Returns
    -------
    x : ndarray
        Solution vector
    """
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)

    # Forward elimination
    for k in range(n - 1):
        # Partial pivoting
        pivot = np.argmax(np.abs(A[k:, k])) + k
        if pivot != k:
            A[[k, pivot]] = A[[pivot, k]]
            b[[k, pivot]] = b[[pivot, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x


if __name__ == "__main__":
    A = np.array([[3, -0.1, -0.2],
                  [0.1, 7, -0.3],
                  [0.3, -0.2, 10]])
    b = np.array([7.85, -19.3, 71.4])
    x = gauss_elimination(A, b)

    print("Gaussian Elimination Solution")
    print("------------------------------")
    print(f"x = {x}")