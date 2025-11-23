import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def f(x):
    """
    The function whose root we want to find.
    Example: f(x) = x^3 - x - 1
    """
    return x**3 - x - 1

def df(x):
    """
    The derivative of f(x), required for Newton-Raphson.
    f'(x) = 3x^2 - 1
    """
    return 3*x**2 - 1

def get_true_error(current_est, true_val):
    """Calculates absolute true error."""
    return abs(true_val - current_est)

def run_bisection(func, xl, xu, true_root, max_iter=20):
    """Runs Bisection and returns a list of errors at each iteration."""
    errors = []
    
    if func(xl) * func(xu) >= 0:
        print("Bisection error: Root not bracketed.")
        return []

    for i in range(max_iter):
        xr = (xl + xu) / 2
        errors.append(get_true_error(xr, true_root))
        
        if func(xl) * func(xr) < 0:
            xu = xr
        else:
            xl = xr
            
        # Stop if error is extremely small (optional)
        if errors[-1] < 1e-15:
            break
            
    return errors

def run_newton(func, d_func, x0, true_root, max_iter=20):
    """Runs Newton-Raphson and returns a list of errors at each iteration."""
    errors = []
    xr = x0
    
    for i in range(max_iter):
        errors.append(get_true_error(xr, true_root))
        
        # Newton-Raphson formula: x_new = x_old - f(x)/f'(x)
        try:
            xr = xr - func(xr) / d_func(xr)
        except ZeroDivisionError:
            print("Newton error: Derivative is zero.")
            break
            
        if errors[-1] < 1e-15:
            break
            
    return errors

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Setup the Problem
    # We use scipy to get the "ground truth" root for comparison
    true_root = optimize.fsolve(f, 1.0)[0] 
    print(f"True Root: {true_root:.8f}")

    # 2. Run the Methods
    # Bisection setup: Bracket [1, 2]
    bisection_errors = run_bisection(f, 1, 2, true_root)
    
    # Newton setup: Initial guess 1.0
    newton_errors = run_newton(f, df, 1.0, true_root)

    # 3. Plot the Comparison
    plt.figure(figsize=(10, 6))
    
    # Plot Bisection
    plt.semilogy(range(len(bisection_errors)), bisection_errors, 
                 label='Bisection Method', marker='o', linestyle='--', color='blue')
    
    # Plot Newton-Raphson
    plt.semilogy(range(len(newton_errors)), newton_errors, 
                 label='Newton-Raphson', marker='s', linestyle='-', color='red')

    # Formatting
    plt.title('Convergence Speed: Bisection vs. Newton-Raphson', fontsize=14)
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Absolute True Error (Log Scale)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # Add text annotation to explain the graph
    plt.text(2, 1e-5, "Newton drops quadratically\n(very fast)", color='red')
    plt.text(5, 1e-2, "Bisection drops linearly\n(slow & steady)", color='blue')

    # Save or Show
    plt.tight_layout()
    plt.savefig('convergence_comparison.png') # Saves the plot for your README
    plt.show()