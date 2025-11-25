import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from golden_section_search import golden_section_search
from parabolic_interpolation import parabolic_interpolation

# define a cost function 
# (Example: designing a container to minimize material for a specific volume)
def cost_function(x):
    """
    Example function: f(x) = (x^2)/10 - 2*sin(x)
    This has a global minimum around x = 1.42
    """
    return (x**2)/10 - 2*np.sin(x)

# setup parameters
x_start, x_mid, x_end = 0, 1, 4 # initial bracket

# run implementations
# --- Golden Section ---
gs_x, gs_val, gs_iter = golden_section_search(cost_function, x_start, x_end)

# --- Parabolic Interpolation ---
pi_x, pi_val, pi_iter = parabolic_interpolation(cost_function, x_start, x_mid, x_end)

# --- SciPy implementation (The "Gold Standard") ---
# We use 'minimize_scalar' which automatically selects the best method (usually Brent's)
sp_result = optimize.minimize_scalar(cost_function, bounds=(x_start, x_end), method='bounded')

# the comparison table
print(f"{'METHOD':<25} | {'MINIMUM X':<12} | {'ITERATIONS':<10} | {'ERROR vs SCIPY'}")
print("-" * 70)

print(f"{'Golden Section':<25} | {gs_x:.8f}   | {gs_iter:<10} | {abs(gs_x - sp_result.x):.2e}")
print(f"{'Parabolic Interpolation':<25} | {pi_x:.8f}   | {pi_iter:<10} | {abs(pi_x - sp_result.x):.2e}")
print(f"{'SciPy (Brent)':<25} | {sp_result.x:.8f}   | {sp_result.nfev:<10} | 0.00 (Baseline)")

# visualization confirmation
x_vals = np.linspace(0, 4, 100)
y_vals = cost_function(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, 'k-', label='Cost Function')
plt.plot(gs_x, gs_val, 'bo', label='Golden Section Found')
plt.plot(sp_result.x, sp_result.fun, 'rx', markersize=12, label='SciPy Found')
plt.title('Optimization Methods Comparison')
plt.legend()
plt.grid(True)
plt.savefig('optimization_comparison.png')
plt.show()