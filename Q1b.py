import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
# Define the second ODE
def ode2(t, y):
    return 1 + (t - y)**2

# Exact solution
def exact_solution2(t):
    return t + 1 / (1 - t)

# Time range
t2 = np.linspace(2, 3, 10)

# Solve using solve_ivp
y2_0 = [1]
sol2 = solve_ivp(ode2, [2, 3], y2_0, t_eval=t2)

# Compute exact solution
y_exact2 = exact_solution2(t2)

# Create a DataFrame for comparison
df2 = pd.DataFrame({
    "t": t2,
    "Exact": y_exact2,
    "solve_ivp": sol2.y[0]
})

print("\nNumerical Solutions for Second IVP:")
print(df2)

# Plot results
plt.plot(t2, y_exact2, label="Exact Solution", linestyle="dashed")
plt.plot(sol2.t, sol2.y[0], 's-', label="solve_ivp Solution")
plt.legend()
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Comparison of Exact and Numerical Solutions (Second IVP)")
plt.show()
