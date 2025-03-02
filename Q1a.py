
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

# Define the first ODE
def ode1(t,y):
    return t * np.exp(3*t) - 2*y

# Exact solution
def exact_solution1(t):
    return (1/5)*t*np.exp(3*t) - (1/25)*np.exp(3*t) + (1/25)*np.exp(-2*t)

# Time range
t = np.linspace(0, 1, 10)  # Fewer points to print a table

# Solve using odeint
y0 = 0
y_odeint = odeint(ode1, y0, t,tfirst=True).flatten()

# Solve using solve_ivp
sol = solve_ivp(ode1, [0, 1], [y0], t_eval=t)

# Compute exact solution
y_exact = exact_solution1(t)

# Create a DataFrame for comparison
df1 = pd.DataFrame({
    "t": t,
    "Exact": y_exact,
    "odeint": y_odeint,
    "solve_ivp": sol.y[0]
})

print("\nNumerical Solutions for First IVP:")
print(df1)

# Plot results
plt.plot(t, y_exact, label="Exact Solution", linestyle="dashed")
plt.plot(t, y_odeint, 'o-', label="odeint Solution")
plt.plot(sol.t, sol.y[0], 's-', label="solve_ivp Solution")
plt.legend()
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Comparison of Exact and Numerical Solutions")
plt.show()
