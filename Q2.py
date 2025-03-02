import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
# Define the Lotka-Volterra equations
def lotka_volterra(t, z):
    x, y = z
    dxdt = -0.1*x + 0.02*x*y
    dydt = 0.2*y - 0.025*x*y
    return [dxdt, dydt]

# Initial conditions
x0, y0 = 6, 6
t_range = np.linspace(0, 100, 1000)

# Solve using solve_ivp
sol_lv = solve_ivp(lotka_volterra, [0, 100], [x0, y0], t_eval=t_range)

# Create a DataFrame for numerical values
df_lv = pd.DataFrame({
    "t": t_range,
    "Predator (x)": sol_lv.y[0],
    "Prey (y)": sol_lv.y[1]
})

print("\nNumerical Solutions for Lotka-Volterra Model:")
print(df_lv)

# Plot results
plt.plot(sol_lv.t, sol_lv.y[0], label="Predator Population (x)")
plt.plot(sol_lv.t, sol_lv.y[1], label="Prey Population (y)")
plt.legend()
plt.xlabel("Time (t)")
plt.ylabel("Population")
plt.title("Lotka-Volterra Predator-Prey Model")
plt.show()
