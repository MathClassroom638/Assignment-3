import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
# Define the competition model equations
def competition_model(t, z):
    x, y = z
    dxdt = x * (2 - 0.4*x - 0.3*y)
    dydt = y * (1 - 0.1*y - 0.3*x)
    return [dxdt, dydt]

# Time range
t_range = np.linspace(0, 20, 10)

# Initial conditions for different cases
initial_conditions = [(1.5, 3.5), (1, 1), (2, 7), (4.5, 0.5)]

# Solve for each case and display values
for i, (x0, y0) in enumerate(initial_conditions):
    sol_comp = solve_ivp(competition_model, [0, 20], [x0, y0], t_eval=t_range)

    # Create DataFrame for numerical values
    df_comp = pd.DataFrame({
        "t": t_range,
        f"x (x0={x0}, y0={y0})": sol_comp.y[0],
        f"y (x0={x0}, y0={y0})": sol_comp.y[1]
    })
    
    print(f"\nNumerical Solutions for Competition Model (Case {i+1}):")
    print(df_comp)

# Plot results
plt.figure(figsize=(10, 6))

for x0, y0 in initial_conditions:
    sol_comp = solve_ivp(competition_model, [0, 20], [x0, y0], t_eval=t_range)
    plt.plot(sol_comp.t, sol_comp.y[0], label=f"x(t) for x0={x0}, y0={y0}")
    plt.plot(sol_comp.t, sol_comp.y[1], linestyle="dashed", label=f"y(t) for x0={x0}, y0={y0}")

plt.legend()
plt.xlabel("Time (t)")
plt.ylabel("Population")
plt.title("Competition Model Dynamics")
plt.show()
