import numpy as np
import pandas as pd
import usefulfunctions as uf

seed = 42
rng = np.random.default_rng(seed)

# Generate scenarios
scenarios = uf.generate_scenarios(random_state=rng)


#%% ----------------------
# Task 1.1) Offering Strategy Under a One-Price Balancing Scheme
# ------------------------

# Pick 100 random scenarios
idx = rng.choice(scenarios.shape[1], size=100, replace=False)
in_sample_scenarios = scenarios[:, idx, :]

# Solve optimization problem
m, p_DA, Delta = uf.solve_stochastic_strategy_one_price(in_sample_scenarios)

# Print values
p_DA_values = np.array([v.X for v in p_DA.values()])
print("Expected Profit (One-Price):", round(m.ObjVal, 3), "MDKK")
print("Day-Ahead offers:", p_DA_values)


#%% ----------------------
# Task 1.2) Offering Strategy Under a Two-Price Balancing Scheme
# ------------------------

# Use same 100 random scenarios as in Task 1.1
# Solve optimization problem
m_2, p_DA_2, Delta_up, Delta_down = uf.solve_stochastic_strategy_two_price(in_sample_scenarios)

# Print values
p_DA_2_values = np.array([v.X for v in p_DA_2.values()])
print("Expected Profit (Two-Price):", round(m_2.ObjVal,3), "MDKK")
print("Day-Ahead offers:", p_DA_2_values)
