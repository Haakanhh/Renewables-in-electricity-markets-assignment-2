import numpy as np
import pandas as pd
import usefulfunctions as uf
import matplotlib.pyplot as plt

seed = 429
rng = np.random.default_rng(seed)

# Generate scenarios
scenarios = uf.generate_scenarios(random_state=rng)

#%% ----------------------
# Task 1.1) Offering Strategy Under a One-Price Balancing Scheme
# ------------------------

# Pick 100 random scenarios
n_insample_scenarios = 200
idx = rng.choice(scenarios.shape[1], size=n_insample_scenarios, replace=False)
in_sample_scenarios = scenarios[:, idx, :]

# Solve optimization problem
m, p_DA, Delta, profit_matrix = uf.solve_stochastic_strategy_one_price(in_sample_scenarios)

# Print expected profit and day-ahead offers
p_DA_values = np.array([v.X for v in p_DA.values()])
print("Expected Profit (One-Price):", round(m.ObjVal, 3), "MDKK")
print("Day-Ahead offers:", p_DA_values)

# Plot profit distribution per scenario
profit_per_hour = profit_matrix.mean(axis=1)
profit_per_scenario = profit_matrix.sum(axis=0)
uf.plot_profit_distribution(profit_per_scenario, n_bins = 30, title="Profit distribution per scenario in one-price system")


#%% ----------------------
# Task 1.2) Offering Strategy Under a Two-Price Balancing Scheme
# ------------------------

# Use same 100 random scenarios as in Task 1.1
# Solve optimization problem
m_2, p_DA_2, Delta_up, Delta_down, profit_matrix_2 = uf.solve_stochastic_strategy_two_price(in_sample_scenarios)

# Print expected profit and day-ahead offers
p_DA_2_values = np.array([v.X for v in p_DA_2.values()])
print("Expected Profit (Two-Price):", round(m_2.ObjVal,3), "MDKK")
print("Day-Ahead offers:", p_DA_2_values)

# Plot profit distribution per scenario
profit_per_hour_2 = profit_matrix_2.mean(axis=1)
profit_per_scenario_2 = profit_matrix_2.sum(axis=0)
uf.plot_profit_distribution(profit_per_scenario_2, n_bins = 30, title="Profit distribution per scenario in two-price system")
# %%

# Create shared bin edges
all_profits = np.concatenate([profit_per_scenario, profit_per_scenario_2])
bins = np.linspace(all_profits.min(), all_profits.max(), 16)  # 12 bins → 13 edges

plt.figure(figsize=(10,6))

plt.hist(profit_per_scenario, bins=bins, alpha=0.5, label="One-price")
plt.hist(profit_per_scenario_2, bins=bins, alpha=0.5, label="Two-price")

plt.axvline(profit_per_scenario.mean(), color='blue', linestyle='dashed')
plt.axvline(profit_per_scenario_2.mean(), color='orange', linestyle='dashed')

plt.title("Profit distribution per scenario: One-price vs Two-price", fontsize=18)
plt.xlabel("Total profit (MDKK)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)

plt.legend()
plt.show()












#%% ----------------------------------------
# EXTRA DATA PRINT
# ------------------------------------------

# Calculate deficit probability across scenarios per hour
prob_deficit = in_sample_scenarios[:, :, 2].mean(axis=1)
df_prob = pd.DataFrame({
    "hour": np.arange(24),
    "P(deficit)": prob_deficit,
    "P(surplus)": 1 - prob_deficit,
    "DA offers one price": p_DA_values,
    "DA offers two price": p_DA_2_values
})

print(df_prob)
# %%
