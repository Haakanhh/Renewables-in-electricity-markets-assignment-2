#%%
import importlib
import numpy as np
import pandas as pd
import usefulfunctions as uf
import matplotlib.pyplot as plt

importlib.reload(uf)

seed = 40
rng = np.random.default_rng(seed)

# Generate scenarios
scenarios = uf.generate_scenarios(random_state=rng, n_wind=20, n_price=20, n_surp_def=4)

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
print("Expected Profit (One-Price):", round(m.ObjVal, 3), "MDKK")
print("Day-Ahead offers:", p_DA)

# Plot day-ahead offers
uf.plot_DA_offers(p_DA, in_sample_scenarios, title="Day-ahead offers - One-Price", Threshold_value=0.375)

# Plot profit distribution per scenario
profit_per_hour = profit_matrix.mean(axis=1)
profit_per_scenario = profit_matrix.sum(axis=0)
uf.plot_profit_distribution(profit_per_scenario, n_bins = 30, title="Profit distribution per scenario in one-price system")
uf.plot_cumulative_profit_distribution(profit_per_scenario, title="Cumulative profit distribution - One-price")


#%% ----------------------
# Task 1.2) Offering Strategy Under a Two-Price Balancing Scheme
# ------------------------

# Use same 100 random scenarios as in Task 1.1
# Solve optimization problem
m_2, p_DA_2, Delta_up, Delta_down, profit_matrix_2 = uf.solve_stochastic_strategy_two_price(in_sample_scenarios)

# Print expected profit and day-ahead offers
print("Expected Profit (Two-Price):", round(m_2.ObjVal,3), "MDKK")
print("Day-Ahead offers:", p_DA_2)

# Plot day-ahead offers
uf.plot_DA_offers(p_DA_2, in_sample_scenarios, title="Day-ahead offers - Two-Price")

# Plot profit distribution per scenario
profit_per_hour_2 = profit_matrix_2.mean(axis=1)
profit_per_scenario_2 = profit_matrix_2.sum(axis=0)
uf.plot_profit_distribution(profit_per_scenario_2, n_bins = 30, title="Profit distribution per scenario in two-price system")
uf.plot_cumulative_profit_distribution(profit_per_scenario_2, title="Cumulative profit distribution - Two-price")

# Profit comparison plot
uf.plot_profit_distribution_comparison(profit_per_scenario, profit_per_scenario_2, n_bins=30)



#%% -----------------------
# Task 1.3) Cross-Validation of Offering Strategies
# -------------------------

folds = uf.create_folds(scenarios, n_in_sample=200, seed=rng) # Creates a list of arrays, each of shape (24, 200, 3)

# One-price cross-validation
in_sample_means_one, out_sample_means_one = uf.cross_validate_folds(folds, two_price=False)

print(f"One-Price mean in-sample profit:  {np.mean(in_sample_means_one):.15f} MDKK")
print(f"One-Price mean out-of-sample profit: {np.mean(out_sample_means_one):.15f} MDKK")

print(f"Std in-sample profit:   {np.std(in_sample_means_one):.3f}")
print(f"Std out-of-sample profit:{np.std(out_sample_means_one):.3f}")

uf.plot_Cross_Validation_Profits(in_sample_means_one, out_sample_means_one, title="Cross-Validation mean profits - One-Price")

# Two_Price cross-validation
in_sample_means_two, out_sample_means_two = uf.cross_validate_folds(folds, two_price=True)

print(f"Two-Price mean in-sample profit:  {np.mean(in_sample_means_two):.15f} MDKK")
print(f"Two-Price mean out-of-sample profit: {np.mean(out_sample_means_two):.15f} MDKK")

print(f"Std in-sample profit:   {np.std(in_sample_means_two):.3f}")
print(f"Std out-of-sample profit:{np.std(out_sample_means_two):.3f}")

uf.plot_Cross_Validation_Profits(in_sample_means_two, out_sample_means_two, title="Cross-Validation mean profits - Two-Price")




























#%% ----------------------------------------
# EXTRA DATA PRINT AND EXTRA PLOTS
# ------------------------------------------

# Calculate deficit probability across scenarios per hour
prob_deficit = in_sample_scenarios[:, :, 2].mean(axis=1)
df_prob = pd.DataFrame({
    "hour": np.arange(24),
    "P(deficit)": prob_deficit,
    "P(surplus)": 1 - prob_deficit,
    "DA offers one price": p_DA,
    "DA offers two price": p_DA_2
})

print(df_prob)

#%% -------------
# PLOT OF SAMPLES
# ---------------

imbalance_scenarios = np.array([
    scenarios[:, sd_i, 2]  # w_i = 0, p_i = 0
    for sd_i in range(4)
])

price_scenarios = np.array([
    scenarios[:, p_i * 4, 1]   # w_i = 0, sd_i = 0
    for p_i in range(20)
])

wind_scenarios = np.array([
    scenarios[:, w_i * 80, 0]   # p_i = 0, sd_i = 0
    for w_i in range(20)
])

n_w = 20
n_p = 20
n_sd = 4

hours = np.arange(24)

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# ----------------------
# WIND (20 distinct)
# ----------------------
for i in range(20):
    axes[0].plot(hours, wind_scenarios[i], alpha=0.7)

axes[0].set_title("20 Wind Scenarios (Distinct)")
axes[0].set_ylabel("Wind Production")
axes[0].grid(True)

# ----------------------
# PRICE (20 distinct)
# ----------------------
for i in range(20):
    axes[1].plot(hours, price_scenarios[i], alpha=0.7)

axes[1].set_title("20 Price Scenarios (Distinct)")
axes[1].set_ylabel("Price")
axes[1].grid(True)

# ----------------------
# IMBALANCE HEATMAP (4 distinct)
# ----------------------
heatmap_data = np.where(imbalance_scenarios == 1, 1, -1)

im = axes[2].imshow(
    heatmap_data,
    aspect='auto',
    cmap='bwr',
    vmin=-1,
    vmax=1
)

axes[2].set_title("4 Imbalance Scenarios")
axes[2].set_ylabel("Scenario")
axes[2].set_xlabel("Hour")

axes[2].set_yticks(np.arange(4))
axes[2].set_yticklabels([f"SD {i}" for i in range(4)])

# grid for clarity
axes[2].set_xticks(np.arange(-.5, 24, 1), minor=True)
axes[2].set_yticks(np.arange(-.5, 4, 1), minor=True)
axes[2].grid(which='minor', color='black', linestyle='-', linewidth=0.5)
axes[2].tick_params(which='minor', bottom=False, left=False)

# colorbar
cbar = fig.colorbar(im, ax=axes[2])
cbar.set_ticks([-1, 1])
cbar.set_ticklabels(["Surplus", "Deficit"])

plt.tight_layout()
plt.show()


# %%
