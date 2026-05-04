#%%
import importlib
import numpy as np
import pandas as pd
import usefulfunctions as uf
import matplotlib.pyplot as plt

importlib.reload(uf)

seed = 42
rng = np.random.default_rng(seed)

# Generate scenarios
scenarios = uf.generate_scenarios(random_state=rng, n_wind=20, n_price=20, n_surp_def=4)

#%% ----------------------
# Task 1.1) Offering Strategy Under a One-Price Balancing Scheme
# ------------------------

# Pick random scenarios
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

# Use same 200 random scenarios as in Task 1.1
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

#%% 
importlib.reload(uf)
m_2, p_DA_2, Delta_up, Delta_down, profit_matrix_2 = uf.solve_stochastic_strategy_two_price(in_sample_scenarios)
min_profit, max_profit = uf.scenario_profit_stats(profit_matrix_2)

print(f"Min profit: {min_profit:.2f} MDKK")
print(f"Max profit: {max_profit:.2f} MDKK")

#%% -----------------------
# Task 1.3) Cross-Validation of Offering Strategies
# -------------------------

folds = uf.create_folds(scenarios, n_in_sample=200, seed=rng) # Creates a list of arrays, each of shape (24, 200, 3)

# One-price cross-validation
in_sample_means_one, out_sample_means_one = uf.cross_validate_folds(folds, two_price=False, silent=True)

print(f"One-Price mean in-sample profit:  {np.mean(in_sample_means_one):.15f} MDKK")
print(f"One-Price mean out-of-sample profit: {np.mean(out_sample_means_one):.15f} MDKK")

print(f"Std in-sample profit:   {np.std(in_sample_means_one):.3f}")
print(f"Std out-of-sample profit:{np.std(out_sample_means_one):.3f}")

uf.plot_Cross_Validation_Profits(in_sample_means_one, out_sample_means_one, title="Cross-Validation mean profits - One-Price")

# Two_Price cross-validation
in_sample_means_two, out_sample_means_two = uf.cross_validate_folds(folds, two_price=True, silent=True)

print(f"Two-Price mean in-sample profit:  {np.mean(in_sample_means_two):.15f} MDKK")
print(f"Two-Price mean out-of-sample profit: {np.mean(out_sample_means_two):.15f} MDKK")

print(f"Std in-sample profit:   {np.std(in_sample_means_two):.3f}")
print(f"Std out-of-sample profit:{np.std(out_sample_means_two):.3f}")

uf.plot_Cross_Validation_Profits(in_sample_means_two, out_sample_means_two, title="Cross-Validation mean profits - Two-Price")


# Results with varying in-sample size
in_sample_list = [50, 100, 200, 400, 800]

for n_in in in_sample_list:
    folds_check = uf.create_folds(scenarios, n_in_sample=n_in, seed=rng)
    in_means_one, out_means_one = uf.cross_validate_folds(folds_check, two_price=False, silent=True)
    in_means_two, out_means_two = uf.cross_validate_folds(folds_check, two_price=True, silent=True)

    print(f"In-sample size: {n_in}")
    print(f"One-Price mean in-sample profit:  {np.mean(in_means_one):.15f} +- {np.std(in_means_one):.3f} MDKK")
    print(f"One-Price mean out-of-sample profit: {np.mean(out_means_one):.15f} +- {np.std(out_means_one):.3f} MDKK")
    print(f"Two-Price mean in-sample profit:  {np.mean(in_means_two):.15f} +- {np.std(in_means_two):.3f} MDKK")
    print(f"Two-Price mean out-of-sample profit: {np.mean(out_means_two):.15f} +- {np.std(out_means_two):.3f} MDKK")
    print("-" * 50)


#%% -------------------
# Task 1.4) Risk-Averse Offering Strategy
# ---------------------

beta_range = np.linspace(1e-6, 1, 20)

exp_profit_one, cvar_list_one, p_DA_list_one = uf.compute_profit_cvar_tradeoff(in_sample_scenarios, beta_range, scheme="one_price")

uf.plot_profit_cvar_tradeoff(cvar_list_one, exp_profit_one, beta_range, annotate=False)

uf.plot_DA_offers_risk(beta_range, p_DA_list_one)


# Two price

exp_profit_two, cvar_list_two, p_DA_list_two = uf.compute_profit_cvar_tradeoff(in_sample_scenarios, beta_range, scheme="two_price")


uf.plot_profit_cvar_tradeoff(cvar_list_two, exp_profit_two, beta_range)

# Only take 5 values from beta range for plotting DA offers
chosen_beta_two = [beta_range[0], beta_range[6], beta_range[13],beta_range[19]]
chosen_DA_offers_two = [p_DA_list_two[i] for i in [0, 6, 13, 19]]
uf.plot_DA_offers_risk(chosen_beta_two, chosen_DA_offers_two)



#%% EXAMINING DA OFFERS IN RISK-AVERSION

# Solve with beta = 1
_, p_DA_vec_one, _, profit_matrix_one, cvar_one, eta_vec_one = uf.solve_risk_averse_one_price(in_sample_scenarios, alpha=0.9, beta=1)
_, p_DA_vec_two, profit_matrix_two, cvar_two, eta_vec_two = uf.solve_risk_averse_two_price(in_sample_scenarios, alpha=0.9, beta=1)

# Function that solves one-price with a fixed bid
res_opt = uf.evaluate_fixed_bid_risk(in_sample_scenarios, p_DA_vec_one)

# Solve with a slightly lower bid to see if worst scenarios change
p_new = p_DA_vec_one.copy()
p_new[5] += -1
res_new = uf.evaluate_fixed_bid_risk(in_sample_scenarios, p_new)

# 4. Compare results
print(f"Optimal (43.586) CVaR: {res_opt['cvar']:.6f}")
print(f"New Bid (42.586) CVaR: {res_new['cvar']:.6f}")
print(f"\nWorst Scenarios at Optimal: {sorted(res_opt['worst_indices'])}")
print(f"Worst Scenarios at New Bid: {sorted(res_new['worst_indices'])}")

# Find which scenarios entered/left the tail
new_entries = set(res_new['worst_indices']) - set(res_opt['worst_indices'])
if new_entries:
    print(f"Scenarios that became 'worst' due to higher bid: {new_entries}")

# Check mean wind for worst-case scenario in 2-price case
worst_idx= np.where(eta_vec_two > 1e-9)[0]
average_wind_worst = in_sample_scenarios[:, worst_idx, 0].mean()
average_wind_overall = in_sample_scenarios[:, :, 0].mean()
print(f"\nAverage wind in worst-case scenarios: {average_wind_worst:.3f}")
print(f"Average wind overall: {average_wind_overall:.3f}")


#%% Change in profit distribution

# Comparison one price
profit_per_scenario_risk_1 = profit_matrix_one.sum(axis=0)
print("Plot comparison of profit distributions for one-price strategy")
print("Difference in total profit: ", round(profit_per_scenario.mean() - profit_per_scenario_risk_1.mean(),3), "MDKK")
uf.plot_cdf_comparison_cvar(profit_per_scenario_risk_1, profit_per_scenario, label_a="Risk-averse", label_b="Risk-neutral", title=None)
uf.plot_cdf_comparison_cvar(profit_per_scenario_risk_1, profit_per_scenario, label_a="Risk-averse", label_b="Risk-neutral", title=None, alpha=0)

#Comparison two price
profit_per_scenario_risk_2 = profit_matrix_two.sum(axis=0)
print("Plot comparison of profit distributions for two-price strategy")
print("Difference in total profit: ", round(profit_per_scenario_2.mean() - profit_per_scenario_risk_2.mean(),3), "MDKK")
uf.plot_cdf_comparison_cvar(profit_per_scenario_risk_2, profit_per_scenario_2, label_a="Risk-averse", label_b="Risk-neutral", title=None)
uf.plot_cdf_comparison_cvar(profit_per_scenario_risk_2, profit_per_scenario_2, label_a="Risk-averse", label_b="Risk-neutral", title=None, alpha=0)



#%% Comparison of different in-sample scenarios


_, p_DA_vec_comp, _, profit_matrix_comp, cvar_comp, eta_vec_comp = uf.solve_risk_averse_one_price(folds[1], alpha=0.9, beta=1)
_, p_DA_vec_comp2, _, profit_matrix_comp2, cvar_comp2, eta_vec_comp2 = uf.solve_risk_averse_one_price(folds[2], alpha=0.9, beta=1)
hours = np.arange(24)


# Compare beta=1 solutions for insample and fold 1

plt.figure(figsize=(12, 6))
plt.step(hours, p_DA_vec_one, where='post', label="Beta=1, All Scenarios", linewidth=2)
plt.step(hours, p_DA_vec_comp, where='post', label="Beta=1, Fold 1", linewidth=2)
plt.step(hours, p_DA_vec_comp2, where='post', label="Beta=1, Fold 2", linewidth=2)
plt.xticks(hours)
plt.xlim(0, 23)
plt.xlabel("Hour")
plt.ylabel("DA Offer (MW)")

plt.title("Comparison of DA Offers for Beta=1 with Different In-Sample Scenarios")
plt.legend()
plt.grid(True)
plt.show()


# Twoprice

_, p_DA_vec_comp, profit_matrix_comp, cvar_comp, eta_vec_comp = uf.solve_risk_averse_two_price(folds[1], alpha=0.9, beta=1)
_, p_DA_vec_comp2, profit_matrix_comp2, cvar_comp2, eta_vec_comp2 = uf.solve_risk_averse_two_price(folds[2], alpha=0.9, beta=1)
hours = np.arange(24)


# Compare beta=1 solutions for insample and fold 1

plt.figure(figsize=(12, 6))
plt.step(hours, p_DA_vec_two, where='post', label="Beta=1, All Scenarios", linewidth=2)
plt.step(hours, p_DA_vec_comp, where='post', label="Beta=1, Fold 1", linewidth=2)
plt.step(hours, p_DA_vec_comp2, where='post', label="Beta=1, Fold 2", linewidth=2)
plt.xticks(hours)
plt.xlim(0, 23)
plt.xlabel("Hour")
plt.ylabel("DA Offer (MW)")

plt.title("Comparison of DA Offers for Beta=1 with Different In-Sample Scenarios")
plt.legend()
plt.grid(True)
plt.show()










#%% ----------------------------------------
# EXTRA DATA PRINT AND EXTRA PLOTS
# ------------------------------------------

#%% Spotprice distribution

price_tdkk = in_sample_scenarios[:,:,1]*1e3
uf.plot_profit_distribution(price_tdkk.flatten(), n_bins = 30, title="Spotprice distribution per scenario and hour", x_label="Spotprice (tDKK/MWh)")

#%%


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

# compute deficit probabilities from folds

deficit_probs_all = np.array([
    fold[:, :, 2].mean(axis=1) for fold in folds
])

hours = np.arange(0, 24)

labelsize = 16
ticksize = 12
titlesize= 20
legendsize = 16

mean_probs = deficit_probs_all.mean(axis=0)

plt.figure(figsize=(12, 4))

for i in range(deficit_probs_all.shape[0]):
    plt.step(hours, deficit_probs_all[i], where='pre', alpha=0.7)

plt.axhline(0.375, linestyle='--', label="Threshold (0.375)")

plt.xlabel("Hour", fontsize=labelsize)
plt.ylabel("Deficit Probability", fontsize=labelsize)
plt.xlim(0,23)
plt.xticks(hours, fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.title("Deficit Probability per Fold", fontsize=titlesize)
plt.legend(fontsize=legendsize)
plt.grid(True, alpha=0.3)

plt.show()