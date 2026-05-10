#%%
import importlib
import numpy as np
import usefulfunctions as uf

# Ensure newest version of usefulfunctions
importlib.reload(uf)

# Prints appendix if true
APPENDIX = True

# Set seed, 42 have been used for step 1
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
profit_per_hour = profit_matrix.mean(axis=1)
profit_per_scenario = profit_matrix.sum(axis=0)

# Print expected profit and day-ahead offers
print("Expected Profit (One-Price):", round(m.ObjVal, 3), "MDKK")
print(f"Min profit (One-price): {profit_per_scenario.min():.2f} MDKK")
print(f"Max profit (One-price): {profit_per_scenario.max():.2f} MDKK")

# Plot day-ahead offers
#uf.plot_DA_offers(p_DA, in_sample_scenarios, title="Day-ahead offers - One-Price", Threshold_value=0.375)
uf.plot_DA_offers(p_DA, in_sample_scenarios, title=None, Threshold_value=0.375)

# Plot profit distribution per scenario
uf.plot_profit_distribution(profit_per_scenario, n_bins = 30, title="Profit distribution per scenario in one-price system")

#uf.plot_cumulative_profit_distribution(profit_per_scenario, title="Cumulative profit distribution - One-price")
uf.plot_cumulative_profit_distribution(profit_per_scenario, title=None)


#%% ----------------------
# Task 1.2) Offering Strategy Under a Two-Price Balancing Scheme
# ------------------------

# Use same 200 random scenarios as in Task 1.1
# Solve optimization problem
m_2, p_DA_2, Delta_up, Delta_down, profit_matrix_2 = uf.solve_stochastic_strategy_two_price(in_sample_scenarios)
profit_per_hour_2 = profit_matrix_2.mean(axis=1)
profit_per_scenario_2 = profit_matrix_2.sum(axis=0)

# Print expected profit and day-ahead offers
print("Expected Profit (Two-Price):", round(m_2.ObjVal,3), "MDKK")
print(f"Min profit (Two-price): {profit_per_scenario_2.min():.2f} MDKK")
print(f"Max profit (Two-price): {profit_per_scenario_2.max():.2f} MDKK")

# Plot day-ahead offers
uf.plot_DA_offers(p_DA_2, in_sample_scenarios, title="Day-ahead offers - Two-Price")

# CDF comparing one-price and two-price
uf.plot_cdf_comparison(profit_per_scenario_2, profit_per_scenario, label_a="Two-price", label_b="One-price", title="Cumulative profit distribution comparison", alpha=0, ls_second=":", mean=True)

# Profit comparison plot
uf.plot_profit_distribution_comparison(profit_per_scenario, profit_per_scenario_2, n_bins=30)

#%% -----------------------
# Task 1.3) Cross-Validation of Offering Strategies
# -------------------------
importlib.reload(uf)

# Create 8 folds
folds = uf.create_folds(scenarios, n_in_sample=200, seed=rng) # Creates a list of arrays, each of shape (24, 200, 3)

# One-price cross-validation
in_sample_means_one, out_sample_means_one = uf.cross_validate_folds(folds, two_price=False)

print(f"One-Price mean in-sample profit:  {np.mean(in_sample_means_one):.15f} MDKK")
print(f"One-Price mean out-of-sample profit: {np.mean(out_sample_means_one):.15f} MDKK")
print(f"Std in-sample profit:   {np.std(in_sample_means_one):.3f}")
print(f"Std out-of-sample profit:{np.std(out_sample_means_one):.3f}")

# Bar chart with profits under one-price
uf.plot_Cross_Validation_Profits(in_sample_means_one, out_sample_means_one, title="Cross-Validation mean profits - One-Price")

# Plot of deficit probability per fold
uf.plot_deficit_probabilities(folds)


# Two_Price cross-validation
in_sample_means_two, out_sample_means_two = uf.cross_validate_folds(folds, two_price=True)

print(f"Two-Price mean in-sample profit:  {np.mean(in_sample_means_two):.15f} MDKK")
print(f"Two-Price mean out-of-sample profit: {np.mean(out_sample_means_two):.15f} MDKK")
print(f"Std in-sample profit:   {np.std(in_sample_means_two):.3f}")
print(f"Std out-of-sample profit:{np.std(out_sample_means_two):.3f}")

# Bar chart with profits under two-price
uf.plot_Cross_Validation_Profits(in_sample_means_two, out_sample_means_two, title="Cross-Validation mean profits - One-Price")



# Results with varying in-sample size
in_sample_list = [50, 100, 200, 400, 800]

for n_in in in_sample_list:
    folds_check = uf.create_folds(scenarios, n_in_sample=n_in, seed=rng)
    in_means_one, out_means_one = uf.cross_validate_folds(folds_check, two_price=False)
    in_means_two, out_means_two = uf.cross_validate_folds(folds_check, two_price=True)

    print(f"In-sample size: {n_in}")
    print(f"One-Price mean in-sample profit:  {np.mean(in_means_one):.3f} +- {np.std(in_means_one):.3f} MDKK")
    print(f"One-Price mean out-of-sample profit: {np.mean(out_means_one):.3f} +- {np.std(out_means_one):.3f} MDKK")
    print(f"Two-Price mean in-sample profit:  {np.mean(in_means_two):.3f} +- {np.std(in_means_two):.3f} MDKK")
    print(f"Two-Price mean out-of-sample profit: {np.mean(out_means_two):.3f} +- {np.std(out_means_two):.3f} MDKK")
    print("-" * 50)



#%% -------------------
# Task 1.4) Risk-Averse Offering Strategy
# ---------------------
importlib.reload(uf)

# Test for 20 values of beta between 0 and 1
beta_range = np.linspace(1e-6, 1, 20)

# One-price
print("Risk analysis under one-price scheme:")
exp_profit_one, cvar_list_one, p_DA_list_one = uf.compute_profit_cvar_tradeoff(in_sample_scenarios, beta_range, scheme="one_price", silent=True)

# Plot of profit vs cvar, and of DA offers
uf.plot_profit_cvar_tradeoff(cvar_list_one, exp_profit_one, beta_range, annotate=False, title="Two-price: risk-return tradeoff")
uf.plot_DA_offers_risk(beta_range, p_DA_list_one, title="One-price: Day-ahead offers for different β")


# Two price
print("Risk analysis under two-price scheme:")
exp_profit_two, cvar_list_two, p_DA_list_two = uf.compute_profit_cvar_tradeoff(in_sample_scenarios, beta_range, scheme="two_price")

# Plot of profit vs cvar, and of DA offers for 5 distinct beta values
uf.plot_profit_cvar_tradeoff(cvar_list_two, exp_profit_two, beta_range, title="Two-price: risk-return tradeoff")

chosen_beta_two = [beta_range[0], beta_range[6], beta_range[13],beta_range[19]]
chosen_DA_offers_two = [p_DA_list_two[i] for i in [0, 6, 13, 19]]
uf.plot_DA_offers_risk(chosen_beta_two, chosen_DA_offers_two, title="Two-price: Day-ahead offers for different β")


#%% Change in profit distribution

# Solve with beta = 1
_, p_DA_vec_one, _, profit_matrix_one, cvar_one, eta_vec_one = uf.solve_risk_averse_one_price(in_sample_scenarios, alpha=0.9, beta=1, silent=True)
_, p_DA_vec_two, profit_matrix_two, cvar_two, eta_vec_two = uf.solve_risk_averse_two_price(in_sample_scenarios, alpha=0.9, beta=1, silent=True)


# Comparison one price
profit_per_scenario_risk_1 = profit_matrix_one.sum(axis=0)
print("Plot comparison of profit distributions for one-price strategy")
print("Difference in total profit: ", round(profit_per_scenario.mean() - profit_per_scenario_risk_1.mean(),3), "MDKK")
uf.plot_cdf_comparison(profit_per_scenario_risk_1, profit_per_scenario, label_a="Risk-averse", label_b="Risk-neutral", title="One-price: CDF tail of risk-averse and risk-neutral profits")
uf.plot_cdf_comparison(profit_per_scenario_risk_1, profit_per_scenario, label_a="Risk-averse", label_b="Risk-neutral", title="One-price: CDF comparison risk-averse and risk-neutral profits", alpha=0)

#Comparison two price
profit_per_scenario_risk_2 = profit_matrix_two.sum(axis=0)
print("Plot comparison of profit distributions for two-price strategy")
print("Difference in total profit: ", round(profit_per_scenario_2.mean() - profit_per_scenario_risk_2.mean(),3), "MDKK")
uf.plot_cdf_comparison(profit_per_scenario_risk_2, profit_per_scenario_2, label_a="Risk-averse", label_b="Risk-neutral", title="Two-price: CDF tail of risk-averse and risk-neutral profits")
uf.plot_cdf_comparison(profit_per_scenario_risk_2, profit_per_scenario_2, label_a="Risk-averse", label_b="Risk-neutral", title="One-price: CDF comparison risk-averse and risk-neutral profits", alpha=0)



#%% EXAMINE VARIATION IN FOLD OR INPUT DATA

# Plot DA offers for different folds under risk aversion
uf.plot_DA_offer_folds(folds)

# Compute DA offers and profits with 100 different input seeds
p_da_list, profits, cvar = uf.compute_DA_offer_samples(n_runs=100)

profits_day = profits.sum(axis=1)

# Visualize results with boxplot and scatterplot
uf.plot_boxplot_profit_cvar(profits_day, cvar, profit_per_scenario_2.mean(), cvar_two)
uf.plot_scatter_profit_cvar(profits_day, cvar, profit_per_scenario_2.mean(), cvar_two)


#%% ----------------------------------------
# APPENDIX
# ------------------------------------------

if (APPENDIX):

    price_tdkk = in_sample_scenarios[:,:,1]*1e3
    uf.plot_profit_distribution(price_tdkk.flatten(), n_bins = 30, title="Spotprice distribution per scenario and hour", x_label="Spotprice (tDKK/MWh)")


    # ---------------
    # PLOT OF SAMPLES
    # ---------------

    uf.plot_scenarios(scenarios)


    # EXAMINING DA OFFERS IN RISK-AVERSION
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
