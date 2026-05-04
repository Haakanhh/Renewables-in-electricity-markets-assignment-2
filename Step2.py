#%%
import importlib
import numpy as np
import pandas as pd
import usefulfunctions as uf
import matplotlib.pyplot as plt
import time

importlib.reload(uf)

seed = 40
rng = np.random.default_rng(seed)

# Generate scenarios
Load_profiles = uf.Load_profile_generation(random_state=seed, Profiles=300, P_max=600, P_min=220, P_delta=35)

#%% ----------------------
# Task 2.1) In-sample Decision Making: Offering Strategy Under the P90 Requirement
# ------------------------

# Pick random profiles
sample_rng = np.random.default_rng(seed)
n_insample_profiles = 100
idx = sample_rng.choice(Load_profiles.shape[0], size=n_insample_profiles, replace=False)
in_sample_profiles = Load_profiles[idx, :]

# Epsilon under P90 requirement
epsilon = 0.1

#Budget for violation under P90 requirement
q = epsilon*n_insample_profiles*60

# Solve ALSO-X optimization problem
m, c_up_AlsoX, y, F_up_AlsoX =uf.Optimal_reserve_bid_ALSO_X (in_sample_profiles, q, M=10**4, silent=False)

print(c_up_AlsoX)
#%%
uf.histogram_of_violations(c_up_AlsoX, F_up_AlsoX, title="Histogram of violations per profile - ALSO-X")

#%%
# Epsilon under P90 requirement
epsilon = 0.1

m, c_up_CVaR, beta, zeta, F_up_CVaR = uf.Optimal_reserve_bid_CVaR (in_sample_profiles, epsilon, silent=False)

print(c_up_CVaR)

uf.histogram_of_violations(c_up_CVaR, F_up_CVaR, title="Histogram of violations per profile - CVaR")


#%% ----------------------
# Task 2.2) Verification of the P90 Requirement Using Out-of-Sample Analysis
# ------------------------

# Remaining 200 profiles for out-of-sample analysis
out_idx = np.setdiff1d(np.arange(Load_profiles.shape[0]), idx)
out_sample_profiles = Load_profiles[out_idx, :]

#Total number of points in out-of-sample profiles
n_out_profiles, n_minutes = out_sample_profiles.shape
n_points = n_out_profiles * n_minutes

# ALSO-X
not_available_AlsoX_mask = out_sample_profiles < c_up_AlsoX
n_not_available_AlsoX = int(not_available_AlsoX_mask.sum())
share_not_available_AlsoX = n_not_available_AlsoX / n_points

# CVaR
not_available_CVaR_mask = out_sample_profiles < c_up_CVaR
n_not_available_CVaR = int(not_available_CVaR_mask.sum())
share_not_available_CVaR = n_not_available_CVaR / n_points


print(f"Total points (profiles x minutes): {n_points}")
print(f"ALSO-X  -> not available count: {n_not_available_AlsoX}, share: {share_not_available_AlsoX:.4%}")
print(f"CVaR    -> not available count: {n_not_available_CVaR}, share: {share_not_available_CVaR:.4%}")


# Profile-level violations (a profile violates if any minute violates)
profiles_with_violation_AlsoX = (not_available_AlsoX_mask.sum(axis=1) > 6)
profiles_with_violation_CVaR = (not_available_CVaR_mask.sum(axis=1) > 6)

n_profiles_with_violation_AlsoX = int(profiles_with_violation_AlsoX.sum())
n_profiles_with_violation_CVaR = int(profiles_with_violation_CVaR.sum())

share_profiles_with_violation_AlsoX = n_profiles_with_violation_AlsoX / n_out_profiles
share_profiles_with_violation_CVaR = n_profiles_with_violation_CVaR / n_out_profiles

print(f"ALSO-X -> profiles with >6 violations: {n_profiles_with_violation_AlsoX}/{n_out_profiles}, share: {share_profiles_with_violation_AlsoX:.4%}")
print(f"CVaR -> profiles with >6 violations: {n_profiles_with_violation_CVaR}/{n_out_profiles}, share: {share_profiles_with_violation_CVaR:.4%}")

#%% ----------------------
# Task 2.3) Energinet Perspective
# ------------------------

# Sweep epsilon from 0.00 to 0.20 in steps of 0.01 and store the ALSO-X solution.
epsilon_values = np.round(np.arange(0.00, 0.201, 0.01), 2)
alsox_results = []

start_time = time.time()

for epsilon in epsilon_values:
	# Budget for violation under P90 requirement
	q = epsilon * n_insample_profiles * 60

	# Solve ALSO-X optimization problem
	m, c_up_AlsoX, y, F_up_AlsoX = uf.Optimal_reserve_bid_ALSO_X(in_sample_profiles, q, M=10**4, silent=True)

	alsox_results.append({"epsilon": epsilon, "q": q, "c_up_AlsoX": c_up_AlsoX})
	print(f"epsilon={epsilon:.2f}, c_up_AlsoX={c_up_AlsoX}")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"\nLoop completed in {elapsed_time:.2f} seconds")

alsox_results_df = pd.DataFrame(alsox_results)


#%%

# Add reliability requirement (Pxx)
alsox_results_df["Reliability requirement"] = alsox_results_df["epsilon"].apply(lambda e: f"P{int((1-e)*100)}")

# Calculate share of unavailable points for each reliability requirement
alsox_results_df["share_not_available"] = alsox_results_df["c_up_AlsoX"].apply(lambda c: (out_sample_profiles < c).sum() / n_points *100)

uf.plot_Pxx_comparison(alsox_results_df)

#%%

alsox_results_df["mean_shortfall"] = alsox_results_df["c_up_AlsoX"].apply(
    lambda c: np.maximum(c - out_sample_profiles, 0).mean()
)

uf.plot_Pxx_comparison_mean(alsox_results_df)