import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp

def generate_scenarios(random_state=None, n_wind=20, n_price=20, n_surp_def=4):
    rng = np.random.default_rng(random_state)

    # Extract wind data
    wind_data = pd.read_csv(
        "Data/ninja_wind_55.5783_15.7764_corrected.csv",
        comment='#',
        parse_dates=['time', 'local_time']
    )

    # REMOVE DAYS WITH 23 OR 25 HOURS
    wind_data["dayofyear"] = wind_data["time"].dt.dayofyear
    wind_data["hour"] = wind_data["time"].dt.hour

    # keep only days with exactly 24 observations
    valid_wind_days = (
        wind_data.groupby("dayofyear")["hour"]
        .count()
        .pipe(lambda x: x[x == 24].index)
    )

    wind_data = wind_data[wind_data["dayofyear"].isin(valid_wind_days)].copy()

    wind_data["electricity_mwh"] = wind_data["electricity"] * 1e-3
    wind_data.drop(columns=["local_time"], inplace=True)

    days_per_section = 365 / n_wind
    wind_data["section"] = ((wind_data["time"].dt.dayofyear - 1) // days_per_section).astype(int)
    
    selected_days = []
    wind_scenarios = []

    for section in range(n_wind):
        section_data = wind_data[wind_data["section"] == section]

        random_day = rng.choice(section_data["time"].dt.dayofyear.unique())
        selected_days.append(random_day)

        scenario = section_data[section_data["time"].dt.dayofyear == random_day].copy()
        scenario["hour"] = scenario["time"].dt.hour
        scenario = scenario[["hour", "electricity_mwh"]]

        wind_scenarios.append(scenario["electricity_mwh"].values)

    # Extract price data
    price_data = pd.read_csv("Data/DayAheadPrices_DK2.csv", sep=";", decimal=",")
    price_data["SpotPriceMDKK"] = price_data["SpotPriceDKK"] * 1e-6 # convert to MDKK / MWh
    price_data.drop(columns=["HourUTC", "PriceArea", "SpotPriceEUR"], inplace=True)
    price_data["HourDK"] = pd.to_datetime(price_data["HourDK"])

    # REMOVE DAYS WITH 23 OR 25 HOURS
    price_data["dayofyear"] = price_data["HourDK"].dt.dayofyear
    price_data["hour"] = price_data["HourDK"].dt.hour

    valid_price_days = (
        price_data.groupby("dayofyear")["hour"]
        .count()
        .pipe(lambda x: x[x == 24].index)
    )

    price_data = price_data[price_data["dayofyear"].isin(valid_price_days)].copy()

    # SET NEGATIVE PRICES TO MINIMAL PRICE
    price_data["SpotPriceMDKK"] = price_data["SpotPriceMDKK"].clip(lower=1e-6)

    price_data_filtered = price_data[~price_data["dayofyear"].isin(selected_days)].copy()
    price_data_filtered["section"] = pd.cut(
        price_data_filtered["HourDK"].dt.dayofyear - 1,
        bins=n_price,
        labels=False
    )
    price_scenarios = []

    for section in range(n_price):
        section_data = price_data_filtered[price_data_filtered["section"] == section]

        random_day = rng.choice(section_data["dayofyear"].unique())

        scenario = section_data[section_data["dayofyear"] == random_day].copy()
        scenario["hour"] = scenario["HourDK"].dt.hour
        scenario = scenario[["hour", "SpotPriceMDKK"]]
        scenario = scenario.sort_values("hour")

        price_scenarios.append(scenario["SpotPriceMDKK"].values)

    # Surplus/deficit scenarios
    surp_def_scenarios = rng.integers(0, 2, size=(n_surp_def, 24))

    # Build combined scenarios
    all_scenarios = [
        (w_i, p_i, sd_i)
        for w_i in range(len(wind_scenarios))
        for p_i in range(len(price_scenarios))
        for sd_i in range(len(surp_def_scenarios))
    ]

    n_hours = 24
    n_scenarios = len(all_scenarios)

    data = np.zeros((n_hours, n_scenarios, 3))

    for s, (w_i, p_i, sd_i) in enumerate(all_scenarios):
        data[:, s, 0] = wind_scenarios[w_i]
        data[:, s, 1] = price_scenarios[p_i]
        data[:, s, 2] = surp_def_scenarios[sd_i]

    return data


def solve_stochastic_strategy_one_price(in_sample_scenarios, silent=False):
    
    # in_sample_scenarios has shape (n_hours, n_scenarios, 3)
    p_real = in_sample_scenarios[:,:,0]
    lambda_DA = in_sample_scenarios[:,:,1]
    deficit_bin = in_sample_scenarios[:,:,2]

    m = gp.Model("stochastic_one_price")
    m.Params.OutputFlag = 0

    # Parameters
    P_nom = 500
    n_hours = len(in_sample_scenarios[:, 0, 0])
    n_scenarios = len(in_sample_scenarios[0, :, 0])
    prob_scenarios = 1 / n_scenarios
    lambda_bal = 1.25 * lambda_DA * deficit_bin + 0.85 * lambda_DA * (1 - deficit_bin) 

    # Variables
    p_DA = m.addVars(n_hours, lb=0, name="DayAhead offer")
    Delta = m.addVars(n_hours, n_scenarios, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Difference_DA_real")

    # Objective
    m.setObjective(gp.quicksum(prob_scenarios * 
                                (lambda_DA[t, s] * p_DA[t]
                                 + lambda_bal[t, s] * Delta[t, s])
                                for t in range(n_hours) for s in range(n_scenarios)), gp.GRB.MAXIMIZE)


    # Capacity limit constraint
    for t in range(n_hours):
        m.addConstr(p_DA[t] <= P_nom, name=f"Capacity_limit_{t}")

    # Constraints setting difference between DA and real production
    for t in range(n_hours):
        for s in range(n_scenarios):
            m.addConstr(Delta[t, s] == p_real[t, s] - p_DA[t], name=f"Difference_DA_real_{t}_{s}")


    # Optimize
    m.optimize()

    if silent == False:
        # Print computational details
        runtime_sec    = m.Runtime
        num_vars       = int(m.NumVars)
        num_constrs    = int(m.NumConstrs)
        print(f"  Computational time:    {runtime_sec:.6f} s")
        print(f"  Decision variables:    {num_vars}")
        print(f"  Constraints:           {num_constrs}")

    # Compute profit per hour and scenario
    p_DA_vec = np.array([p_DA[t].X for t in range(n_hours)])
    Delta_mat = np.array([[Delta[t, s].X for s in range(n_scenarios)] for t in range(n_hours)])
    profit_matrix = lambda_DA * p_DA_vec[:, None] + lambda_bal * Delta_mat

    return m, p_DA, Delta, profit_matrix


def solve_stochastic_strategy_two_price(in_sample_scenarios):

    # in_sample_scenarios has shape (n_hours, n_scenarios, 3)
    p_real = in_sample_scenarios[:,:,0]
    lambda_DA = in_sample_scenarios[:,:,1]
    deficit_bin = in_sample_scenarios[:,:,2]

    m = gp.Model("stochastic_two_price")
    m.Params.OutputFlag = 0

    # Parameters
    n_hours = len(in_sample_scenarios[:, 0, 0])
    n_scenarios = len(in_sample_scenarios[0, :, 0])
    prob_scenarios = 1 / n_scenarios
    P_nom = 500 
    lambda_bal_up   = deficit_bin * lambda_DA + (1 - deficit_bin) * 0.85 * lambda_DA
    lambda_bal_down = deficit_bin * 1.25 * lambda_DA + (1 - deficit_bin) * lambda_DA

    # Variables
    p_DA = m.addVars(n_hours, lb=0, name="DayAhead offer")
    Delta = m.addVars(n_hours, n_scenarios, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Difference_DA_real")
    Delta_up = m.addVars(n_hours, n_scenarios, lb=0, name="difference over 0")
    Delta_down = m.addVars(n_hours, n_scenarios, lb=0, name="difference under 0")

    # Objective
    m.setObjective(gp.quicksum(
        prob_scenarios * (
            lambda_DA[t, s] * p_DA[t]
            + lambda_bal_up[t, s]   * Delta_up[t, s]
            - lambda_bal_down[t, s] * Delta_down[t, s])
        for t in range(n_hours) for s in range(n_scenarios)),  gp.GRB.MAXIMIZE)

    # Capacity limit constraint
    for t in range(n_hours):
        m.addConstr(p_DA[t] <= P_nom, name=f"Capacity_limit_{t}")

    # Constraints setting limits for Delta up and down
    for t in range(n_hours):
        for s in range(n_scenarios):
            #m.addConstr(Delta_up[t, s] - Delta_down[t, s] == p_real[t, s] - p_DA[t])
            m.addConstr(Delta[t, s] == p_real[t, s] - p_DA[t], name=f"Difference_DA_real_{t}_{s}")
            m.addConstr(Delta[t, s] == Delta_up[t, s] - Delta_down[t, s], name=f"Difference_split_{t}_{s}")

    b = m.addVars(n_hours, n_scenarios, vtype=gp.GRB.BINARY, name="split_bin")
    M = P_nom  # natural upper bound, since imbalance can't exceed wind capacity

    for t in range(n_hours):
        for s in range(n_scenarios):
            m.addConstr(Delta_up[t, s]   <= M * b[t, s])
            m.addConstr(Delta_down[t, s] <= M * (1 - b[t, s]))

    # Optimize
    m.optimize()
    
    # Print computational details
    runtime_sec    = m.Runtime
    num_vars       = int(m.NumVars)
    num_constrs    = int(m.NumConstrs)
    print(f"  Computational time:    {runtime_sec:.6f} s")
    print(f"  Decision variables:    {num_vars}")
    print(f"  Constraints:           {num_constrs}")

    # Compute profit per hour and scenario
    p_DA_vec = np.array([p_DA[t].X for t in range(n_hours)])
    Delta_up_mat = np.array([[Delta_up[t, s].X for s in range(n_scenarios)] for t in range(n_hours)])
    Delta_down_mat = np.array([[Delta_down[t, s].X for s in range(n_scenarios)] for t in range(n_hours)])
    profit_matrix = (lambda_DA * p_DA_vec[:, None] + lambda_bal_up * Delta_up_mat - lambda_bal_down * Delta_down_mat)

    return m, p_DA, Delta_up, Delta_down, profit_matrix

def plot_DA_offers(p_DA, in_sample_scenarios, title="Day-ahead offers"):
    # Average forecasted wind production per hour (24-hour profile)
    avg_forecasted_wind_per_hour = in_sample_scenarios[:, :, 0].mean(axis=1)
    df_avg_forecasted_wind = pd.DataFrame({
        "hour": np.arange(24),
        "avg_forecasted_wind_mwh": avg_forecasted_wind_per_hour
    })

    hours = np.arange(24)

    plt.figure(figsize=(12, 6))
    plt.step(hours, p_DA, where="post", marker="o", linewidth=2, label="Offering capacity")
    plt.step(hours, avg_forecasted_wind_per_hour, where="post", marker="s", linewidth=2, label="Avg. forecasted wind production")

    plt.title(title, fontsize=16)
    plt.xlabel("Hour", fontsize=12)
    plt.ylabel("Power (MW)", fontsize=12)
    plt.xticks(np.arange(24))
    plt.xlim(0, 23)
    plt.ylim(0, 550)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_profit_distribution(profit_per_scenario, n_bins = 15, title="Profit distribution per scenario"):
    plt.figure(figsize=(10,6))
    plt.hist(profit_per_scenario, bins=n_bins)
    plt.axvline(profit_per_scenario.mean(), color='red', linestyle='dashed', label=f"Mean: {profit_per_scenario.mean():.2f} MDKK")
    plt.title(title, fontsize=22)
    plt.xlabel("Total profit (MDKK)", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.show()

def plot_profit_distribution_comparison(profit_per_scenario, profit_per_scenario_2, n_bins=15):

    all_profits = np.concatenate([profit_per_scenario, profit_per_scenario_2])
    bins = np.linspace(all_profits.min(), all_profits.max(), n_bins)  # 12 bins → 13 edges

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

def create_folds(scenarios, n_in_sample, seed=42):
    n_samples = scenarios.shape[1]
    rng = np.random.default_rng(seed)

    # Shuffle indices
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_folds = n_samples // n_in_sample

    # Create folds
    folds = [
        scenarios[:, indices[i * n_in_sample : (i + 1) * n_in_sample], :]
        for i in range(n_folds)
    ]

    return folds

def calculate_profit(scenarios_in, scenarios_out, p_DA, two_price=False):

    # scenarios shape: (n_hours, n_scenarios, 3)
    p_real_out = scenarios_out[:, :, 0]
    lambda_DA_out = scenarios_out[:, :, 1]
    deficit_bin_out = scenarios_out[:, :, 2]

    # Reconstruct p_DA as vector
    # (since it's a gurobi dict)
    n_hours = scenarios_out.shape[0]
    p_DA_vec = np.array([p_DA[t].X for t in range(n_hours)])

    if two_price:
        # Balancing price
        lambda_bal_up   = deficit_bin_out * lambda_DA_out + (1 - deficit_bin_out) * 0.85 * lambda_DA_out
        lambda_bal_down = deficit_bin_out * 1.25 * lambda_DA_out + (1 - deficit_bin_out) * lambda_DA_out
        Delta = p_real_out - p_DA_vec[:, None]

        Delta_up = np.maximum(Delta, 0)
        Delta_down = np.maximum(-Delta, 0)

        profit_matrix = (
            lambda_DA_out * p_DA_vec[:, None]
            + lambda_bal_up * Delta_up
            - lambda_bal_down * Delta_down
        )

        profit_per_scenario = profit_matrix.sum(axis=0)

        return profit_per_scenario

    else:
        # Balancing price
        lambda_bal = 1.25 * lambda_DA_out * deficit_bin_out + 0.85 * lambda_DA_out * (1 - deficit_bin_out)

        # Compute Delta = p_real - p_DA
        Delta = p_real_out - p_DA_vec[:, None]

        # Profit per hour & scenario
        profit_matrix = lambda_DA_out * p_DA_vec[:, None] + lambda_bal * Delta

        # Return total profit per scenario
        profit_per_scenario = profit_matrix.sum(axis=0)

        return profit_per_scenario