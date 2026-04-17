def generate_scenarios(random_state=None):
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(random_state)

    # Extract wind data
    wind_data = pd.read_csv(
        "Data/ninja_wind_55.5783_15.7764_corrected.csv",
        comment='#',
        parse_dates=['time', 'local_time']
    )
    wind_data["electricity_mwh"] = wind_data["electricity"] * 1e-3
    wind_data.drop(columns=["local_time"], inplace=True)

    wind_data["section"] = (wind_data["time"].dt.dayofyear - 1) // 18

    selected_days = []
    wind_scenarios = []

    for section in range(20):
        section_data = wind_data[wind_data["section"] == section]

        random_day = rng.choice(section_data["time"].dt.dayofyear.unique())
        selected_days.append(random_day)

        scenario = section_data[section_data["time"].dt.dayofyear == random_day].copy()
        scenario["hour"] = scenario["time"].dt.hour
        scenario = scenario[["hour", "electricity_mwh"]]

        wind_scenarios.append(scenario["electricity_mwh"].values)

    # Extract price data
    price_data = pd.read_csv("Data/DayAheadPrices_DK2.csv", sep=";", decimal=",")
    price_data["SpotPriceMDKK"] = price_data["SpotPriceDKK"] * 1e-6
    price_data.drop(columns=["HourUTC", "PriceArea", "SpotPriceEUR"], inplace=True)
    price_data["HourDK"] = pd.to_datetime(price_data["HourDK"])

    price_data["dayofyear"] = price_data["HourDK"].dt.dayofyear

    price_data_filtered = price_data[~price_data["dayofyear"].isin(selected_days)].copy()
    price_data_filtered["section"] = (price_data_filtered["HourDK"].dt.dayofyear - 1) // 18

    price_scenarios = []

    for section in range(20):
        section_data = price_data_filtered[price_data_filtered["section"] == section]

        random_day = rng.choice(section_data["dayofyear"].unique())

        scenario = section_data[section_data["dayofyear"] == random_day].copy()
        scenario["hour"] = scenario["HourDK"].dt.hour
        scenario = scenario[["hour", "SpotPriceMDKK"]]
        scenario = scenario.sort_values("hour")

        price_scenarios.append(scenario["SpotPriceMDKK"].values)

    # Surplus/deficit scenarios
    surp_def_scenarios = rng.integers(0, 2, size=(4, 24))

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


def solve_stochastic_strategy_one_price(in_sample_scenarios):
    import gurobipy as gp
    import numpy as np

    P_nom = 500
    # in_sample_scenarios has shape (n_hours, n_scenarios, 3)
    p_real = in_sample_scenarios[:,:,0]
    lambda_DA = in_sample_scenarios[:,:,1]
    deficit_bin = in_sample_scenarios[:,:,2]

    m = gp.Model("stochastic_one_price")
    m.Params.OutputFlag = 0

    n_hours = len(in_sample_scenarios[:, 0, 0])
    n_scenarios = len(in_sample_scenarios[0, :, 0])
    prob_scenarios = 1 / n_scenarios

    # Parameters
    lambda_bal = 1.25 * lambda_DA * deficit_bin + 0.85 * lambda_DA * (1 - deficit_bin) 

    # Variables
    p_DA = m.addVars(n_hours, lb=0, name="DayAhead offer")
    Delta = m.addVars(n_hours, n_scenarios, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Difference_DA_real")

    m.setObjective(gp.quicksum(prob_scenarios * 
                                (lambda_DA[t, s] * p_DA[t]
                                 + lambda_bal[t, s] * Delta[t, s])
                                for t in range(n_hours) for s in range(n_scenarios)), gp.GRB.MAXIMIZE)


    # Constraints
    # Capacity limit 
    for t in range(n_hours):
        m.addConstr(p_DA[t] <= P_nom, name=f"Capacity_limit_{t}")

    # Constraints setting difference between DA and real production
    for t in range(n_hours):
        for s in range(n_scenarios):
            m.addConstr(Delta[t, s] == p_real[t, s] - p_DA[t], name=f"Difference_DA_real_{t}_{s}")


    # Optimize
    m.optimize()

    runtime_sec    = m.Runtime
    num_vars       = int(m.NumVars)
    num_constrs    = int(m.NumConstrs)

    print(f"  Computational time:    {runtime_sec:.6f} s")
    print(f"  Decision variables:    {num_vars}")
    print(f"  Constraints:           {num_constrs}")

    p_DA_vec = np.array([p_DA[t].X for t in range(n_hours)])
    Delta_mat = np.array([[Delta[t, s].X for s in range(n_scenarios)] for t in range(n_hours)])
    profit_matrix = lambda_DA * p_DA_vec[:, None] + lambda_bal * Delta_mat

    return m, p_DA, Delta, profit_matrix


def solve_stochastic_strategy_two_price(in_sample_scenarios):
    import gurobipy as gp
    import numpy as np

    # in_sample_scenarios has shape (n_hours, n_scenarios, 3)
    p_real = in_sample_scenarios[:,:,0]
    lambda_DA = in_sample_scenarios[:,:,1]
    deficit_bin = in_sample_scenarios[:,:,2]

    m = gp.Model("stochastic_two_price")
    m.Params.OutputFlag = 0

    n_hours = len(in_sample_scenarios[:, 0, 0])
    n_scenarios = len(in_sample_scenarios[0, :, 0])
    prob_scenarios = 1 / n_scenarios

    # Parameters
    P_nom = 500 
    lambda_bal_up   = deficit_bin * lambda_DA + (1 - deficit_bin) * 0.85 * lambda_DA
    lambda_bal_down = deficit_bin * 1.25 * lambda_DA + (1 - deficit_bin) * lambda_DA

    # Variables
    p_DA = m.addVars(n_hours, lb=0, name="DayAhead offer")
    Delta = m.addVars(n_hours, n_scenarios, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Difference_DA_real")
    Delta_up = m.addVars(n_hours, n_scenarios, lb=0, name="difference over 0")
    Delta_down = m.addVars(n_hours, n_scenarios, lb=0, name="difference under 0")

    m.setObjective(gp.quicksum(
        prob_scenarios * (
            lambda_DA[t, s] * p_DA[t]
            + lambda_bal_up[t, s]   * Delta_up[t, s]
            - lambda_bal_down[t, s] * Delta_down[t, s])
        for t in range(n_hours) for s in range(n_scenarios)),  gp.GRB.MAXIMIZE)

    # Capacity limit 
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
    
    runtime_sec    = m.Runtime
    num_vars       = int(m.NumVars)
    num_constrs    = int(m.NumConstrs)

    print(f"  Computational time:    {runtime_sec:.6f} s")
    print(f"  Decision variables:    {num_vars}")
    print(f"  Constraints:           {num_constrs}")


    p_DA_vec = np.array([p_DA[t].X for t in range(n_hours)])
    Delta_up_mat = np.array([[Delta_up[t, s].X for s in range(n_scenarios)] for t in range(n_hours)])
    Delta_down_mat = np.array([[Delta_down[t, s].X for s in range(n_scenarios)] for t in range(n_hours)])

    profit_matrix = (lambda_DA * p_DA_vec[:, None] + lambda_bal_up * Delta_up_mat - lambda_bal_down * Delta_down_mat)

    return m, p_DA, Delta_up, Delta_down, profit_matrix


def plot_profit_distribution(profit_per_scenario, n_bins = 15, title="Profit distribution per scenario"):
    import matplotlib.pyplot as plt

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
