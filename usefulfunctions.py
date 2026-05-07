import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp

def generate_scenarios(random_state=None, n_wind=20, n_price=20, n_surp_def=4):
    """
    Generate combined scenario data from historical wind, price, and surplus/deficit data.
    
    Parameters:
    -----------
    random_state : int or np.random.Generator, optional
        Seed for the random number generator for reproducibility. If None, uses unseeded randomness.
    n_wind : int, optional
        Number of seasonal wind scenarios to generate - 1 scenario is hourly data for 1 day, default 20. 
    n_price : int, optional
        Number of seasonal price scenarios to generate - 1 scenario is hourly data for 1 day, default 20. 
    n_surp_def : int, optional
        Number of random surplus/deficit scenarios to generate - 1 scenario is hourly data for 1 day, default 4.
        
    Returns:
    --------
    data : np.ndarray
        Array of shape (24, n_scenarios, 3) where:
        - 24 = hours per day
        - n_scenarios = n_wind × n_price × n_surp_def (total scenario count)
        - 3 channels: [wind production (MWh), day-ahead price (DKK/MWh), surplus/deficit binary]
    """
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
    """
    Solve optimal day-ahead offer using stochastic optimization with single balancing price.
    
    Parameters:
    -----------
    in_sample_scenarios : np.ndarray
        Array of shape (n_hours, n_scenarios, 3) containing sampled scenarios where:
        - n_hours = 24 (hourly offers over one day)
        - n_scenarios = number of Monte Carlo scenarios
        - 3 channels: [wind production (MWh), day-ahead price (DKK/MWh), surplus/deficit binary]
    silent : bool, optional
        If False (default), print computational details (runtime, variables, constraints).
        
    Returns:
    --------
    m : gurobipy.Model
        The solved Gurobi optimization model.
    p_DA_vec : np.ndarray
        Array of shape (24,) containing optimal day-ahead offer per hour in MWh.
    Delta : gurobipy.tupledict
        Variable dictionary of shape (24, n_scenarios) representing the difference between 
        real production and day-ahead offer for each hour and scenario.
    profit_matrix : np.ndarray
        Array of shape (24, n_scenarios) containing profit per hour and scenario in MDKK.
    """
    
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

    return m, p_DA_vec, Delta, profit_matrix


def solve_stochastic_strategy_two_price(in_sample_scenarios, silent=False):
    """
    Solve optimal day-ahead offer using stochastic optimization with two-price balancing.
    
    Parameters:
    -----------
    in_sample_scenarios : np.ndarray
        Array of shape (n_hours, n_scenarios, 3) containing sampled scenarios where:
        - n_hours = 24 (hourly offers over one day)
        - n_scenarios = number of Monte Carlo scenarios
        - 3 channels: [wind production (MWh), day-ahead price (DKK/MWh), surplus/deficit binary]
    silent : bool, optional
        If False (default), print computational details (runtime, variables, constraints).
        
    Returns:
    --------
    m : gurobipy.Model
        The solved Gurobi optimization model.
    p_DA_vec : np.ndarray
        Array of shape (24,) containing optimal day-ahead offer per hour in MWh.
    Delta_up : gurobipy.tupledict
        Variable dictionary of shape (24, n_scenarios) representing positive imbalances 
        (production above day-ahead offer) for each hour and scenario.
    Delta_down : gurobipy.tupledict
        Variable dictionary of shape (24, n_scenarios) representing negative imbalances 
        (production below day-ahead offer) for each hour and scenario.
    profit_matrix : np.ndarray
        Array of shape (24, n_scenarios) containing profit per hour and scenario in MDKK.
    """

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
    #Delta = m.addVars(n_hours, n_scenarios, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Difference_DA_real")
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
            m.addConstr(Delta_up[t, s] - Delta_down[t, s] == p_real[t, s] - p_DA[t], name=f"Difference_DA_real_{t}_{s}")
            #m.addConstr(Delta[t, s] == Delta_up[t, s] - Delta_down[t, s], name=f"Difference_split_{t}_{s}")

    
    #b = m.addVars(n_hours, n_scenarios, vtype=gp.GRB.BINARY, name="split_bin")
    #M = P_nom  # natural upper bound, since imbalance can't exceed wind capacity

    #for t in range(n_hours):
    #    for s in range(n_scenarios):
    #        m.addConstr(Delta_up[t, s]   <= M * b[t, s])
    #        m.addConstr(Delta_down[t, s] <= M * (1 - b[t, s]))

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
    Delta_up_mat = np.array([[Delta_up[t, s].X for s in range(n_scenarios)] for t in range(n_hours)])
    Delta_down_mat = np.array([[Delta_down[t, s].X for s in range(n_scenarios)] for t in range(n_hours)])
    profit_matrix = (lambda_DA * p_DA_vec[:, None] + lambda_bal_up * Delta_up_mat - lambda_bal_down * Delta_down_mat)

    return m, p_DA_vec, Delta_up, Delta_down, profit_matrix

def plot_DA_offers(p_DA, in_sample_scenarios, title="Day-ahead offers", Threshold_value=None):
    """
    Visualize day-ahead offers compared to forecasted wind production and deficit probability.
    
    Parameters:
    -----------
    p_DA : np.ndarray
        Array of shape (24,) containing day-ahead offers per hour in MWh.
    in_sample_scenarios : np.ndarray
        Array of shape (24, n_scenarios, 3) containing sampled scenarios where:
        - 24 = hours per day
        - n_scenarios = number of scenarios
        - 3 channels: [wind production (MWh), day-ahead price (DKK/MWh), surplus/deficit binary]
    title : str, optional
        Title for the plot, default "Day-ahead offers".
    Threshold_value : float, optional
        Optional threshold value to display on the deficit probability subplot (0-1), default None.
        
    Returns:
    --------
    None
        Displays a two-panel plot with day-ahead offers and average wind production (top panel) 
        and deficit probability prediction (bottom panel).
    """
    # Average forecasted wind production per hour (24-hour profile)
    avg_forecasted_wind_per_hour = in_sample_scenarios[:, :, 0].mean(axis=1)
    df_avg_forecasted_wind = pd.DataFrame({
        "hour": np.arange(24),
        "avg_forecasted_wind_mwh": avg_forecasted_wind_per_hour
    })

    avg_deficit_prediction_per_hour = in_sample_scenarios[:, :, 2].mean(axis=1)
    df_avg_deficit_prediction = pd.DataFrame({
        "hour": np.arange(24),
        "avg_deficit_prediction": avg_deficit_prediction_per_hour
    })

    hours = np.arange(24)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 5),
        gridspec_kw={"height_ratios": [2.5, 2]}
    )

    # Top subplot: offering + avg forecasted wind
    ax1.step(hours, p_DA, where="post", marker="o", linewidth=2, label="Offering capacity")
    ax1.step(
        hours,
        avg_forecasted_wind_per_hour,
        where="post",
        marker="s",
        linewidth=2,
        label="Avg. forecasted wind production"
    )
    ax1.set_title(title, fontsize=16)
    ax1.set_xlabel("Hour", fontsize=12)
    ax1.set_ylabel("Power (MW)", fontsize=12)
    ax1.set_xticks(np.arange(24))
    ax1.set_xlim(0, 23)
    ax1.tick_params(labelbottom=True)
    ax1.set_ylim(0, 550)
    ax1.grid(True, alpha=0.3)
    #ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    # Bottom subplot: avg deficit prediction
    ax2.step(
        hours,
        avg_deficit_prediction_per_hour,
        where="post",
        color="tab:red",
        marker="o",
        linewidth=2,
        label="Avg. deficit prediction"
    )
    ax2.set_xlabel("Hour", fontsize=12)
    ax2.set_ylabel("Deficit probability", fontsize=12)
    ax2.set_xticks(np.arange(24))
    ax2.set_xlim(0, 23)
    ax2.set_ylim(0, 1.05)
    if Threshold_value is not None:
        ax2.axhline(Threshold_value, color="black", linestyle="--", linewidth=1, label=f"P_deficit = {Threshold_value}")
    ax2.grid(True, alpha=0.3)
    #ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)


        # Collect handles and labels from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine
    handles = handles1 + handles2
    labels = labels1 + labels2

    # Create one legend for the whole figure
    fig.legend(handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.1),
            ncol=4,
            fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    plt.show()



def plot_profit_distribution(profit_per_scenario, n_bins = 15, title="Profit distribution per scenario", x_label="Total profit (MDKK)"):
    """
    Visualize the distribution of profits across scenarios using a histogram.
    
    Parameters:
    -----------
    profit_per_scenario : np.ndarray
        Array containing profit values for each scenario.
    n_bins : int, optional
        Number of bins for the histogram, default 15.
    title : str, optional
        Title for the plot, default "Profit distribution per scenario".
    x_label : str, optional
        Label for the x-axis, default "Total profit (MDKK)".
        
    Returns:
    --------
    None
        Displays a histogram of profit distribution with a vertical line indicating the mean profit.
    """
    plt.figure(figsize=(10,6))
    plt.hist(profit_per_scenario, bins=n_bins)
    plt.axvline(profit_per_scenario.mean(), color='red', linestyle='dashed', label=f"Mean: {profit_per_scenario.mean():.2f} MDKK")
    plt.title(title, fontsize=22)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.show()

def plot_cumulative_profit_distribution(profit_per_scenario, title="Cumulative profit distribution", tail_fraction=1.0, mean=True
):
    """
    Visualize the cumulative distribution function (CDF) of profits across scenarios.
    
    Parameters:
    -----------
    profit_per_scenario : np.ndarray
        Array containing profit values for each scenario.
    title : str, optional
        Title for the plot, default "Cumulative profit distribution".
    tail_fraction : float, optional
        Fraction of lowest profits to display (0-1), default 1.0 (shows all scenarios).
        Use values < 1.0 to focus on the worst-case scenarios.
    mean : bool, optional
        If True (default), display a vertical line at the mean profit across all scenarios.
        
    Returns:
    --------
    None
        Displays a step plot of the empirical CDF with optional tail focus and mean line.
    """

    plt.figure(figsize=(8, 6))

    # Sort profits
    sorted_profit = np.sort(np.asarray(profit_per_scenario))
    n = sorted_profit.size

    # --- Select lowest fraction
    k = int(np.ceil(tail_fraction * n))
    subset_profit = sorted_profit[:k]

    # Recompute CDF on subset
    cdf = np.arange(1, k + 1) / k
    if tail_fraction < 1.0:
        label = f"Lowest {tail_fraction*100:.0f}%"
    else:
        label = "Empirical CDF"

    plt.step(subset_profit, cdf, where="post", linewidth=2,
             label=label)

    # Mean of full distribution (optional, usually more meaningful)
    if mean:
        mean_profit = sorted_profit.mean()
        plt.axvline(
            mean_profit,
            color="red",
            linestyle="dashed",
            label=f"Mean (all): {mean_profit:.2f} MDKK"
        )

    plt.title(title, fontsize=18)
    plt.xlabel("Total profit per scenario (MDKK)", fontsize=16)
    plt.ylabel("Probability", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16)
    plt.show()

def plot_cdf_comparison(
    profit_a,
    profit_b,
    label_a="Strategy A",
    label_b="Strategy B",
    title="CDF comparison (CVaR view)",
    alpha=0.9,
    ls_second="-",
    mean=False
):
    """
    Compare cumulative distribution functions (CDFs) of profits from two strategies, 
    focusing on the tail.
    
    Parameters:
    -----------
    profit_a : np.ndarray
        Array of profit values from the first strategy (baseline).
    profit_b : np.ndarray
        Array of profit values from the second strategy for comparison.
    label_a : str, optional
        Label for the first strategy in the legend, default "Strategy A".
    label_b : str, optional
        Label for the second strategy in the legend, default "Strategy B".
    title : str, optional
        Title for the plot, default "CDF comparison (CVaR view)".
    alpha : float, optional
        Confidence level (0-1) for the CVaR tail region, default 0.9. Only the worst (1-alpha) 
        fraction of scenarios are displayed to emphasize tail behavior.
    ls_second : str, optional
        Line style for the second strategy's CDF line (matplotlib line style), default "-" (solid).
    mean : bool, optional
        If True (default False), display a vertical line at the mean profit of the first strategy.
        
    Returns:
    --------
    None
        Displays a two-strategy CDF comparison plot with the tail region (worst-case scenarios) emphasized.
    """

    plt.figure(figsize=(8, 6))

    def prepare(profit):
        sorted_profit = np.sort(np.asarray(profit))
        n = len(sorted_profit)
        cdf = np.arange(1, n + 1) / n
        var = np.quantile(sorted_profit, 1-alpha)
        return sorted_profit, cdf, var

    # --- Prepare both strategies
    x_a, cdf_a, var_a = prepare(profit_a)
    x_b, cdf_b, var_b = prepare(profit_b)

    # --- Plot full CDFs
    plt.step(x_a, cdf_a, where="post", linewidth=2, label=label_a)
    plt.step(x_b, cdf_b, where="post", linewidth=2, label=label_b, linestyle=ls_second)

    if mean:
        mean_profit = x_a.mean()
        plt.axvline(
            mean_profit,
            color="red",
            linestyle="dashed",
            label=f"Mean {label_a}: {mean_profit:.2f} MDKK"
        )

    # --- Only show tail
    plt.ylim(0, 1-alpha+0.01)
    plt.xlim(min(x_a.min(), x_b.min()), max(var_a, var_b))

    # --- Labels
    plt.title(title, fontsize=18)
    plt.xlabel("Total profit per scenario (MDKK)", fontsize=14)
    plt.ylabel("Cumulative probability", fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()


def scenario_profit_stats(profit_matrix):
    """
    Compute the minimum and maximum total profit across all scenarios.
    
    Parameters:
    -----------
    profit_matrix : np.ndarray
        Array of shape (n_hours, n_scenarios) containing profit per hour and scenario.
        The values are summed over hours to obtain total profit per scenario.
        
    Returns:
    --------
    tuple of float
        A pair containing the minimum and maximum total profit across all scenarios.
    """
    profit_per_scenario = profit_matrix.sum(axis=0)
    return profit_per_scenario.min(), profit_per_scenario.max()



def plot_profit_distribution_comparison(profit_per_scenario, profit_per_scenario_2, n_bins=15):

    """
    Compare the distribution of scenario profits for two strategies using overlaid histograms.
    
    Parameters:
    -----------
    profit_per_scenario : np.ndarray
        Array of total profit values for the first strategy.
    profit_per_scenario_2 : np.ndarray
        Array of total profit values for the second strategy.
    n_bins : int, optional
        Number of bins to use for the shared histogram range, default 15.
        
    Returns:
    --------
    None
        Displays a histogram comparison with mean profit markers for both strategies.
    """

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
    """
    Split scenario data into shuffled folds for cross-validation.
    
    Parameters:
    -----------
    scenarios : np.ndarray
        Array of shape (n_hours, n_samples, n_features) containing the full scenario set.
    n_in_sample : int
        Number of samples to include in each fold.
    seed : int, optional
        Seed for the random number generator used to shuffle scenario indices, default 42.
        
    Returns:
    --------
    list of np.ndarray
        List of folds, where each fold has shape (n_hours, n_in_sample, n_features).
    """
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

def cross_validate_folds(folds, two_price=False, silent=False):
    """
    Run cross-validation over a list of folds and collect in-sample and out-of-sample profits.
    
    Parameters:
    -----------
    folds : list of np.ndarray
        List of scenario folds, where each fold has shape (n_hours, n_samples, 3).
    two_price : bool, optional
        If True, solve the two-price stochastic strategy for each fold.
        If False (default), solve the one-price stochastic strategy.
    silent : bool, optional
        If False (default), print fold-by-fold evaluation results.
        
    Returns:
    --------
    tuple of list of float
        Two lists containing the mean in-sample profit and mean out-of-sample profit for each fold.
    """

    in_sample_means = []
    out_sample_means = []

    for i in range(len(folds)):

        # in-sample
        fold = folds[i]

        # out-of-sample
        out_of_sample = np.concatenate(
            [folds[j] for j in range(len(folds)) if j != i],
            axis=1
        )

        # solve model
        if two_price:
            _, p_DA_vec, _, _, profit_matrix_fold = solve_stochastic_strategy_two_price(fold, silent=True)
        else:
            _, p_DA_vec, _, profit_matrix_fold = solve_stochastic_strategy_one_price(fold, silent=True)

        # in-sample evaluation
        in_sample_profit = profit_matrix_fold.sum(axis=0)

        # out-of-sample evaluation
        profit_out = calculate_profit(fold, out_of_sample, p_DA_vec, two_price=two_price)
        
        # store
        in_sample_means.append(in_sample_profit.mean())
        out_sample_means.append(profit_out.mean())

        if silent == False:
            print(f"Fold {i+1}: in={in_sample_profit.mean():.3f}, out={profit_out.mean():.3f}")
            print("DA offers: ", p_DA_vec)
            print("deficit probabilities: ", fold[:, :, 2].mean(axis=1))

    return in_sample_means, out_sample_means, 


def calculate_profit(scenarios_in, scenarios_out, p_DA_vec, two_price=False):
    """
    Calculate out-of-sample profits for a fixed day-ahead bid vector.
    
    Parameters:
    -----------
    scenarios_in : np.ndarray
        Array of shape (n_hours, n_scenarios, 3) containing the in-sample scenarios used to
        compute the mean day-ahead price per hour.
    scenarios_out : np.ndarray
        Array of shape (n_hours, n_scenarios, 3) containing the out-of-sample scenarios used
        to evaluate the bid.
    p_DA_vec : np.ndarray
        Array of shape (n_hours,) containing the fixed day-ahead offer per hour in MWh.
    two_price : bool, optional
        If True, use the two-price balancing scheme. If False (default), use the one-price scheme.
        
    Returns:
    --------
    np.ndarray
        Array of total profit values per out-of-sample scenario.
    """

    # scenarios shape: (n_hours, n_scenarios, 3)
    p_real_out = scenarios_out[:, :, 0]
    lambda_DA_out = scenarios_out[:, :, 1]
    deficit_bin_out = scenarios_out[:, :, 2]

    # use in-sample scenarios for DA profit
    lambda_DA_in = scenarios_in[:, :, 1]

    # mean DA price per hour across in-sample scenarios
    lambda_mean = lambda_DA_in.mean(axis=1)   # shape: (24,)

    DA_profit = p_DA_vec * lambda_mean # shape: (24,)

    if two_price:
        # Balancing price
        lambda_bal_up   = deficit_bin_out * lambda_DA_out + (1 - deficit_bin_out) * 0.85 * lambda_DA_out
        lambda_bal_down = deficit_bin_out * 1.25 * lambda_DA_out + (1 - deficit_bin_out) * lambda_DA_out
        Delta = p_real_out - p_DA_vec[:, None]

        Delta_up = np.maximum(Delta, 0)
        Delta_down = np.maximum(-Delta, 0)

        profit_matrix = (
            DA_profit[:, None]
            + lambda_bal_up * Delta_up
            - lambda_bal_down * Delta_down
        )

        profit_out = profit_matrix.sum(axis=0)

        return profit_out

    else:
        # Balancing price
        lambda_bal = 1.25 * lambda_DA_out * deficit_bin_out + 0.85 * lambda_DA_out * (1 - deficit_bin_out)

        # Compute Delta = p_real - p_DA
        Delta = p_real_out - p_DA_vec[:, None]

        # Profit per hour & scenario
        profit_matrix = DA_profit[:, None] + lambda_bal * Delta

        # Return total profit per scenario
        profit_out = profit_matrix.sum(axis=0)

        return profit_out
    

def plot_Cross_Validation_Profits(in_sample_means, out_sample_means, title="Expected profits for each fold"):
    """
    Visualize in-sample and out-of-sample mean profits for each cross-validation fold.
    
    Parameters:
    -----------
    in_sample_means : list of float
        Mean in-sample profit for each fold.
    out_sample_means : list of float
        Mean out-of-sample profit for each fold.
    title : str, optional
        Title for the plot, default "Expected profits for each fold".
        
    Returns:
    --------
    None
        Displays a grouped bar chart with average in-sample and out-of-sample profit lines.
    """
    folds = np.arange(1, len(in_sample_means) + 1)
    x = np.arange(len(folds))
    bar_width = 0.4
    in_mean = np.mean(in_sample_means)
    out_mean = np.mean(out_sample_means)

    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_width / 2, in_sample_means, width=bar_width, label="In-sample mean profit")
    plt.bar(x + bar_width / 2, out_sample_means, width=bar_width, label="Out-of-sample mean profit")
    plt.axhline(in_mean, linestyle="--", linewidth=1.5, color="tab:blue", label=f"Avg in-sample: {in_mean:.3f}")
    plt.axhline(out_mean, linestyle="--", linewidth=1.5, color="tab:orange", label=f"Avg out-of-sample: {out_mean:.3f}")
    plt.title(title, fontsize=18)
    plt.xlabel("Fold", fontsize=14)
    plt.ylabel("Mean profit (MDKK)", fontsize=14)
    plt.xticks(x, folds)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    plt.show()

def solve_risk_averse_one_price(in_sample_scenarios, alpha=0.9, beta=0, silent=False):
    """
    Solve the risk-averse stochastic day-ahead bidding problem with single balancing price.
    
    Parameters:
    -----------
    in_sample_scenarios : np.ndarray
        Array of shape (n_hours, n_scenarios, 3) containing sampled scenarios where:
        - n_hours = 24 (hourly offers over one day)
        - n_scenarios = number of Monte Carlo scenarios
        - 3 channels: [wind production (MWh), day-ahead price (DKK/MWh), surplus/deficit binary]
    alpha : float, optional
        Confidence level used in the CVaR term, default 0.9.
    beta : float, optional
        Risk-aversion weight between expected profit and CVaR, default 0.
        Higher values put more emphasis on tail performance.
    silent : bool, optional
        If False (default), print computational details (runtime, variables, constraints).
        
    Returns:
    --------
    m : gurobipy.Model
        The solved Gurobi optimization model.
    p_DA_vec : np.ndarray
        Array of shape (24,) containing optimal day-ahead offer per hour in MWh.
    Delta : gurobipy.tupledict
        Variable dictionary of shape (24, n_scenarios) representing the difference between
        real production and day-ahead offer for each hour and scenario.
    profit_matrix : np.ndarray
        Array of shape (24, n_scenarios) containing profit per hour and scenario in MDKK.
    cvar : float
        Optimized CVaR value of the total profit distribution.
    eta_vec : np.ndarray
        Array of shape (n_scenarios,) containing the CVaR auxiliary slack variables.
    """
    
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
    alpha = alpha # used quantile
    beta = beta # balance between risk-neutral and risk-averse

    # Variables
    p_DA = m.addVars(n_hours, lb=0, name="DayAhead offer")
    Delta = m.addVars(n_hours, n_scenarios, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Difference_DA_real")
    zeta = m.addVar(lb=-gp.GRB.INFINITY, name = "VaR")
    eta = m.addVars(n_scenarios, lb=0, name = "auxillary_helper_variable")

    # Objective
    m.setObjective((1-beta) * gp.quicksum(prob_scenarios * 
                                (lambda_DA[t, s] * p_DA[t]
                                 + lambda_bal[t, s] * Delta[t, s])
                                for t in range(n_hours) for s in range(n_scenarios))
                    + beta * (zeta - 1/(1-alpha) * gp.quicksum(prob_scenarios * eta[s] for s in range(n_scenarios))), gp.GRB.MAXIMIZE)


    # Capacity limit constraint
    for t in range(n_hours):
        m.addConstr(p_DA[t] <= P_nom, name=f"Capacity_limit_{t}")

    # Constraints setting difference between DA and real production
    for t in range(n_hours):
        for s in range(n_scenarios):
            m.addConstr(Delta[t, s] == p_real[t, s] - p_DA[t], name=f"Difference_DA_real_{t}_{s}")

    # eta constraints
    for s in range(n_scenarios):
        m.addConstr(-gp.quicksum(lambda_DA[t, s] * p_DA[t] + lambda_bal[t, s] * Delta[t, s] for t in range(n_hours)) + zeta - eta[s] <= 0)

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

    zeta_val = zeta.X
    eta_vec = np.array([eta[s].X for s in range(n_scenarios)])
    cvar = zeta_val - (1/(1-alpha)) * prob_scenarios * eta_vec.sum()

    return m, p_DA_vec, Delta, profit_matrix, cvar, eta_vec


def solve_risk_averse_two_price(in_sample_scenarios, alpha=0.9, beta=0, silent=False):
    """
    Solve the risk-averse stochastic day-ahead bidding problem with two-price balancing.
    
    Parameters:
    -----------
    in_sample_scenarios : np.ndarray
        Array of shape (n_hours, n_scenarios, 3) containing sampled scenarios where:
        - n_hours = 24 (hourly offers over one day)
        - n_scenarios = number of Monte Carlo scenarios
        - 3 channels: [wind production (MWh), day-ahead price (DKK/MWh), surplus/deficit binary]
    alpha : float, optional
        Confidence level used in the CVaR term, default 0.9.
    beta : float, optional
        Risk-aversion weight between expected profit and CVaR, default 0.
        Higher values put more emphasis on tail performance.
    silent : bool, optional
        If False (default), print computational details (runtime, variables, constraints).
        
    Returns:
    --------
    m : gurobipy.Model
        The solved Gurobi optimization model.
    p_DA_vec : np.ndarray
        Array of shape (24,) containing optimal day-ahead offer per hour in MWh.
    profit_matrix : np.ndarray
        Array of shape (24, n_scenarios) containing profit per hour and scenario in MDKK.
    cvar : float
        Optimized CVaR value of the total profit distribution.
    eta_vec : np.ndarray
        Array of shape (n_scenarios,) containing the CVaR auxiliary slack variables.
    """

    # in_sample_scenarios has shape (n_hours, n_scenarios, 3)
    p_real = in_sample_scenarios[:,:,0]
    lambda_DA = in_sample_scenarios[:,:,1]
    deficit_bin = in_sample_scenarios[:,:,2]

    m = gp.Model("risk_averse_two_price")
    m.Params.OutputFlag = 0

    # Parameters
    P_nom = 500
    n_hours = len(in_sample_scenarios[:, 0, 0])
    n_scenarios = len(in_sample_scenarios[0, :, 0])
    prob_scenarios = 1 / n_scenarios
    lambda_bal_up   = deficit_bin * lambda_DA + (1 - deficit_bin) * 0.85 * lambda_DA
    lambda_bal_down = deficit_bin * 1.25 * lambda_DA + (1 - deficit_bin) * lambda_DA

    # Variables
    p_DA = m.addVars(n_hours, lb=0, name="DayAhead offer")
    Delta_up = m.addVars(n_hours, n_scenarios, lb=0, name="difference over 0")
    Delta_down = m.addVars(n_hours, n_scenarios, lb=0, name="difference under 0")
    zeta = m.addVar(lb=-gp.GRB.INFINITY, name="VaR")
    eta = m.addVars(n_scenarios, lb=0, name="auxiliary_helper_variable")

    # Objective
    m.setObjective(
        (1 - beta) * gp.quicksum(
            prob_scenarios * (
                lambda_DA[t, s] * p_DA[t]
                + lambda_bal_up[t, s]   * Delta_up[t, s]
                - lambda_bal_down[t, s] * Delta_down[t, s])
            for t in range(n_hours) for s in range(n_scenarios))
        + beta * (zeta - 1/(1-alpha) * gp.quicksum(prob_scenarios * eta[s] for s in range(n_scenarios))),
        gp.GRB.MAXIMIZE)

    # Capacity limit constraint
    for t in range(n_hours):
        m.addConstr(p_DA[t] <= P_nom, name=f"Capacity_limit_{t}")

    # Delta split constraints
    for t in range(n_hours):
        for s in range(n_scenarios):
            m.addConstr(Delta_up[t, s] - Delta_down[t, s] == p_real[t, s] - p_DA[t], name=f"Difference_DA_real_{t}_{s}")

    # CVaR eta constraints: eta[s] >= zeta - profit[s]
    for s in range(n_scenarios):
        m.addConstr(
            -gp.quicksum(
                lambda_DA[t, s] * p_DA[t]
                + lambda_bal_up[t, s]   * Delta_up[t, s]
                - lambda_bal_down[t, s] * Delta_down[t, s]
                for t in range(n_hours))
            + zeta - eta[s] <= 0,
            name=f"CVaR_eta_{s}")

    # Optimize
    m.optimize()

    if not silent:
        print(f"  Computational time:    {m.Runtime:.6f} s")
        print(f"  Decision variables:    {int(m.NumVars)}")
        print(f"  Constraints:           {int(m.NumConstrs)}")

    # Extract solution
    p_DA_vec = np.array([p_DA[t].X for t in range(n_hours)])
    Delta_up_mat   = np.array([[Delta_up[t, s].X   for s in range(n_scenarios)] for t in range(n_hours)])
    Delta_down_mat = np.array([[Delta_down[t, s].X for s in range(n_scenarios)] for t in range(n_hours)])
    profit_matrix  = lambda_DA * p_DA_vec[:, None] + lambda_bal_up * Delta_up_mat - lambda_bal_down * Delta_down_mat

    zeta_val = zeta.X
    eta_vec  = np.array([eta[s].X for s in range(n_scenarios)])
    cvar = zeta_val - (1/(1-alpha)) * prob_scenarios * eta_vec.sum()

    return m, p_DA_vec, profit_matrix, cvar, eta_vec


def plot_profit_cvar_tradeoff(cvar_list, exp_profit, beta_range, annotate=True):
    """
    Visualize the tradeoff between CVaR and expected profit across risk-aversion weights.
    
    Parameters:
    -----------
    cvar_list : list of float
        CVaR values corresponding to each tested beta value.
    exp_profit : list of float
        Expected profit values corresponding to each tested beta value.
    beta_range : iterable of float
        Sequence of beta values used to generate the tradeoff points.
    annotate : bool, optional
        If True (default), annotate every third point with its beta value.
        
    Returns:
    --------
    None
        Displays a line plot of CVaR versus expected profit.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(cvar_list, exp_profit, marker='o', linewidth=2)

    if annotate:
        for i, b in enumerate(beta_range):
            if i % 3 == 0:
                plt.annotate(f"β={b:.2f}", (cvar_list[i], exp_profit[i]),
                             textcoords="offset points", xytext=(6, 4), fontsize=12)

    plt.xlabel("CVaR (MDKK)", fontsize=16)
    plt.ylabel("Expected Profit (MDKK)", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compute_profit_cvar_tradeoff(in_sample_scenarios, beta_range, alpha=0.9, scheme="two_price"):
    """
    Compute expected profit and CVaR values for a range of risk-aversion weights.
    
    Parameters:
    -----------
    in_sample_scenarios : np.ndarray
        Array of shape (n_hours, n_scenarios, 3) containing the sampled in-sample scenarios.
    beta_range : iterable of float
        Sequence of beta values used to solve the risk-averse optimization problems.
    alpha : float, optional
        Confidence level used in the CVaR term, default 0.9.
    scheme : str, optional
        Risk model to use, either "two_price" (default) or "one_price".
        
    Returns:
    --------
    tuple of list
        Three lists containing the expected profit, CVaR, and optimal day-ahead offer vector for each beta.
    """
    exp_profit = []
    cvar_list = []
    p_DA_all = []
    
    for b in beta_range:
        if scheme == "two_price":
            _, p_DA_vec, profit_matrix, cvar, _ = solve_risk_averse_two_price(
                in_sample_scenarios, alpha=alpha, beta=b, silent=False)

        elif scheme == "one_price":
            _, p_DA_vec, _, profit_matrix, cvar, _ = solve_risk_averse_one_price(
                in_sample_scenarios, alpha=alpha, beta=b, silent=False)

        profit_per_scenario = profit_matrix.sum(axis=0)
        expected_profit = profit_per_scenario.mean()

        exp_profit.append(expected_profit)
        cvar_list.append(cvar)
        p_DA_all.append(p_DA_vec)

        print(f"beta={b:.2f}: E[profit]={expected_profit:.3f}, CVaR={cvar:.3f}")

    return exp_profit, cvar_list, p_DA_all


def plot_DA_offers_risk(beta_range, p_DA_list, tol=1e-6):
    """
    Visualize day-ahead offers across different risk-aversion weights and group identical strategies.
    
    Parameters:
    -----------
    beta_range : iterable of float
        Sequence of beta values corresponding to the risk-aversion levels used to generate the offers.
    p_DA_list : list of np.ndarray
        List of day-ahead offer vectors, where each array has shape (24,) and corresponds to one beta value.
    tol : float, optional
        Absolute tolerance used to decide whether two offer vectors are identical, default 1e-6.
        
    Returns:
    --------
    None
        Displays a step plot of day-ahead offers, with identical strategies grouped and labeled together.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    hours = np.arange(24)
    plt.figure(figsize=(8, 4))

    # --- Group identical full-day strategies
    unique_strategies = []
    group_indices = []

    for i, p in enumerate(p_DA_list):
        found = False
        for j, ref in enumerate(unique_strategies):
            if np.allclose(p, ref, atol=tol):
                group_indices[j].append(i)
                found = True
                break
        if not found:
            unique_strategies.append(p)
            group_indices.append([i])

    # --- Assign colors per unique strategy
    cmap = plt.cm.get_cmap("tab10", len(unique_strategies))
    distinct_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for g_idx, indices in enumerate(group_indices):
        color = distinct_colors[g_idx % len(distinct_colors)]
        
        for k, i in enumerate(indices):
            b = beta_range[i]


            if len(indices) > 1:
                label = f"β ∈ [{beta_range[indices[0]]:.2f}, {beta_range[indices[-1]]:.2f}]" if k == 0 else None
            else:
                label = f"β={b:.2f}" if k == 0 else None

            plt.step(
                hours,
                p_DA_list[i],
                where="post",
                #marker="o",
                linewidth=2 + 1*len(indices),
                alpha=0.7,
                color=color,
                # Only label once per group
                label=label
            )

    # --- Styling (match your original)
    plt.xlabel("Hour", fontsize=16)
    plt.ylabel("DA Offer (MW)", fontsize=16)
    #plt.title("Day-ahead offers for different β", fontsize=)

    plt.xticks(np.arange(24), fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0, 23)
    plt.ylim(0, 550)

    plt.grid(True, alpha=0.3)
    plt.legend(ncol=1, fontsize=16)

    plt.tight_layout()
    plt.show()



def evaluate_fixed_bid_risk(in_sample_scenarios, p_DA_input, alpha=0.9):
    """
    Evaluate the profit distribution and CVaR statistics for a fixed day-ahead bid vector.
    
    Parameters:
    -----------
    in_sample_scenarios : np.ndarray
        Array of shape (n_hours, n_scenarios, 3) containing sampled scenarios where:
        - n_hours = 24 (hourly offers over one day)
        - n_scenarios = number of Monte Carlo scenarios
        - 3 channels: [wind production (MWh), day-ahead price (DKK/MWh), surplus/deficit binary]
    p_DA_input : np.ndarray
        Array of shape (n_hours,) containing the fixed day-ahead offer per hour in MWh.
    alpha : float, optional
        Confidence level used to define the CVaR tail, default 0.9.
        
    Returns:
    --------
    dict
        Dictionary containing the CVaR value, VaR threshold, worst-scenario indices,
        total profit per scenario, and the corresponding eta values.
    """
    n_hours, n_scenarios, _ = in_sample_scenarios.shape
    p_real = in_sample_scenarios[:, :, 0]
    lambda_DA = in_sample_scenarios[:, :, 1]
    deficit_bin = in_sample_scenarios[:, :, 2]
    
    # One-price balancing logic
    lambda_bal = 1.25 * lambda_DA * deficit_bin + 0.85 * lambda_DA * (1 - deficit_bin)
    
    # Calculate profit matrix and daily totals
    # p_DA_input[:, None] broadcasts the 24h bids across all scenarios
    profit_matrix = (lambda_DA * p_DA_input[:, None]) + (lambda_bal * (p_real - p_DA_input[:, None]))
    daily_profits = np.sum(profit_matrix, axis=0)
    
    # Identify the Tail (Worst Scenarios)
    # CVaR is the mean of the worst (1-alpha) fraction of scenarios
    n_tail = int(np.ceil(n_scenarios * (1 - alpha)))
    sorted_indices = np.argsort(daily_profits)
    worst_indices = sorted_indices[:n_tail]
    
    # Calculate VaR (the profit of the best scenario in the worst group)
    zeta_val = daily_profits[sorted_indices[n_tail - 1]]
    
    # Calculate CVaR
    cvar_val = np.mean(daily_profits[worst_indices])
    
    # Calculate eta (how much each scenario falls below VaR)
    # Matches Gurobi: eta[s] = max(0, zeta - daily_profit[s])
    eta_vec = np.maximum(0, zeta_val - daily_profits)
    
    return {
        "cvar": cvar_val,
        "zeta": zeta_val,
        "worst_indices": worst_indices,
        "daily_profits": daily_profits,
        "eta_vec": eta_vec
    }


# %%
def compute_DA_offer_samples(
    n_runs=50,
    n_in_sample=200,
    alpha=0.9,
    beta=1,
    n_wind=20,
    n_price=20,
    n_surp_def=4,
    seed=42
):
    """
    Generate multiple risk-averse day-ahead offer solutions from repeated scenario sampling.
    
    Parameters:
    -----------
    n_runs : int, optional
        Number of repeated sampling and optimization runs to perform, default 50.
    n_in_sample : int, optional
        Number of scenarios to draw for each in-sample optimization problem, default 200.
    alpha : float, optional
        Confidence level used in the CVaR term, default 0.9.
    beta : float, optional
        Risk-aversion weight passed to the optimization model, default 1.
    n_wind : int, optional
        Number of wind sections used when generating scenarios, default 20.
    n_price : int, optional
        Number of price sections used when generating scenarios, default 20.
    n_surp_def : int, optional
        Number of surplus/deficit scenarios used when generating scenarios, default 4.
    seed : int, optional
        Seed for the master random number generator, default 42.
        
    Returns:
    --------
    tuple of np.ndarray
        Three arrays containing the sampled day-ahead offers, average hourly profits,
        and CVaR values for each run.
    """
    rng_master = np.random.default_rng(seed)
    p_DA_list = []
    profit_list = []
    cvar_list = []

    for run in range(n_runs):
        rng = np.random.default_rng(rng_master.integers(0, 1e9))

        # Generate scenarios
        scenarios = generate_scenarios(
            random_state=rng,
            n_wind=n_wind,
            n_price=n_price,
            n_surp_def=n_surp_def
        )

        # Sample in-sample set
        idx = rng.choice(scenarios.shape[1], size=n_in_sample, replace=False)
        in_sample = scenarios[:, idx, :]

        # Solve
        _, p_DA, profit, cvar, _ = solve_risk_averse_two_price(
            in_sample, alpha=alpha, beta=beta, silent=True
        )

        p_DA_list.append(p_DA)
        profit_list.append(profit)
        cvar_list.append(cvar)

        # Progress print every 10 runs
        if (run + 1) % 10 == 0 or run == 0:
            print(f"Computed {run + 1}/{n_runs} DA offers")

    return np.array(p_DA_list), np.array(profit_list).mean(axis=2), np.array(cvar_list)

def plot_DA_offer_folds(folds):
    """
    Visualize the day-ahead offers obtained for each cross-validation fold.
    
    Parameters:
    -----------
    folds : list of np.ndarray
        List of scenario folds, where each fold has shape (n_hours, n_samples, 3).
        Each fold is passed to the risk-averse optimization model to obtain one day-ahead offer vector.
        
    Returns:
    --------
    None
        Displays a step plot of the day-ahead offer curve for each fold.
    """
    
    hours = np.arange(24)

    plt.figure(figsize=(12, 6))
    # Loop over all folds
    import itertools

    linestyles = itertools.cycle(['-', '--', '-.', ':'])

    for i, fold in enumerate(folds):
        _, p_DA_vec_fold, _, _, _ = solve_risk_averse_two_price(
            fold, alpha=0.9, beta=1, silent=True
        )

        plt.step(
            hours,
            p_DA_vec_fold,
            where='post',
            linestyle=next(linestyles),
            label=f"Fold {i+1}",
            linewidth=2.5,
            alpha=0.8
        )

    plt.xticks(hours)
    plt.xlim(0, 23)
    plt.xlabel("Hour", fontsize=16)
    plt.ylabel("DA Offer (MW)", fontsize=16)
    #plt.title("DA Offers for Beta=1 Across All Folds", fontsize=20)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.show()



def plot_scatter_profit_cvar(profit_list, cvar_list, base_profit, base_cvar):
    """
    Visualize the relationship between expected profit and CVaR using a scatter plot.
    
    Parameters:
    -----------
    profit_list : list of float
        List of expected profit values corresponding to the sampled risk-aversion settings.
    cvar_list : list of float
        List of CVaR values corresponding to the same sampled risk-aversion settings.
    base_profit : float
        Expected profit of the baseline strategy, shown as a highlighted marker.
    base_cvar : float
        CVaR value of the baseline strategy, shown as a highlighted marker.
        
    Returns:
    --------
    None
        Displays a scatter plot with a fitted regression line and a highlighted base-case point.
    """
    plt.figure(figsize=(8, 6))

    coef = np.polyfit(cvar_list, profit_list, 1)
    slope, intercept = coef

    x = np.linspace(0, max(cvar_list), 100)
    y = slope * x + intercept

    y_pred = slope * np.array(cvar_list) + intercept
    ss_res = np.sum((np.array(profit_list) - y_pred)**2)
    ss_tot = np.sum((np.array(profit_list) - np.mean(profit_list))**2)
    r2 = 1 - ss_res / ss_tot

    # Scatter samples
    plt.scatter(cvar_list, profit_list, alpha=0.7)

    # Regression line
    plt.plot(x, y, color='black',
             label=f"Fit: y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r2:.3f}", linestyle="--")

    # Base case point
    plt.scatter(base_cvar, base_profit,
                color='red', s=100, marker='X', label="Base case", zorder=5)
    plt.xlabel("CVaR [MDKK]", fontsize=16)
    plt.ylabel("Expected Profit [MDKK]", fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16)
    plt.xlim(left=0)
    plt.ylim(bottom=min(profit_list)-1)
    plt.show()



def plot_boxplot_profit_cvar(profit_list, cvar_list, base_profit, base_cvar):
    """
    Compare the distributions of expected profit and CVaR using side-by-side boxplots.
    
    Parameters:
    -----------
    profit_list : list of float
        List of expected profit values corresponding to the sampled risk-aversion settings.
    cvar_list : list of float
        List of CVaR values corresponding to the same sampled risk-aversion settings.
    base_profit : float
        Expected profit of the baseline strategy, shown as a highlighted marker.
    base_cvar : float
        CVaR value of the baseline strategy, shown as a highlighted marker.
        
    Returns:
    --------
    None
        Displays a boxplot comparing the spread of expected profit and CVaR values.
    """
    plt.figure(figsize=(8, 6))

    data = [profit_list, cvar_list]

    box = plt.boxplot(
        data,
        patch_artist=True,
        labels=["Expected Profit", "CVaR"]
    )

    colors = ["#b2b2b2", "#b2b2b2"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    # Base case markers
    plt.scatter(1, base_profit, color="red", s=100, marker='X', zorder=5, linewidths=0.5, label="Base case")
    plt.scatter(2, base_cvar, color='red', s=100, marker='X', linewidths=0.5, zorder=5)

    plt.ylabel("MDKK", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(fontsize=16)
    plt.show()


def Load_profile_generation(random_state=None, Profiles=300, P_max=600, P_min=220, P_delta=35, plot=False):
    """
    Generate random load profiles for a single hour at minute-level resolution.
    
    Parameters:
    -----------
    random_state : int or np.random.Generator, optional
        Seed for the random number generator
    Profiles : int
        Number of load profiles to generate, default 300
    P_max : float
        Maximum power consumption (kW), default 600
    P_min : float
        Minimum power consumption (kW), default 220
    P_delta : float
        Maximum minute-to-minute change (kW), default 35
    plot : bool
        If True, plot a histogram of all generated load values.
        
    Returns:
    --------
    Profiles: np.ndarray
        Array of shape (Profiles, 60) containing load profiles in kW.
        Each row is one profile with 60 values (one per minute in an hour).
    """

    rng = np.random.default_rng(random_state)
    n_minutes = 60
    
    # Initialize array to store profiles
    profiles = np.zeros((Profiles, n_minutes))
    
    for i in range(Profiles):
        # Start with a random value within the allowed range
        current_power = rng.uniform(P_min, P_max)
        profiles[i, 0] = current_power
        
        # Generate remaining minutes
        for minute in range(1, n_minutes):
            # Generate random change limited by P_delta
            delta = rng.uniform(-P_delta, P_delta)

            if current_power <= P_min + 1e-6:
                delta = rng.uniform(0, P_delta)

            elif current_power >= P_max - 1e-6:
                delta = rng.uniform(-P_delta, 0)

            # Apply change and clip to stay within bounds
            next_power = np.clip(current_power + delta, P_min, P_max)
            profiles[i, minute] = next_power
            current_power = next_power
    
    if plot:
        plt.figure(figsize=(7,4))
        plt.hist(profiles.flatten(), bins=50)
        plt.xlabel("Load [kW]", fontsize=13)
        plt.ylabel("Frequency", fontsize=13)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.show()

    return profiles


def Optimal_reserve_bid_ALSO_X (in_sample_profiles, q, M=10**4, silent=False):
    """
    Solve for optimal reserve bid using the ALSO-X method.
    
    
    Parameters:
    -----------
    in_sample_profiles : np.ndarray
        Array of shape (n_profiles, n_minutes) containing load profiles in kW.
    q : int
        Violation budget: maximum number of point violations allowed.
    M : float, optional
        Big-M parameter for the constraint linearization, default 10**4.
    silent : bool, optional
        If False (default), print computational details (runtime, variables, constraints).
        
    Returns:
    --------
    m : gurobipy.Model
        The solved Gurobi optimization model.
    c_up_value : float
        Optimal reserve bid value in kW, or np.nan if optimization failed.
    y_value : np.ndarray
        Array of shape (n_minutes, n_profiles) with binary decision variables.
        y[m, w] = 1 indicates that reserve bid c_up is not available at minute m in profile w.
    F_up : np.ndarray
        Transposed load profiles (n_minutes, n_profiles) representing available upward reserve.
    """

    m = gp.Model("Optimal_reserve_bid_ALSO-X")
    m.Params.OutputFlag = 0

    

    # Parameters
    n_profiles = in_sample_profiles.shape[0]
    n_minutes = in_sample_profiles.shape[1]
    F_up = (in_sample_profiles).T # F_up[m, w] = available upward reserve from load reduction
      

   # Variables
    c_up = m.addVar(lb=0.0, name="c_up")
    y = m.addVars(n_minutes, n_profiles, vtype=gp.GRB.BINARY, name="y")


    # Objective
    m.setObjective(c_up, gp.GRB.MAXIMIZE)


    # ALSO-X constraints
    for min in range(n_minutes):
        for w in range(n_profiles):
            m.addConstr(c_up - F_up[min, w] <= M * y[min, w], name=f"alsox_{min}_{w}")

    m.addConstr(
        gp.quicksum(y[min, w] for min in range(n_minutes) for w in range(n_profiles)) <= q,
        name="violation_budget"
    )

    m.optimize()

    if silent == False:
        # Print computational details
        runtime_sec    = m.Runtime
        num_vars       = int(m.NumVars)
        num_constrs    = int(m.NumConstrs)
        print(f"  Computational time:    {runtime_sec:.6f} s")
        print(f"  Decision variables:    {num_vars}")
        print(f"  Constraints:           {num_constrs}")


    c_up_value = c_up.X if m.Status == gp.GRB.OPTIMAL else np.nan
    y_value = np.array([[y[min, w].X for w in range(n_profiles)] for min in range(n_minutes)])

    return m, c_up_value, y_value, F_up


def Optimal_reserve_bid_CVaR (in_sample_profiles, epsilon, silent=False):
    """
    Solve for optimal reserve bid using Conditional Value at Risk (CVaR).
    
    
    Parameters:
    -----------
    in_sample_profiles : np.ndarray
        Array of shape (n_profiles, n_minutes) containing load profiles in kW.
        Each row represents one load profile with values for each minute.
    epsilon : float
        Risk tolerance parameter (0 to 1 - low to high). 
    silent : bool, optional
        If False (default), print computational details (runtime, variables, constraints).
        
    Returns:
    --------
    model : gurobipy.Model
        The solved Gurobi optimization model.
    c_up_value : float
        Optimal reserve bid value in kW, or np.nan if optimization failed.
    beta_value : float
        Optimal beta value (lower bound on tail shortfalls), or np.nan if optimization failed.
    zeta_value : np.ndarray
        Array of shape (n_minutes, n_profiles) representing shortfall variables.
    F_up : np.ndarray
        Transposed load profiles (n_minutes, n_profiles) representing available upward reserve.
    """

    model = gp.Model("Optimal_reserve_bid_CVaR")
    model.Params.OutputFlag = 0

    # Parameters
    n_profiles = in_sample_profiles.shape[0]
    n_minutes = in_sample_profiles.shape[1]
    F_up = (in_sample_profiles).T # F_up[m, w]

    # Decision variables
    c_up = model.addVar(lb=0.0, name="c_up")
    beta = model.addVar(ub=0.0, lb=-gp.GRB.INFINITY, name="beta")
    zeta = model.addVars(
        n_minutes, n_profiles,
        lb=-gp.GRB.INFINITY,
        ub=gp.GRB.INFINITY,
        name="zeta"
    )

    # Objective: maximize reserve bid
    model.setObjective(c_up, gp.GRB.MAXIMIZE)

    # CVaR constraints:
    # c_up - F_up[m,w] <= zeta[m,w]
    for m in range(n_minutes):
        for w in range(n_profiles):
            model.addConstr(c_up - F_up[m, w] <= zeta[m, w], name=f"cvar_link_{m}_{w}")

    # (1/|M||W|) * sum zeta <= (1-epsilon) * beta
    scale = 1.0 / (n_minutes * n_profiles)
    model.addConstr(
        scale * gp.quicksum(zeta[m, w] for m in range(n_minutes) for w in range(n_profiles))
        <= (1.0 - epsilon) * beta,
        name="cvar_avg"
    )

    # beta <= zeta[m,w]
    for m in range(n_minutes):
        for w in range(n_profiles):
            model.addConstr(beta <= zeta[m, w], name=f"beta_lb_{m}_{w}")

    model.optimize()

    if silent == False:
        # Print computational details
        runtime_sec    = model.Runtime
        num_vars       = int(model.NumVars)
        num_constrs    = int(model.NumConstrs)
        print(f"  Computational time:    {runtime_sec:.6f} s")
        print(f"  Decision variables:    {num_vars}")
        print(f"  Constraints:           {num_constrs}")


    c_up_value = c_up.X if model.Status == gp.GRB.OPTIMAL else np.nan
    beta_value = beta.X if model.Status == gp.GRB.OPTIMAL else np.nan
    zeta_value = np.array([[zeta[m, w].X for w in range(n_profiles)] for m in range(n_minutes)])

    return model, c_up_value, beta_value, zeta_value, F_up


def histogram_of_violations(c_up, F_up, title="Histogram of violations"):
    """
    Visualize the distribution of reserve bid violations across load profiles.
    
    Parameters:
    -----------
    c_up : float
        The reserve bid value in kW to evaluate against the available reserves.
    F_up : np.ndarray
        Array of shape (n_minutes, n_profiles) representing available upward reserve
        at each minute for each load profile.
    title : str, optional
        Title for the histogram plot, default "Histogram of violations".
        
    Returns:
    --------
    None
        Displays the histogram plot.
    """
    F_up = np.asarray(F_up, dtype=float)
    
    # Count violations per profile: for each profile w, count minutes m where F_up[m, w] < c_up
    violations_per_profile = (F_up < c_up).sum(axis=0)
    
    # Calculate mean
    mean_violations = violations_per_profile.mean()

    # Create histogram using unique counts and their frequencies
    unique_counts, counts = np.unique(violations_per_profile, return_counts=True)
    

    plt.figure(figsize=(6, 4))
    plt.bar(unique_counts, counts, width=0.8, edgecolor='black', alpha=0.7)
    plt.axvline(6, color='darkgreen', linestyle='--', linewidth=2, 
                label=f'P90 target (6 violations)')
    plt.axvline(mean_violations, color='darkred', linestyle=':', linewidth=2, 
                label=f'Mean: {mean_violations:.2f}')
    plt.xlabel("Number of violations per profile", fontsize=13)
    plt.ylabel("Number of profiles", fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlim(-2, 37)
    plt.ylim(0, 70)
    plt.title(title, fontsize=15)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()
    


def plot_Pxx_comparison_mean(alsox_results_df, title="Reliability requirement comparison"):
    """
    Visualize the comparison between optimal reserve bid and mean shortfall across reliability requirements.
    
    Parameters:
    -----------
    alsox_results_df : pd.DataFrame
        DataFrame containing optimization results with columns:
        - "Reliability requirement": reliability level labels (e.g., "P80", "P90", "P100")
        - "c_up_AlsoX": optimal reserve bid values in kW
        - "mean_shortfall": mean shortfall values in kW
    title : str, optional
        Title for the plot, default "Reliability requirement comparison".
        
    Returns:
    --------
    None
        Displays the dual-axis plot.
    """
    df = alsox_results_df.iloc[::-1].copy()

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    x = np.arange(len(df))
    labels = df["Reliability requirement"].astype(str).values


    line1 = ax1.plot(
        df["Reliability requirement"],
        df["c_up_AlsoX"],
        marker="o",
        linewidth=2,
        color="tab:blue",
        label="Optimal reserve bid"
    )

    line2 = ax2.plot(
        df["Reliability requirement"],
        df["mean_shortfall"],
        marker="s",
        linewidth=2,
        color="tab:orange",
        label="Mean reserve shortfall"
    )

    ax1.set_xlabel("Reliability requirement")
    ax1.set_ylabel("Reserve bid [kW]", color="tab:blue")
    ax2.set_ylabel("Mean shortfall [kW]", color="tab:orange")

    tick_step = 2
    ax1.set_xticks(x[::tick_step])
    ax1.set_xticklabels(labels[::tick_step])
    plt.setp(ax1.get_xticklabels(), rotation=0, ha="center")

    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)
    fig.tight_layout()
    plt.show()

def compute_normalized_Pxx_metrics(alsox_results_df, p_min=80, p_max=100, verbose=True):
    """
    Compute and normalize reliability metrics across different Pxx requirements.
    
    Parameters:
    -----------
    alsox_results_df : pd.DataFrame
        DataFrame containing optimization results with columns:
        - "Reliability requirement": reliability level labels (e.g., "P80", "P90", "P100")
        - "c_up_AlsoX": optimal reserve bid values in kW
        - "mean_shortfall": mean shortfall values in kW
    p_min : int, optional
        Minimum reliability percentile for filtering, default 80 (P80).
    p_max : int, optional
        Maximum reliability percentile for filtering, default 100 (P100).
    verbose : bool, optional
        If True (default), print the normalized results table to console.
        
    Returns:
    --------
    result : pd.DataFrame
        Filtered DataFrame with columns:
        - "Reliability requirement": reliability level labels
        - "c_up_pct": normalized reserve bid as percentage (0-100%)
        - "shortfall_pct": normalized shortfall as percentage (0-100%)
        - "difference_pct_points": percentage point difference between bid and shortfall
    """
    df = alsox_results_df.copy()

    # Extract numeric Pxx
    df["P_value"] = df["Reliability requirement"].str.extract(r'(\d+)').astype(int)

    # Sort properly (P100 → P80)
    df = df.sort_values("P_value", ascending=False).reset_index(drop=True)

    # --- Anchor values ---
    c_up_Pmin = df.loc[df["P_value"] == p_min, "c_up_AlsoX"].values[0]
    c_up_Pmax = df.loc[df["P_value"] == p_max, "c_up_AlsoX"].values[0]

    shortfall_Pmin = df.loc[df["P_value"] == p_min, "mean_shortfall"].values[0]
    shortfall_Pmax = df.loc[df["P_value"] == p_max, "mean_shortfall"].values[0]

    # --- Normalize ---
    df["c_up_pct"] = (
        (df["c_up_AlsoX"] - c_up_Pmax) / (c_up_Pmin - c_up_Pmax)
    ) * 100

    df["shortfall_pct"] = (
        (df["mean_shortfall"] - shortfall_Pmax) / (shortfall_Pmin - shortfall_Pmax)
    ) * 100

    # --- Difference ---
    df["difference_pct_points"] = df["c_up_pct"] - df["shortfall_pct"]

    # --- Filter range ---
    df_filtered = df[(df["P_value"] >= p_min) & (df["P_value"] <= p_max)].copy()

    result = df_filtered[[
        "Reliability requirement",
        "c_up_pct",
        "shortfall_pct",
        "difference_pct_points"
    ]]

    if verbose:
        print(result.to_string(index=False, float_format="%.2f"))

    return result


def plot_Pxx_comparison(alsox_results_df, title="Reliability requirement comparison"):
    """
    Visualize the comparison between optimal reserve bid and reserve shortfall availability across reliability requirements.
    
    Parameters:
    -----------
    alsox_results_df : pd.DataFrame
        DataFrame containing optimization results with columns:
        - "Reliability requirement": reliability level labels (e.g., "P80", "P90", "P100")
        - "c_up_AlsoX": optimal reserve bid values in kW
        - "share_not_available": percentage of time reserve is not available (0-100%)
    title : str, optional
        Title for the plot, default "Reliability requirement comparison".
        
    Returns:
    --------
    None
        Displays the dual-axis plot.
    """
    df = alsox_results_df.iloc[::-1].copy()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    line1 = ax1.plot(
        df["Reliability requirement"],
        df["c_up_AlsoX"],
        marker="o",
        linewidth=2,
        color="tab:blue",
        label="Optimal reserve bid"
    )

    line2 = ax2.plot(
        df["Reliability requirement"],
        df["share_not_available"],
        marker="s",
        linewidth=2,
        color="tab:orange",
        label="Expected reserve shortfall"
    )

    ax1.set_xlabel("Reliability requirement")
    ax1.set_ylabel("Reserve bid [kW]", color="tab:blue")
    ax2.set_ylabel("Expected reserve shortfall [%]", color="tab:orange")

    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)
    fig.tight_layout()
    plt.show()
