def scenario_generator():
    import pandas as pd
    # Extract wind data
    wind_data = pd.read_csv("Data/ninja_wind_55.5783_15.7764_corrected.csv", comment='#', parse_dates=['time', 'local_time'])

    wind_data.drop(columns=["local_time"], inplace=True)

    # Turn into 





def solve_stochastic_strategy_one_price(in_sample_scenarios):
    import gurobipy as gp

    m = gp.Model("stochastic_one_price")
    m.Params.OutputFlag = 0

    n_hours = 24
    n_scenarios = 1600
    prob_scenarios = 1 / n_scenarios

    # (scenarios, hours, parameters)

    # Parameters
    P_nom = 500 
    lambda_DA = in_sample_scenarios[:,:,1] # fix how it is extracted
    p_real = in_sample_scenarios[:,:,2]
    deficit_bin = in_sample_scenarios[:,:,3]
    lambda_bal = 1.25 * lambda_DA * deficit_bin + 0.85 * lambda_DA * (1 - deficit_bin) 

    # Variables
    p_DA = m.addVars(n_hours, lb=0, name="DayAhead offer")
    Delta = m.addVars(n_hours, n_scenarios, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="Difference_DA_real")

    m.setObjective(gp.quicksum(prob_scenarios * 
                                (lambda_DA[t, s] * p_DA[t]
                                 + lambda_bal[t, s] * Delta[t, s])
                                for t in range(n_hours) for s in range(n_scenarios)))


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

    return m, p_DA, Delta


def solve_stochastic_strategy_two_price(in_sample_scenarios):
    import gurobipy as gp

    m = gp.Model("stochastic_two_price")
    m.Params.OutputFlag = 0

    n_hours = 24
    n_scenarios = 1600
    prob_scenarios = 1 / n_scenarios

    # Parameters
    P_nom = 500 
    lambda_DA = in_sample_scenarios[:,:,1] # fix how it is extracted
    p_real = in_sample_scenarios[:,:,2]
    deficit_bin = in_sample_scenarios[:,:,3]
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
        for t in range(n_hours) for s in range(n_scenarios)))

    # Constraints

    # Capacity limit 
    for t in range(n_hours):
        m.addConstr(p_DA[t] <= P_nom, name=f"Capacity_limit_{t}")

    # Constraints setting limits for Delta up and down
    for t in range(n_hours):
        for s in range(n_scenarios):
            m.addConstr(Delta[t, s] == p_real[t, s] - p_DA[t], name=f"Difference_DA_real_{t}_{s}")
            m.addConstr(Delta[t, s] == Delta_up[t, s] - Delta_down[t, s], name=f"Difference_split_{t}_{s}")
            m.addConstr(Delta_up[t, s] >= 0, name=f"Difference_up_{t}_{s}")
            m.addConstr(Delta_down[t, s] >= 0, name=f"Difference_down_{t}_{s}")

    # Optimize
    m.optimize()
    return m, p_DA, Delta_up, Delta_down