#%%
""" Creation of the uncertainty scenarios related to the need
for up and down regulation """

import random

num_scenarios = 4
hours = 24
random.seed(42) # This is added for reproducibility
scenarios = [
    [random.randint(0, 1) for _ in range(hours)]
    for _ in range(num_scenarios)
]

for i, scenario in enumerate(scenarios, start=1):
    print(f"Scenario {i}: {scenario}")
# SI=1 represents supply deficit and SI=0 represents supply surplus.
#%% Balancing prices for up and down regulation

# If the system is in deficit: BP = 1.25 × DA price.
# If the system is in surplus: BP = 0.85 × DA price.
