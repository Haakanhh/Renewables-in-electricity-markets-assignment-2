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
Load_profiles = uf.Load_profile_generation(random_state=seed, Profiles=300, P_max=600, P_min=220, P_delta=35)

#%% ----------------------
# Task 2.1) In-sample Decision Making: Offering Strategy Under the P90 Requirement
# ------------------------

