import numpy as np

data = {
    'BASE': np.array([-463, -612, -907]),
    'CAMON': np.array([-519, -689, -1002]),
    'COELA': np.array([-541, -688, -1025]),
    'Embodied': np.array([-555, -730, -987]),
    'Hmas': np.array([-521, -692, -1004]),
}

results = {}
for k,v in data.items():
    mean = v.mean()
    # population std
    pop_std = v.std(ddof=0)
    # sample std
    sam_std = v.std(ddof=1)
    results[k] = (mean, pop_std, sam_std)
print(results)