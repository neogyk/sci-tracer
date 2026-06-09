import numpy as np

def SpeciesGrowth(g, pref, Rs, alive):
    N, R = g.shape
    g_temp = np.zeros(N)
    r_temp = np.zeros(N, dtype=int)
    available_resources = Rs.copy()
    for i in range(N):
        if not alive[i]:
            g_temp[i] = 0
            r_temp[i] = 0
            continue
        pref_order = pref[i] - 1
        for res_idx in pref_order:
            if available_resources[res_idx] > 1e-12:
                r_temp[i] = res_idx + 1
                g_temp[i] = g[i, res_idx]
                available_resources[res_idx] = 0.0
                break
        else:
            r_temp[i] = 0
            g_temp[i] = 0
    return g_temp, r_temp