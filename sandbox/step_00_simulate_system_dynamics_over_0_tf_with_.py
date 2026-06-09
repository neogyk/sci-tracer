import numpy as np
from typing import List

def SpeciesGrowth(spc: np.ndarray, res: np.ndarray, b: np.ndarray, c: np.ndarray, w: np.ndarray, m: np.ndarray) -> np.ndarray:
    N = len(spc)
    g_spc = np.zeros(N)
    for i in range(N):
        sum_res = np.dot(c[i], w * res)
        g_spc[i] = b[i] * (sum_res - m[i])
    return g_spc

def ResourcesUpdate(spc: np.ndarray, res: np.ndarray, c: np.ndarray, r: np.ndarray, K: np.ndarray) -> np.ndarray:
    R = len(res)
    consumption = np.zeros(R)
    for alpha in range(R):
        for i in range(len(spc)):
            consumption[alpha] += c[i, alpha] * spc[i] * res[alpha]
    f_res = r * (K - res) - consumption
    return f_res

def Simulate(spc_init: np.ndarray, res_init: np.ndarray, b: np.ndarray, c: np.ndarray, w: np.ndarray, m: np.ndarray, r: np.ndarray, K: np.ndarray, tf: float, dt: float, SPC_THRES: float) -> List[int]:
    N = len(spc_init)
    R = len(res_init)
    spc = spc_init.copy()
    res = res_init.copy()
    steps = int(round(tf / dt))
    for _ in range(steps):
        g_spc = SpeciesGrowth(spc, res, b, c, w, m)
        f_res = ResourcesUpdate(spc, res, c, r, K)
        spc = spc + dt * g_spc
        res = res + dt * f_res
    survivors = [i for i in range(N) if spc[i] > SPC_THRES]
    return survivors