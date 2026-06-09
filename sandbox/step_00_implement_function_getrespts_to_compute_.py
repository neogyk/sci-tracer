import numpy as np

def GetResPts(M):
    # M: R x N numpy array with non-negative entries
    col_sums = M.sum(axis=0)
    col_sums[col_sums == 0] = 1  # prevent division by zero; zero columns become zero vector
    res_pts = M / col_sums
    return res_pts