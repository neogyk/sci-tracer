import numpy as np

def make_IC(n):
    a = -np.pi / 2
    b = np.pi / 2
    h = (b - a) / (n - 1)          # cell width
    x_edges = np.linspace(a, b, n) # vertices
    # Gauss points on [-1,1] and weights
    gauss_nodes = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    gauss_weights = np.array([5/9, 8/9, 5/9])
    v = np.zeros(n-1)
    for i in range(n-1):
        xL = x_edges[i]
        xR = x_edges[i+1]
        mid = (xL + xR) / 2
        half = (xR - xL) / 2
        integral = 0.0
        for j in range(3):
            xi = gauss_nodes[j]
            x = mid + half * xi   # map to cell
            # piecewise u0(x)
            if x <= 0:
                u_val = np.sin(x) - 1
            else:
                u_val = np.sin(x) + 1
            integral += gauss_weights[j] * u_val
        integral *= half
        v[i] = integral / h
    return v