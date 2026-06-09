import numpy as np

def build_mesh(N):
    h = 1.0 / N
    x = np.linspace(0, 1, N+1)
    elements = [(i, i+1) for i in range(N)]
    return x, elements, h