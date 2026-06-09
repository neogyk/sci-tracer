import numpy as np

def calc_transition(T_list, mag2_list):
    # Compute the numerical derivative of mag2 with respect to T using central differences
    derivative = np.gradient(mag2_list, T_list)
    # Find the index of the minimum (most negative) derivative
    min_idx = np.argmin(derivative)
    return float(T_list[min_idx])