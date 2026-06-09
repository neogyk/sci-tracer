import numpy as np
from typing import Tuple

def initialize_grid(
    price_step: int,
    time_step: int,
    strike: float,
    max_price: float,
    min_price: float
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """
    Initialize price and time grids for Black-Scholes finite difference method.

    Parameters:
        price_step (int): Number of price grid points.
        time_step (int): Number of time grid points.
        strike (float): Strike price (not used in grid creation, included for consistency).
        max_price (float): Maximum asset price in original scale.
        min_price (float): Minimum asset price in original scale.

    Returns:
        p (np.ndarray): Log-transformed price grid of shape (price_step,).
        dp (float): Step size in log-price.
        T (np.ndarray): Time grid from 0 to 1 of shape (time_step,).
        dt (float): Step size in time.
    """
    p = np.linspace(np.log(min_price), np.log(max_price), price_step)
    dp = (np.log(max_price) - np.log(min_price)) / (price_step - 1)
    T = np.linspace(0, 1, time_step)
    dt = 1.0 / (time_step - 1)
    return p, dp, T, dt