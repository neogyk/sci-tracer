from typing import List, Tuple

def main(
    n_grid: int,
    x_out: float,
    courant: float,
    t_max: float,
    t_check: float
) -> List[Tuple[float, float]]:
    """
    Main simulation routine for the 3+1 Maxwell system.

    Parameters
    ----------
    n_grid : int
        Number of grid points in each spatial dimension.
    x_out : float
        Physical size of the cubic domain (from -x_out/2 to x_out/2).
    courant : float
        Courant factor (0 < courant <= 1) for the Crank-Nicholson step.
    t_max : float
        Total simulation time.
    t_check : float
        Time interval between constraint evaluations.

    Returns
    -------
    List[Tuple[float, float]]
        List of (time, constraint L2 norm) pairs recorded during the evolution.
    """
    obj = MaxwellObject(n_grid, x_out)
    initialize_dipole(obj)
    constraints = integrate(obj, t_max, courant, t_check)
    return constraints