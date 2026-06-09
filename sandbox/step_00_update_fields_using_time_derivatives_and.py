from typing import List

import numpy as np


def update_fields(
    maxwell: object,
    fields: List[np.ndarray],
    fields_dot: List[np.ndarray],
    factor: float,
    dt: float,
) -> List[np.ndarray]:
    """
    Perform Euler integration update on the field components.

    Parameters
    ----------
    maxwell : object
        Object with attribute `n_vars` indicating the number of field variables.
    fields : List[np.ndarray]
        Current field values (list of 3D numpy arrays, length = maxwell.n_vars).
    fields_dot : List[np.ndarray]
        Time derivatives of the fields (same structure as fields).
    factor : float
        Scaling factor for multi‑stage methods (e.g., 1 for forward Euler, 1/6 for RK4 sum).
    dt : float
        Time step size.

    Returns
    -------
    List[np.ndarray]
        Updated field arrays after one Euler substep.
    """
    return [fields[i] + factor * dt * fields_dot[i] for i in range(maxwell.n_vars)]