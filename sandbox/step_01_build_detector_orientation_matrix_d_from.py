def compute_D(yaw: float, pitch: float, roll: float) -> "np.ndarray":
    """
    Compute the detector orientation matrix D = Rx(roll) @ Ry(pitch) @ Rz(yaw)
    from yaw, pitch, roll angles given in degrees.

    Parameters
    ----------
    yaw : float
        Yaw angle (rotation about z-axis) in degrees.
    pitch : float
        Pitch angle (rotation about y-axis) in degrees.
    roll : float
        Roll angle (rotation about x-axis) in degrees.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix D.
    """
    import numpy as np

    # Convert degrees to radians
    yaw_r = np.deg2rad(yaw)
    pitch_r = np.deg2rad(pitch)
    roll_r = np.deg2rad(roll)

    # Elementary rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll_r), -np.sin(roll_r)],
                   [0, np.sin(roll_r), np.cos(roll_r)]])
    Ry = np.array([[np.cos(pitch_r), 0, np.sin(pitch_r)],
                   [0, 1, 0],
                   [-np.sin(pitch_r), 0, np.cos(pitch_r)]])
    Rz = np.array([[np.cos(yaw_r), -np.sin(yaw_r), 0],
                   [np.sin(yaw_r), np.cos(yaw_r), 0],
                   [0, 0, 1]])

    # Composite rotation: yaw -> pitch -> roll
    D = Rx @ Ry @ Rz
    return D