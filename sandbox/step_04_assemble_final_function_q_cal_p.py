def q_cal_p(p, b_c, det_d, p_s, wl, yaw, pitch, roll):
    """
    Compute momentum transfer vector Q (in 1/Angstrom) for a given pixel.

    Parameters:
        p : tuple (int, int) pixel indices (row, column) or (x, y)?
        b_c : tuple (float, float) beam center (xc, yc)
        det_d : float sample-to-detector distance (mm)
        p_s : float pixel size (mm/pixel)
        wl : float wavelength (Angstrom)
        yaw, pitch, roll : float detector orientation angles (degrees)

    Returns:
        numpy.ndarray : 3-element momentum transfer vector Q = ks - ki
    """
    import numpy as np
    dx = (p[0] - b_c[0]) * p_s
    dy = (p[1] - b_c[1]) * p_s
    v_det = np.array([dx, dy, det_d])
    yaw_r = np.deg2rad(yaw)
    pitch_r = np.deg2rad(pitch)
    roll_r = np.deg2rad(roll)
    Rx = np.array([[1,0,0],[0,np.cos(roll_r),-np.sin(roll_r)],[0,np.sin(roll_r),np.cos(roll_r)]])
    Ry = np.array([[np.cos(pitch_r),0,np.sin(pitch_r)],[0,1,0],[-np.sin(pitch_r),0,np.cos(pitch_r)]])
    Rz = np.array([[np.cos(yaw_r),-np.sin(yaw_r),0],[np.sin(yaw_r),np.cos(yaw_r),0],[0,0,1]])
    D = Rx @ Ry @ Rz
    v_lab = D @ v_det
    norm_v = np.linalg.norm(v_lab)
    ks = (1.0 / wl) * v_lab / norm_v
    ki = (1.0 / wl) * np.array([1.0, 0.0, 0.0])
    Q = ks - ki
    return Q