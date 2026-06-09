def propagate_gaussian_beam(N, Ld, w0, z, L):
    # Coordinate grid
    x = np.linspace(-L/2, L/2, N+1)
    y = x
    X, Y = np.meshgrid(x, y)
    # Initial Gaussian field at waist (z=0)
    E0 = np.exp(-(X**2 + Y**2) / w0**2)
    # Spatial frequency grid
    dx = L / N  # spacing
    fx = np.fft.fftfreq(N+1, d=dx)
    fy = fx
    FX, FY = np.meshgrid(fx, fy)
    # Transfer function (paraxial Fresnel)
    H = np.exp(-1j * np.pi * Ld * z * (FX**2 + FY**2))
    # Propagate
    E0_FT = np.fft.fft2(E0)
    E_FT = E0_FT * H
    E = np.fft.ifft2(E_FT)
    # Output amplitudes
    Gau = np.abs(E0)
    Gau_Pro = np.abs(E)
    return Gau, Gau_Pro