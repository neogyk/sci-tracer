def Numerov(f_in, u_at_0, up_at_0, step):
    import numpy as np
    N = len(f_in)
    u = np.zeros(N)
    u[0] = u_at_0
    # second-order Taylor for u[1]
    u[1] = u_at_0 + step * up_at_0 + 0.5 * step**2 * f_in[0] * u_at_0
    # explicit Numerov recursion
    for i in range(1, N-1):
        coeff = 1.0 + (step**2 / 12.0) * f_in[i+1]
        u[i+1] = (2.0 * (1.0 - 5.0 * step**2 / 12.0 * f_in[i]) * u[i]
                   - (1.0 + step**2 / 12.0 * f_in[i-1]) * u[i-1]) / coeff
    return u