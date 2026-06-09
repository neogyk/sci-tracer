def NURBS_2D(xi_1, xi_2, i_1, i_2, p_1, p_2, n_1, n_2, Xi_1, Xi_2, w):
    def bspline_basis(i, p, u, Xi):
        if p == 0:
            return 1.0 if Xi[i] <= u < Xi[i+1] else 0.0
        else:
            left = 0.0
            if Xi[i+p] != Xi[i]:
                left = ((u - Xi[i]) / (Xi[i+p] - Xi[i])) * bspline_basis(i, p-1, u, Xi)
            right = 0.0
            if Xi[i+p+1] != Xi[i+1]:
                right = ((Xi[i+p+1] - u) / (Xi[i+p+1] - Xi[i+1])) * bspline_basis(i+1, p-1, u, Xi)
            return left + right
    denom = 0.0
    for j1 in range(n_1):
        for j2 in range(n_2):
            N1 = bspline_basis(j1, p_1, xi_1, Xi_1)
            N2 = bspline_basis(j2, p_2, xi_2, Xi_2)
            idx = j1 * n_2 + j2
            denom += N1 * N2 * w[idx]
    N1_i = bspline_basis(i_1, p_1, xi_1, Xi_1)
    N2_i = bspline_basis(i_2, p_2, xi_2, Xi_2)
    idx_i = i_1 * n_2 + i_2
    num = N1_i * N2_i * w[idx_i]
    if denom == 0:
        return 0.0
    return num / denom