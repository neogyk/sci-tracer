def Bspline(xi, i, p, Xi):
    if p == 0:
        if Xi[i] <= xi < Xi[i+1]:
            return 1.0
        else:
            return 0.0
    else:
        left = 0.0
        if Xi[i+p] - Xi[i] != 0:
            left = (xi - Xi[i]) / (Xi[i+p] - Xi[i]) * Bspline(xi, i, p-1, Xi)
        right = 0.0
        if Xi[i+p+1] - Xi[i+1] != 0:
            right = (Xi[i+p+1] - xi) / (Xi[i+p+1] - Xi[i+1]) * Bspline(xi, i+1, p-1, Xi)
        return left + right