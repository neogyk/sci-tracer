N = 256
Ld = 525e-9
w0 = 0.01
z = 1.0
L = 0.03
Gau, Gau_Pro = propagate_gaussian_beam(N, Ld, w0, z, L)
assert Gau.shape == (N+1, N+1), f'Gau shape {Gau.shape} != ({N+1},{N+1})'
assert Gau_Pro.shape == (N+1, N+1), f'Gau_Pro shape {Gau_Pro.shape} != ({N+1},{N+1})'
print('All assertions passed: function works correctly.')