import sympy as sp
import numpy as np

def init_eji_array(energy, energy_vertices):
    # energy_vertices assumed sorted: [ε1, ε2, ε3, ε4] with ε1<ε2<ε3<ε4
    e = [energy] + list(energy_vertices)  # indices 0..4
    symbols = {}
    value_map = {}
    for i in range(5):
        for j in range(5):
            sym_name = f'epsilon_{j}{i}'
            sym = sp.symbols(sym_name)
            symbols[sym_name] = sym
            value_map[sym] = float(e[j] - e[i])
    return symbols, value_map

# Test init_eji_array
energy = 0.5
energy_vertices = [0.1, 0.3, 0.7, 0.9]
symbols, value_map = init_eji_array(energy, energy_vertices)

# Check a few values
assert abs(value_map[symbols['epsilon_00']] - 0.0) < 1e-12
assert abs(value_map[symbols['epsilon_21']] - (0.7-0.3)) < 1e-12
assert abs(value_map[symbols['epsilon_40']] - (0.9-0.5)) < 1e-12
assert abs(value_map[symbols['epsilon_10']] - (0.1-0.5)) < 1e-12
assert abs(value_map[symbols['epsilon_34']] - (0.7-0.9)) < 1e-12
print('All assertions passed.')