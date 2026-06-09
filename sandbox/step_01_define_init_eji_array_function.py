import sympy as sp
import numpy as np

def init_eji_array(energy: float, energy_vertices: list[float]) -> tuple:
    """
    Create a 5x5 array of symbolic variables epsilon_{ji} and map them to float values.
    
    Args:
        energy: Energy E (float)
        energy_vertices: List of 4 vertex energies sorted in increasing order [ε1, ε2, ε3, ε4]
    
    Returns:
        symbols: dict mapping symbol names to sympy symbols
        value_map: dict mapping sympy symbols to float values (energies differences)
    """
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