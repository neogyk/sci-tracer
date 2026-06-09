from typing import List, Tuple

# Assume refined is a list of float energies from previous steps
refined.sort()
bound_states: List[Tuple[int, float]] = [(n, E) for n, E in enumerate(refined)]