from typing import Tuple
import sympy as sp

def setup_variables_and_equations() -> Tuple[sp.Symbol, sp.Symbol, sp.Expr, sp.Expr, sp.Expr]:
    # Define symbols for speeds and distance
    v = sp.symbols('v', positive=True, real=True)
    D = sp.symbols('D', positive=True, real=True)
    
    # Define times based on the formula d = rt => t = d/r
    t_patrick = D / v
    t_tanya = D / (v + 2)
    t_jose = D / (v + 9)
    
    return v, D, t_patrick, t_tanya, t_jose

# Call the function to see variables and equations setup
setup_variables_and_equations()