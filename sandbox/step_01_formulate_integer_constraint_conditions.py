from typing import Tuple, List

def validate_integer_sides(x: int, y: int, AI: int, IC: int) -> Tuple[bool, List[str]]:
    """
    Validates integer constraints on the sides of the triangles.
    
    Args:
    x (int): Side measure corresponding to $x$ in perimeter expressions.
    y (int): Side measure corresponding to $y$ in perimeter expressions.
    AI (int): Side measure from vertex A to the incenter I.
    IC (int): Side measure from incenter I to vertex C.
    
    Returns:
    Tuple[bool, List[str]]: A tuple containing a boolean indicating whether all constraints are met and a list of
                            strings detailing any violations.
    """
    violations = []
    
    # Ensure all sides are positive integers
    if x <= 0:
        violations.append("Side x is not a positive integer.")
    if y <= 0:
        violations.append("Side y is not a positive integer.")
    if AI <= 0:
        violations.append("Side AI is not a positive integer.")
    if IC <= 0:
        violations.append("Side IC is not a positive integer.")
    
    # Validate triangle inequality for all possible combinations of triangle formations
    if x + y <= AI:
        violations.append("Triangle inequality violated: x + y <= AI.")
    if x + AI <= y:
        violations.append("Triangle inequality violated: x + AI <= y.")
    if y + AI <= x:
        violations.append("Triangle inequality violated: y + AI <= x.")
    
    if AI + IC <= y:
        violations.append("Triangle inequality violated: AI + IC <= y.")
    if AI + y <= IC:
        violations.append("Triangle inequality violated: AI + y <= IC.")
    if IC + y <= AI:
        violations.append("Triangle inequality violated: IC + y <= AI.")
    
    is_valid = len(violations) == 0
    return (is_valid, violations)

# Example usage:
# Suppose x, y, AI, IC are given specific values from prior algebraic solving steps
x, y, AI, IC = 10, 20, 5, 15
validity, constraint_violations = validate_integer_sides(x, y, AI, IC)
if validity:
    print("All integer constraints are satisfied.")
else:
    print("Constraint violations found:")
    for violation in constraint_violations:
        print(violation)