from sympy import symbols, sqrt

def calculate_perimeter_squared() -> float:
    # Define the variable x for the side length
    x = symbols('x', real=True, positive=True)
    
    # Equation for side sums: 5*x^2 = 308
    x_value = sqrt(308 / 5)

    # Calculate the perimeter
    perimeter = 5 * x_value
    
    # Calculate the square of the perimeter
    perimeter_squared = perimeter ** 2
    
    return perimeter_squared.evalf()

# Call the function and print the result
perimeter_squared_value = calculate_perimeter_squared()
perimeter_squared_value