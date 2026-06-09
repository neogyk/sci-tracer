def shape1(x, xi_minus1, xi, h):
    """
    Shape function for the right-hand node of the element.
    Parameters:
        x: evaluation point
        xi_minus1: coordinate of left node
        xi: coordinate of right node
        h: element length (xi - xi_minus1)
    Returns:
        shape function value at x
    """
    return (x - xi_minus1) / h

def shape2(x, xi, xi_plus1, h):
    """
    Shape function for the left-hand node of the element.
    Parameters:
        x: evaluation point
        xi: coordinate of left node
        xi_plus1: coordinate of right node
        h: element length (xi_plus1 - xi)
    Returns:
        shape function value at x
    """
    return (xi_plus1 - x) / h