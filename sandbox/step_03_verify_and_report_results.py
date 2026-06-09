import numpy as np
import matplotlib.pyplot as plt

def plot_hexagon(vertices: np.ndarray, title: str = "Hexagon Verification") -> None:
    """
    Plots a hexagon given by the array of vertices.

    :param vertices: numpy array of shape (6, 2) containing the hexagon's vertices.
    :param title: Title of the plot for distinction.
    """
    
    # Append the first vertex to the end to close the hexagon loop
    vertices = np.append(vertices, [vertices[0]], axis=0)
    
    # Plot hexagon
    plt.figure()
    plt.plot(vertices[:, 0], vertices[:, 1], 'b-o', label='Hexagon Edge')
    
    # Set plot parameters
    plt.title(title)
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)

    # Annotate vertices
    for i, vertex in enumerate(vertices[:-1]):
        plt.text(vertex[0], vertex[1], f'V{i}', fontsize=12, ha='right')

    # Show plot
    plt.legend()
    plt.show()

# Example dummy vertices for demonstration, to be replaced with computed vertices
vertices = np.array([
    [0.0, 1.0],
    [1.0, 1.0],
    [1.5, 0.0],
    [1.0, -1.0],
    [0.0, -1.0],
    [-0.5, 0.0]
])

# Call plotting function
plot_hexagon(vertices, "Computed Hexagon Verification")