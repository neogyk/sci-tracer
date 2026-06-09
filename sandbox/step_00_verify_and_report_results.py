import numpy as np
import matplotlib.pyplot as plt

# Given vertices of triangle ABC
A = np.array([0, 13])
B = np.array([12, 0])
C = np.array([-8, 0])

# Function to rotate a point around the origin
def rotate_point(point: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a point counterclockwise by a given angle around the origin."""
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return np.dot(rotation_matrix, point)

# Rotate the triangle by a specified angle counterclockwise (e.g., 60 degrees)
angle_degrees = 60
angle_radians = np.radians(angle_degrees)

# Perform rotation
A_prime = rotate_point(A, angle_radians)
B_prime = rotate_point(B, angle_radians)
C_prime = rotate_point(C, angle_radians)

# Vertices of hexagon AA'CC'BB'
hexagon_vertices = np.array([A, A_prime, C, C_prime, B, B_prime, A])

# Plotting to verify the results
plt.figure(figsize=(8, 8))
plt.plot(hexagon_vertices[:, 0], hexagon_vertices[:, 1], 'b-o', label='Hexagon Vertices')
plt.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], 'r--', label='Original Triangle')
plt.plot([A_prime[0], B_prime[0], C_prime[0], A_prime[0]], [A_prime[1], B_prime[1], C_prime[1], A_prime[1]], 'g--', label='Rotated Triangle')
plt.scatter([0], [0], c='k', marker='+', s=100, label='Circumcenter')
plt.title('Hexagon Formed by Triangle Rotation')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.legend()
plt.axis('equal')
plt.show()