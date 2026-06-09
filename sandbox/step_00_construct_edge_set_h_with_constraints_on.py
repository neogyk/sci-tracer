# Given a vertex set V of size 64, we need to construct an edge set H
# such that each edge consists of 3 vertices and there are at most 20 edges.

# Define vertex set V as the integers from 1 to 64
V = set(range(1, 65))

# Now define the edge set H with specific constraints
H = [
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9},
    {10, 11, 12},
    {13, 14, 15},
    {16, 17, 18},
    {19, 20, 21},
    {22, 23, 24},
    {25, 26, 27},
    {28, 29, 30},
    {31, 32, 33},
    {34, 35, 36},
    {37, 38, 39},
    {40, 41, 42},
    {43, 44, 45},
    {46, 47, 48},
    {49, 50, 51},
    {52, 53, 54},
    {55, 56, 57},
    {58, 59, 60}
]

# Constraining to no partition having more than 20 elements
# We need to ensure the number of edges H is at most 20
if len(H) <= 20:
    print("Edge set H satisfies the constraints.")
else:
    print("Error: Edge set H exceeds the allowable number of edges.")