# Assuming energies and node_counts are defined from previous steps
bounds = []
for i in range(len(energies) - 1):
    if node_counts[i] < node_counts[i+1]:
        bounds.append((energies[i], energies[i+1]))

# Optionally print the bounds
print("Identified bound state energy intervals:", bounds)