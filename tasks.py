from graphlib import TopologicalSorter

# Define dependencies: 'B' depends on 'A', 'D' depends on 'B' and 'C'
graph = {"B": {"A"}, "D": {"B", "C"}}
ts = TopologicalSorter(graph)
ts.prepare()

while ts.is_active():
    for node in ts.get_ready():
        # You can run these tasks concurrently using a ThreadPool or asyncio
        print(f"Running task: {node}")
        ts.done(node)

#
from pyschedule import Scenario, solvers, plot

# Create a scenario with a planning horizon (e.g., 100 units of time)
S = Scenario('Task_Orchestration', horizon=100)

# Define resources (e.g., Workers, GPUs, or CPU cores)
# Let's say we have 2 CPUs and 1 GPU
CPU = S.Resource('CPU', size=2)
GPU = S.Resource('GPU')

# Solve the scheduling problem
solvers.mip.solve(S)

# Print the start times found by the solver
for t in S.tasks():
    print(f"{t}: starts at {t.start_time}, ends at {t.start_time + t.length}")

# This creates a Gantt chart automatically
plot.plot(S, show_resource_labels=True)