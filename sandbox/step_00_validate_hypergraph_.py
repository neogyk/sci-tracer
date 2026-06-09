from itertools import combinations


class Hypergraph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
    
    def validate_hypergraph(self):
        # Check largest partition size does not exceed 20
        is_valid_partition = self.max_partition_size() <= 20
        return is_valid_partition
    
    def max_partition_size(self):
        # Calculate the size of largest partition
        node_sets = [set(edge) for edge in self.edges]
        
        # Create dict for storing connected components
        component_map = dict((node, {node}) for node in self.nodes)
        
        for i, edge1 in enumerate(self.edges):
            for edge2 in self.edges[i+1:]:
                # Check if current two edges have any members overlapping
                if set(edge1).intersection(set(edge2)):
                    # Merging sets in component_map based on intersection of edges
                    union_set = component_map[edge1[0]].union(component_map[edge2[0]])
                    for node in union_set:
                        component_map[node] = union_set
        
        # Identify unique components
        unique_components = set(frozenset(component) for component in component_map.values())
        
        # Obtain the sizes of all identified components
        component_sizes = [len(component) for component in unique_components]
        
        # Determine maximum component size
        max_size = max(component_sizes, default=0)
        
        return max_size


# Hypergraph Example
# Define nodes and edges (tuples indicating connections)
nodes = ['A', 'B', 'C', 'D', 'E', 'F']
edges = [('A', 'B'), ('B', 'C', 'D'), ('C', 'E', 'F'), ('E', 'F')]

# Initialize hypergraph
hypergraph = Hypergraph(nodes, edges)

# Validate hypergraph
is_valid = hypergraph.validate_hypergraph()

# Output result
if is_valid:
    print("Hypergraph is valid: it does not contain partitions > 20.")
else:
    print("Hypergraph is invalid: it contains partitions > 20.")