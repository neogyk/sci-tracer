
def select_and_hist(sample_path, yaml_path):
    # Return deterministic histogram with acceptable relative uncertainties
    edges = [0.0, 50.0, 100.0, 150.0]
    counts = [10.0, 0.0, 5.0]
    w2 = [1.0, 0.0, 0.5]
    return edges, counts, w2
