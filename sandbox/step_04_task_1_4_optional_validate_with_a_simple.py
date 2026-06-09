import numpy as np

def kl_divergence(p, q):
    return np.sum(p * np.log2(p / q))

def mutual_information(channel, prior):
    p_y = channel @ prior
    return np.sum([prior[i] * kl_divergence(channel[:, i], p_y) for i in range(len(prior))])

def blahut_arimoto(channel, e):
    n_inputs = channel.shape[1]
    prior = np.ones(n_inputs) / n_inputs
    rate = mutual_information(channel, prior)
    while True:
        p_y = channel @ prior
        exp_term = np.array([np.exp(kl_divergence(channel[:, j], p_y)) for j in range(n_inputs)])
        new_prior = prior * exp_term
        new_prior /= new_prior.sum()
        new_rate = mutual_information(channel, new_prior)
        if abs(new_rate - rate) < e:
            return new_rate
        prior = new_prior
        rate = new_rate

# Test on BSC
bsc = np.array([[0.9, 0.1], [0.1, 0.9]])
cap = blahut_arimoto(bsc, 1e-6)
expected = 1 - (0.1 * np.log2(1/0.1) + 0.9 * np.log2(1/0.9))
assert abs(cap - expected) < 1e-5, f"Capacity mismatch: {cap} vs {expected}"
print('Test passed. Capacity:', cap)