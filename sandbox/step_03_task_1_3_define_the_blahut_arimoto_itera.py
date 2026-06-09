import numpy as np

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute the Kullback-Leibler divergence D(p || q) in bits.
    Assumes p and q are 1-D arrays of equal length, q_i > 0.
    """
    return float(np.sum(p * np.log2(p / q)))

def mutual_information(channel: np.ndarray, prior: np.ndarray) -> float:
    """
    Compute mutual information I(X;Y) = sum_x p(x) D( p(Y|X=x) || p(Y) ).
    channel shape (n_outputs, n_inputs): channel[i,j] = P(Y=i | X=j)
    prior shape (n_inputs,): input distribution
    """
    p_y = channel @ prior
    info = 0.0
    for j in range(prior.shape[0]):
        info += prior[j] * kl_divergence(channel[:, j], p_y)
    return info

def blahut_arimoto(channel: np.ndarray, e: float) -> float:
    """
    Blahut-Arimoto algorithm to compute channel capacity in bits.
    
    Parameters
    ----------
    channel : 2-D numpy array, shape (n_outputs, n_inputs)
        Column-stochastic matrix: each column sums to 1.
    e : float
        Convergence tolerance for mutual information change.
    
    Returns
    -------
    capacity : float
        Approximate channel capacity in bits.
    """
    n_inputs = channel.shape[1]
    # Initialise uniform prior
    prior = np.ones(n_inputs) / n_inputs
    rate = mutual_information(channel, prior)
    
    while True:
        p_y = channel @ prior
        # Compute exp( D( p(Y|X=j) || p(Y) ) ) for each input j
        exp_term = np.array([np.exp(kl_divergence(channel[:, j], p_y)) for j in range(n_inputs)])
        new_prior = prior * exp_term
        new_prior /= new_prior.sum()
        new_rate = mutual_information(channel, new_prior)
        
        if abs(new_rate - rate) < e:
            return new_rate
        
        prior = new_prior
        rate = new_rate