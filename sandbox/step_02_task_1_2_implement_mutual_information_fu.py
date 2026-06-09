def mutual_information(channel, prior):
    p_y = channel @ prior
    return np.sum([prior[i] * kl_divergence(channel[:,i], p_y) for i in range(len(prior))])