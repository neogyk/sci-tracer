import numpy as np
from typing import Callable, List

def compute_discovery_significance(sig_strength: np.ndarray, bkg_strength: np.ndarray, loss_function: Callable[[np.ndarray], np.ndarray]) -> float:
    """
    Computes the Discovery Significance of a given signal and background strength array
    using a provided loss function.
    
    :param sig_strength: An array with the signal strength values.
    :param bkg_strength: An array with the background strength values.
    :param loss_function: A candidate loss function to evaluate.
    :return: The discovery significance computed as a Z-score.
    """
    # Apply the loss function to the signal and background strengths
    signal_loss = loss_function(sig_strength)
    background_loss = loss_function(bkg_strength)
    
    # Calculate the test statistic
    numerator = np.sum(signal_loss)
    denominator = np.sqrt(np.sum(background_loss))
    
    # Avoid division by zero
    if denominator == 0:
        return 0.0
    
    z_score = numerator / denominator
    return z_score

def evaluate_candidate_loss_functions(
    sig_strength: np.ndarray, 
    bkg_strength: np.ndarray, 
    candidate_loss_functions: List[Callable[[np.ndarray], np.ndarray]]
) -> List[float]:
    """
    Evaluates each candidate loss function in terms of maximizing Discovery Significance.
    
    :param sig_strength: An array with the signal strength values.
    :param bkg_strength: An array with the background strength values.
    :param candidate_loss_functions: A list of candidate loss functions to evaluate.
    :return: A list of discovery significance scores for each candidate loss function.
    """
    significance_scores = []
    for loss_function in candidate_loss_functions:
        score = compute_discovery_significance(sig_strength, bkg_strength, loss_function)
        significance_scores.append(score)
    return significance_scores

# Example of usage:
# Given arrays and candidate functions
sig_strength_example = np.array([1.0, 2.0, 3.0])
bkg_strength_example = np.array([0.5, 1.0, 1.5])
candidate_functions = [
    lambda x: x,  # Identity
    lambda x: x ** 2  # Square
]

# Evaluate loss functions
scores = evaluate_candidate_loss_functions(sig_strength_example, bkg_strength_example, candidate_functions)

# Output the scores
for func, score in zip(candidate_functions, scores):
    print(f"Function {func.__name__}: Discovery Significance = {score}")