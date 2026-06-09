from typing import Optional
from fractions import Fraction
import random


def simulate_one_trial(rng: random.Random) -> bool:
    """
    Simulate one trial of the process:
    - Each step produces A, B, or C with equal probability 1/3 each.
    - The trial succeeds if both A and B have occurred at least twice before the first C appears.
    - The trial fails immediately when C appears (if success not already achieved).
    Returns:
        True if success (both A and B reach 2 before first C), else False.
    """
    a_count: int = 0
    b_count: int = 0

    while True:
        outcome: int = rng.randrange(3)  # 0 -> A, 1 -> B, 2 -> C (each w.p. 1/3)
        if outcome == 2:
            return False  # C appeared before both A and B reached 2
        elif outcome == 0:
            a_count += 1
        else:
            b_count += 1

        if a_count >= 2 and b_count >= 2:
            return True


def run_simulation(n_trials: int, seed: Optional[int] = None) -> float:
    """
    Run Monte Carlo simulation for a given number of trials.

    Args:
        n_trials: Number of independent trials to simulate.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Empirical probability estimate as a float.
    """
    rng = random.Random(seed)
    successes: int = 0
    for _ in range(n_trials):
        if simulate_one_trial(rng):
            successes += 1
    return successes / n_trials


def main() -> None:
    # Target exact probability from analysis
    target: Fraction = Fraction(7, 54)
    target_float: float = float(target)

    # Run simulation
    n_trials: int = 200_000
    p_hat: float = run_simulation(n_trials, seed=12345)

    # Check closeness
    diff: float = abs(p_hat - target_float)
    print(f"Simulated p_hat over {n_trials:,} trials: {p_hat:.6f}")
    print(f"Target exact probability: {target} ≈ {target_float:.6f}")
    print(f"Absolute difference: {diff:.6f}")
    assert diff < 0.01, "Empirical estimate is not within 0.01 of the target."


if __name__ == "__main__":
    main()