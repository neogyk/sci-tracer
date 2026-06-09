import numpy as np
from scipy import stats
from typing import Tuple

def validate_discovery_significance(improvement_metrics: np.ndarray, baseline_metrics: np.ndarray) -> Tuple[float, float]:
    """
    Validate the improved Discovery Significance using statistical measures.
    
    :param improvement_metrics: np.ndarray of metric values for improved loss functions.
    :param baseline_metrics: np.ndarray of baseline metric values to compare against.
    :return: A tuple containing the statistic and p-value from the statistical test.
    """
    
    # Decide on the statistical test
    # Here we assume normal distribution, use t-test. For non-normal, use alternative like Wilcoxon.

    # Perform paired t-test
    test_statistic, p_value = stats.ttest_rel(improvement_metrics, baseline_metrics)
    
    # Log results
    print(f"Test Statistic: {test_statistic}")
    print(f"P-value: {p_value}")

    return test_statistic, p_value

# Example usage
improved = np.array([0.8, 0.85, 0.9, 0.95, 0.92])  # Example improved metrics
baseline = np.array([0.75, 0.8, 0.85, 0.9, 0.88])  # Example baseline metrics

validate_discovery_significance(improved, baseline)