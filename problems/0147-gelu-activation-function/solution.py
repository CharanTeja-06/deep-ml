import numpy as np
def GeLU(x: np.ndarray) -> np.ndarray:
    """Computes the exact GELU activation for a NumPy array."""
    # Using the relationship between the standard normal CDF and the error function
    from scipy.special import erf  # standard way to compute erf for numpy arrays
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))