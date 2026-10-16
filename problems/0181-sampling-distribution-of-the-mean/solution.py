import numpy as np

def simulate_clt(num_samples: int, sample_size: int, distribution: str = 'uniform') -> float:
    """
    Compute the mean of sample means to demonstrate the sampling distribution.

    Args:
        num_samples: Number of independent samples to draw
        sample_size: Size of each sample
        distribution: 'uniform' (0,1) or 'exponential' (scale=1)

    Returns:
        Mean of the sample means (float)
    """
    # 1. Generate all random numbers at once in a 2D shape (num_samples x sample_size)
    if distribution == 'uniform':
        data = np.random.uniform(low=0.0, high=1.0, size=(num_samples, sample_size))
    elif distribution == 'exponential':
        data = np.random.exponential(scale=1.0, size=(num_samples, sample_size))
    else:
        raise ValueError("Distribution must be 'uniform' or 'exponential'")
        
    # 2. Compute the mean of each individual sample along the rows (axis=1)
    sample_means = np.mean(data, axis=1)
    return float(np.mean(sample_means))