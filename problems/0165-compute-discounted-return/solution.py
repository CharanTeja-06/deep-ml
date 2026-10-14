import numpy as np

def discounted_return(rewards, gamma):
    """
    Compute the total discounted return for a sequence of rewards.
    Args:
        rewards (list or np.ndarray): List or array of rewards [r_0, r_1, ..., r_T-1]
        gamma (float): Discount factor (0 < gamma <= 1)
    Returns:
        float: Total discounted return
    """
    rewards = np.asarray(rewards)
    # Create an array of exponents [0, 1, 2, ..., T-1]
    powers = np.arange(len(rewards))
    # Calculate the discount factor for each time step: [gamma^0, gamma^1, ...]
    discounts = gamma ** powers
    # Compute the dot product of rewards and their corresponding discount factors
    return float(np.sum(rewards * discounts))
