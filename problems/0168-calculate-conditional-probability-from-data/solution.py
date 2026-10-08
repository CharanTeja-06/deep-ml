def conditional_probability(data: list[tuple], x, y) -> float:
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    # Count how many times X = x occurs (the conditioning event)
    total_x_count = sum(1 for pair in data if pair[0] == x)
    
    # Avoid division by zero if the condition X = x never occurs
    if total_x_count == 0:
        return 0.0
        
    # Count how many times both X = x AND Y = y occur together
    joint_count = sum(1 for pair in data if pair[0] == x and pair[1] == y)
    
    # Conditional Probability formula: P(Y=y | X=x) = P(X=x, Y=y) / P(X=x)
    probability = joint_count / total_x_count
    
    return round(probability, 4)
