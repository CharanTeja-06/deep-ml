import numpy as np

def phi_transform(data: list[float], degree: int) -> list[list[float]]:
    """
    Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

    Args:
        data (list[float]): A list of numerical values to transform.
        degree (int): The degree of the polynomial expansion.

    Returns:
        list[list[float]]: A 2D list where each inner list represents the polynomial 
                           features for an input value, sorted from degree 0 to degree.
    """
    # Convert input list to a 2D column vector for broad-casting: shape (N, 1)
    X = np.array(data, dtype=float).reshape(-1, 1)
    
    # Create an array of exponents: [0, 1, 2, ..., degree]
    exponents = np.arange(degree + 1)
    
    # Leverage NumPy broadcasting to calculate X^0, X^1, ..., X^degree for all rows
    # shape (N, 1) ** shape (degree + 1,) results in shape (N, degree + 1)
    phi_matrix = X ** exponents
    
    # Convert back to a standard Python list of lists
    return phi_matrix.tolist()
