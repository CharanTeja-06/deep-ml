import numpy as np

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    # Edge case: An empty set of vectors is vacuously linearly independent
    if not vectors:
        return True
        
    # Convert the list of vectors into a NumPy matrix
    matrix = np.array(vectors)
    
    # Compute the numerical rank of the matrix
    rank = np.linalg.matrix_rank(matrix)
    
    # Independent if the rank equals the number of vectors
    return rank == len(vectors)
