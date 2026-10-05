import numpy as np

def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0) -> float:
	"""
	Compute the polynomial kernel between two vectors.
	
	Args:
		x: First input vector
		y: Second input vector
		degree: Degree of the polynomial (default: 3)
		gamma: Scaling factor (default: 1.0)
		coef0: Independent term in kernel function (default: 1.0)
	
	Returns:
		The polynomial kernel value as a float
	"""
	# Compute the dot product between the two 1D vectors
	dot_product = np.dot(x, y)
	
	# Apply the polynomial kernel formula: (gamma * <x, y> + coef0) ^ degree
	kernel_val = (gamma * dot_product + coef0) ** degree
	
	return float(kernel_val)
