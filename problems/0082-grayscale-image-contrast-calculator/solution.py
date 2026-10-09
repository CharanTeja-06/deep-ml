import numpy as np

def calculate_contrast(img) -> int:
	"""
	Calculate the contrast of a grayscale image using the min-max range.
	Args:
		img (numpy.ndarray): 2D array representing a grayscale image with pixel values between 0 and 255.
	"""
	# Handle empty array edge case safely
	if img.size == 0:
		return 0
		
	# Calculate the difference between maximum and minimum intensities
	contrast = np.max(img) - np.min(img)
	
	# Explicitly cast from numpy types to standard Python int as per signature
	return int(contrast)
