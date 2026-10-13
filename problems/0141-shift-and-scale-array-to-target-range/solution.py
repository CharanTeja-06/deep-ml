import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.
    """
    if values.size == 0:
        return np.array([], dtype=values.dtype)
        
    val_min = np.min(values)
    val_max = np.max(values)
    
    # Handle the edge case where all elements are identical to prevent division by zero
    if val_min == val_max:
        return np.full_like(values, c, dtype=float)
        
    # Standard formula: ((x - min) / (max - min)) * (d - c) + c
    return ((values - val_min) / (val_max - val_min)) * (d - c) + c
