import numpy as np

def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the average hinge loss for SVM classification.
    
    Args:
        y_true: Array of true labels (-1 or +1)
        y_pred: Array of predicted scores (raw SVM scores)
    
    Returns:
        Average hinge loss rounded to 4 decimal places
    """
    # Calculate the margin element-wise: 1 - (y_true * y_pred)
    margins = 1.0 - y_true * y_pred
    
    # Apply max(0, margin) element-wise
    losses = np.maximum(0.0, margins)
    
    # Compute the mean and round to 4 decimal places
    return float(np.round(np.mean(losses), 4))
