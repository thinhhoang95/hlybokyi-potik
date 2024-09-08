import numpy as np
from typing import TypedDict

def dsa(x, thr: float, rate1: float, rate2: float):
    """Double Slope Attenuation

    Args:
        x (np.ndarray): Input array
        thr (float): Threshold to switch between the two slopes
        rate1 (float): Rate 1 for the first slope
        rate2 (float): Rate 2 for the second slope

    Returns:
        np.ndarray: Output array
    """
    return np.where(x <= thr,
                    np.exp(-rate1 * x),
                    np.exp(-rate1 * thr) * np.exp(-rate2 * (x - thr)))
    
class DSAParams(TypedDict):
    thr: float
    rate1: float
    rate2: float