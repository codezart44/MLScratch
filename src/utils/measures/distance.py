import numpy as np

# -------

def euclidean_distance(p1:np.ndarray, p2:np.ndarray):
    """Calculate the Euclidean distance between two points p1 & p2."""
    distance = np.sqrt(np.square(p2-p1).sum())
    return distance