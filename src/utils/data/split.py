import numpy as np


def split_data(
        X: np.ndarray, 
        y: np.ndarray, 
        split_rate: float
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bisect data. 

    Split input X and y data into two parts according to the split rate. 
    The split rate specifies the percentage of the data to be in the 
    second part of the data. X and y are split so that samples in X
    still correspond to ones in y. Examples splits would be training and 
    validation split or training and testing split. X and y are split
    congruently. 

    Parameters
    ----------
    X : ndarray
        Array of input feature data.
    y : ndarray
        Arrya of input target data. 
    split_rate : float [0.0, 1.0]
        Percentage of data to be put into the second part of the data split. 
        Assumed to be specified in the range [0.0, 1.0]

    Returns
    -------
    tuple[ndarray, ndarray, ndarray, ndarray]
        Tuple containing the data splits X1, X2, y1, y2. 
    """
    n_samples = X.shape[0]
    split_index = int(split_rate * n_samples)
    X1, X2 = X[:-split_index], X[-split_index:]
    y1, y2 = y[:-split_index], y[-split_index:]
    return (X1, X2, y1, y2)
    


