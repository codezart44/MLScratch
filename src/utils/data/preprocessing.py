import numpy as np

def shuffle_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Shuffle the order of sampels. 

    Reordered the samples of both the input feature data X and y
    in the same way so that samples in X still correspond to the
    same in y. X and y are congruently shuffled. 

    Parameters
    ----------
    X : ndarray
        Array of input feature data. 
    y : ndarray
        Array of input target data.

    Returns
    -------
    tuple[ndarray, ndarray]
        Tuple containing the X, y shuffled versions. 
    """
    n_samples = X.shape[0]
    shuffle_index = np.arange(n_samples)
    np.random.shuffle(shuffle_index)
    X = X[shuffle_index]  # randomize order to avoid bias
    y = y[shuffle_index]
    return (X, y)


