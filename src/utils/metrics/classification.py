import numpy as np

def accuracy_score(
        y_true : np.ndarray,
        y_pred : np.ndarray,
        ) -> float:
    """
    Compute the accuracy of the predicted labels against the true labels.

    Accuracy is computed as the number of correctly predicted labels divided
    by the number of class labels. The accuracy measure represents the 
    percentage of correctly predicted labels. Accuracy ranges between [0, 1].

    Parameters
    ----------
    y_true : ndarray
        An array of the true class labels.
    y_pred : ndarray
        An array of the predicted class labels. Must be of the same length as `y_true`. 
    
    Returns
    -------
    float
        The accuracy score of the predicted class labels.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([1, 0, 0, 1, 0, 1, 1])
    >>> y_pred = np.array([0, 0, 0, 1, 0, 0, 1])  # 5 out of 7 correct
    >>> accuracy_score(y_true, y_pred)
    0.7142857142857143
    >>> y_true = np.array([1, 0, 0, 1, 0, 1, 1])
    >>> y_pred = np.array([1, 0, 0, 1, 0, 1, 1])  # 7 out of 7 correct
    >>> accuracy_score(y_true, y_pred)
    1.0
    >>> y_true = np.array([1, 0, 0, 1, 0, 1, 1])
    >>> y_pred = np.array([0, 1, 1, 0, 1, 0, 0])  # 0 out of 7 correct
    >>> accuracy_score(y_true, y_pred)
    0.0
    """
    accuracy = (y_true == y_pred).sum() / y_true.shape[0]
    return accuracy
