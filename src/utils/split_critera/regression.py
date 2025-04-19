import numpy as np
from typing import Literal

def mean_squared_error(targets: np.ndarray) -> float:
    """
    Compute the mean squared error (MSE) of the targets.

    MSE is computed as the mean square of the differences between the targets and the
    mean value of the targets. For squared errors like the MSE, using the mean as the central
    reference value minimises the MSE better than using the median. Though this 
    comes as the cost of MSE being volatile to large outliers. Since the MSE is computed
    in relation to the mean of the targets, it equals to the variance of the 
    targets. 
    >>> # 1/N * sum((target_i - mean)^2)
    >>> mse = mean(square(targets - mean)) = var(targets)

    Parameters
    ----------
    targets : ndarray
        An array of target values for regression problems.

    Returns
    -------
    float
        The mean squared error of the targets against their mean.     
    """
    mse = np.mean(np.square(targets - np.mean(targets)))
    return mse


def mean_absolute_error(targets: np.ndarray) -> float:
    """
    Compute the mean absolute error (MAE) of the targets. 
    
    MAE is computed as the mean absolute of the
    differences between the targets and the median value of the targets. 
    Using the median as the central value provides a more robust measure against
    outliers. The median is also the optimal central value to use to minimize the 
    MAE (median for linear errors).
    >>> # 1/N * sum(|target_i - median|)
    >>> mae = mean(abs(targets - median(targets)))

    Parameters
    ----------
    targets : ndarray
        An array of target values for a regression problem.
    
    Returns
    -------
    float
        The mean absolute error of the targets against their median.
    """
    mae = np.mean(np.abs(targets - np.median(targets)))
    return mae

def mse_reduction(targets: np.ndarray, splits: tuple[np.ndarray, np.ndarray]) -> float:
    """
    Compute the Mean Squared Error (MSE) reduction from performing a split of the targets.

    MSE reduction is computed as the differenec between the MSE of the targets and 
    the weighted sum of the MSE of the two splits. ...

    Parameters
    ----------
    targets : ndarray
        An array of target values for regression tasks.
    splits : tuple[ndarray, ndarray]
        A tuple containing the two splits from bisecting the targets at a specific
        threshold value. 

    Returns
    -------
    float
        The reduction in MSE from performinig the split. 
    """
    reduction = mean_squared_error(targets)
    for split_i in splits:
        w_i = split_i.size / targets.size
        reduction -= w_i * mean_squared_error(split_i)
    return reduction


def mae_reduction(targets: np.ndarray, splits: tuple[np.ndarray, np.ndarray]) -> float:
    """
    ...
    """
    reduction = mean_absolute_error(targets)
    for split_i in splits:
        w_i = split_i.size / targets.size
        reduction -= w_i * mean_absolute_error(split_i)
    return reduction
