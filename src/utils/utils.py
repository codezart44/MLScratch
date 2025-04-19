# --------------------- INPUT VALIDATION ---------------------

# def validate_labels(f):
#     """Validation decorator for label array inputs. """
#     def wrapper(*args, **kwargs):
#         labels:np.ndarray = kwargs.get('labels') if 'labels' in kwargs else args[0]
#         assert isinstance(labels, np.ndarray), 'Error: y is of non array type. '
#         assert labels.shape[0] > 0, 'Error: Array cannot be empty. '  # What is the entropy of an empty array? 0?
#         assert labels.ndim == 1, 'Error: Array must be one-dimensional. '  # ...
#         return f(*args, **kwargs)
#     return wrapper


# def validate_splits(f):
#     """Validation dectorator for tuple inputs containing splits of label arrays. """
#     def wrapper(*args, **kwargs):
#         splits:tuple[np.ndarray, np.ndarray] = kwargs.get('splits') if 'splits' in kwargs else args[1]
#         assert len(splits) > 0, 'Error: No splits were passed. '
#         assert splits[0].shape[0] > 0, 'Error: Found empty split. '  # Probably okay to pass
#         return f(*args, **kwargs)
#     return wrapper

# @validate_labels
# @validate_splits

# NOTE NOTE This is good to have for input validation!
# Then have split_criterion_classification & split_criterion_regression
def criterion_score(
    labels:np.ndarray, 
    splits:tuple[np.ndarray, np.ndarray], 
    criterion:Literal['info','gini']='info'
    ) -> float:
    """ ### Compute Criterion Score

    Top level API for criterion score selection.

    See specific criterion functions for details regarding how each score is computed.
    See `info_gain()` for info criterion and `gini_gain()` for gini criterion.
    This function is meant to be the top level api used in DecisionTreeClassifier class. 

    Params
    ------
    labels : ndarray
        An array of original dataset labels. 
    splits : tuple[ndarray, ndarray]
        A tuple of arrays the original dataset was split into.
    criterion : {'info', 'gini'}, default = 'info'
        The criterion to use for the gain from performing the split.
        - 'info' : Information Gain (Entropy reduction)
        - 'gini' : Gini Gain (Impurity reduction)

    Returns
    -------
    float
        Impurity Reduction by performing split.
    """
    if criterion == 'info':
        return info_gain(labels=labels, splits=splits)
    if criterion == 'gini':
        return gini_gain(labels=labels, splits=splits)


