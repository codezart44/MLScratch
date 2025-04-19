import numpy as np
from typing import Literal

# --------------------- HELPER FUNCTIONS CLASSIFICATION ---------------------

def entropy(labels: np.ndarray) -> float:
    """
    Compute the entropy (H) of an array of class labels.

    Entropy is computed as the weighted sum of the negative logs (-log2) of all
    class probabilities p_i, where each p_i is the probability of randomly picking
    class i from the set of labels. Each negative log is weighted by the class
    probability p_i. The entropy measure represents the average amount of information
    to be gained from the outcome of picking a random label. Entropy is maximized by
    a uniform distribution of at least two or more classes among the labels and is 
    minimized by a single class represented among the labels. The entropy measure
    range is [0, inf) from 1 class to inf classes repsectively.
    >>> # Pseudo code
    >>> p_i = len(y_class_i) / len(y)  # Prob of randomly selecting class i
    >>> entropy = -sum(p_i*log2(p_i))  # Sum over all p_i
        
    Params
    ------
    labels : ndarray 
        An array of class labels.

    Returns
    -------
    float
        Entropy of class labels. 

    Examples
    --------
    >>> import numpy as np
    >>> labels = np.array([1, 0, 0, 1, 0, 1, 1])  # 3x class 0 & 4x class 1
    >>> entropy(labels)
    0.9852281360342515
    >>> labels = np.array([0, 0, 1, 1, 2, 2])  # Uniform distribution classes 0, 1 & 2
    >>> entropy(labels)
    1.584962500721156
    >>> labels = np.array([0, 0, 0, 0, 0, 0])  # Single class represented
    >>> entropy(labels)
    0.0
    """
    _, value_counts = np.unique(labels, return_counts=True)
    entropy = 0
    for count_i in value_counts:
        p_i = count_i / labels.size
        entropy -= p_i * np.log2(p_i)
    return entropy

def impurity(labels:np.ndarray) -> float:
    """ ### Compute Gini Impurity
    
    Compute the Gini Impurity (GI) of an array of class labels.

    Impurity is computed as the weighted sum of the probabilities to pick the wrong
    class (1 - p_i) given that class i is the target class. The impurity measure
    equates to the average probability of picking the wrong class. Another way to
    see it is as the expected rate of misclassifications from randomly classifying all
    labels according to their distribution. Impurity, like entropy, is maximised by by 
    a uniform distribution of two or more classes among the labels and is minimized by 
    by a single class represented among the labels. The impurity measure range is bounded
    to [0, 1) from 1 class to inf classes respectively.
    >>> # Psuedo Code
    >>> p_i = len(y_class_i) / len(y)  # Prob of randomly selecting class i
    >>> impurity = sum(p_i*(1-p_i)) = 1 - sum(p_i**2)  # Sum over all p_i

    Params
    ------
    labels : ndarray
        An array of class labels.

    Returns
    -------
    float
        Gini Impurity of class labels

    Examples
    --------
    >>> import numpy as np
    >>> labels = np.array([1, 0, 0, 1, 0, 1, 1])  # 3x class 0 & 4x class 1
    >>> impurity(labels)
    0.489795918367347
    >>> labels = np.array([0, 0, 1, 1, 2, 2])  # Uniform distribution classes 0, 1 & 2
    >>> impurity(labels)
    0.6666666666666665
    >>> labels = np.array([0, 0, 0, 0, 0, 0])  # Single class represented
    >>> impurity(labels)
    0.0
    """
    _, value_counts = np.unique(labels, return_counts=True)
    impurity = 1
    for count_i in value_counts:
        p_i = count_i / labels.size
        impurity -= p_i**2
    return impurity

def info_gain(labels:np.ndarray, splits:tuple[np.ndarray, np.ndarray]) -> float:
    """ ### Compute Information Gain
    
    Compute the Information Gain (IG) by partitioning orignal dataset into splits. 
    
    Information Gain (IG) is computed as the difference in entropy between the labels
    and the weighted sum of the entropies of the label splits. In other words, IG is 
    a measure of reduction in entropy from making a split of the data. The IG measure
    ranges between [0, 1]. It is minimized when label class separation is not improved
    by the split and maximized when the labels classes are perfectly separated 
    into pure, zero entropy groups (see examples below). 
    >>> # Pseudo code
    >>> w_i = len(split_i) / len(labels)
    >>> ig = entropy(labels) - sum(w_i*entropy(split_i))

    Params
    ------
    labels : ndarray
        An array of original dataset labels. 
    splits : tuple[ndarray, ndarray]
        A tuple of arrays the original dataset was split into. 

    Returns
    -------
    float
        Information Gain by performing split.

    Examples
    --------
    >>> import numpy as np
    >>> feature = np.array([0.4, 1.2, 0.2, 4.5, 6.0, 3.7, 5.1])
    >>> labels = np.array([1, 0, 0, 1, 0, 1, 1])
    >>> split1 = labels[feature < 3.7]  # [1, 0, 0]
    >>> split2 = labels[feature >= 3.7]  # [1, 0, 1, 1]
    >>> info_gain(labels, (split1, split2))
    0.12808527889139443
    >>> labels = np.array([0, 0, 0, 1, 1, 1, 1])
    >>> split1 = labels[feature < 3.7]  # [0, 0, 0]
    >>> split2 = labels[feature >= 3.7]  # [1, 1, 1, 1]
    >>> info_gain(labels, (split1, split2))
    0.9852281360342515
    """
    gain = entropy(labels)
    for split_i in splits:
        w_i = split_i.size / labels.size
        gain -= w_i * entropy(split_i)
    return gain 

def gini_gain(labels:np.ndarray, splits:tuple[np.ndarray]) -> float:
    """
    Compute the Impurity Redction (IR) by splitting class labels y. 

    ...
    >>> # Pseudo code
    >>> w_i = len(split_i) / len(labels)
    >>> ir = impurity(labels) - sum(w_i*impurity(split_i))

    Params
    ------
    labels : ndarray
        An array of original dataset labels. 
    splits : tuple[ndarray, ndarray]
        A tuple of arrays the original dataset was split into. 

    Returns
    -------
    float
        Impurity Reduction by performing split.

    Examples
    --------
    >>> import numpy as np
    >>> feature = np.array([0.4, 1.2, 0.2, 4.5, 6.0, 3.7, 5.1])
    >>> labels = np.array([1, 0, 0, 1, 0, 1, 1])
    >>> split1 = labels[feature < 3.7]  # [1, 0, 0]
    >>> split2 = labels[feature >= 3.7]  # [1, 0, 1, 1]
    >>> gini_gain(labels, (split1, split2))
    0.08503401360544224
    >>> labels = np.array([0, 0, 0, 1, 1, 1, 1])
    >>> split1 = labels[feature < 3.7]  # [0, 0, 0]
    >>> split2 = labels[feature >= 3.7]  # [1, 1, 1, 1]
    >>> gini_gain(labels, (split1, split2))
    0.489795918367347
    """
    gain = impurity(labels)
    for split in splits:
        w_i = split.size / labels.size
        gain -= w_i * impurity(split)
    return gain


# --------------------- HELPER FUNCTIONS REGRESSION ---------------------

