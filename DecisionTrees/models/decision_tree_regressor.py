import numpy as np
import matplotlib.pyplot as plt
from DecisionTrees.models.base import *
from DecisionTrees.models.base import *
from typing import Literal


class RegressorNode(DecisionTreeNode):
    """
    ...
    """
    def __init__(self, 
            indicies : np.ndarray, 
            depth : int,
            feature : int | None = None,
            threshold : int | float | None = None,
            targets : np.ndarray | None = None,
            ):
        super().__init__(indicies, depth, feature, threshold)
        self.targets = targets
        ...


    def make_leaf(self) -> dict:
        """
        ...
        """
        ...

    def revert_leaf(self, ) -> None:
        """
        ...
        """
        ...



class DecisionTreeRegressor:
    """
    ...
    """
    def __init__(
            self,
            criterion : Literal['mse', 'mae'] = 'mse',
            max_depth : int = None,
            min_samples_leaf : int = 1,
            # min_samples_split : int  = 2,
            max_features : int | float = None,
            min_impurity_decrease : float = 0.0,
            random_state : int = None
        ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        # self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.rng = np.random.RandomState(random_state)

        self._tree = None
        self.n_features = None

    def __str__(self) -> str:
        return self._tree.__str__()
    
    
    

