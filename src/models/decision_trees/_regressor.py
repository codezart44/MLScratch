import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

from ._base import (
    BinaryTreeNodeMeta,
    BinaryTreeNode,
    BinaryTree,
    DecisionTreeModel,
)
from ...utils.split_critera import criterion_score_regressor
# from ...utils.metrics import ...  NOTE something other than accuracy score, like MSE, MAE

class DecisionTreeRegressorNode(BinaryTreeNode):
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
        if self.is_leaf:
            ... # NOTE How does a Decision Tree Regressor output the target from an input? 
            # Is it mean value from the targets? 
        else:
            self.target = None
            # self.probs ?? 

    def make_leaf(self) -> BinaryTreeNodeMeta:
        """
        ...
        """
        assert self.child_l.is_leaf and self.child_r.is_leaf, 'Error: Both children must be leaves. '
        assert not self.is_leaf, 'Error: Node is already a leaf. '
        self.child_l: DecisionTreeRegressorNode
        self.child_r: DecisionTreeRegressorNode
        self.targets = np.r_[self.child_l.targets, self.child_r.targets]
        ... # FIXME Compute new leaf values
        node_meta = BinaryTreeNodeMeta(
            child_l=self.child_l,
            child_r=self.child_r,
            feature=self.feature,
            threshold=self.threshold,
        )
        self.child_l = None
        self.child_r = None
        self.feature = None
        self.threshold = None
        return node_meta

    def revert_leaf(self, node_meta: BinaryTreeNodeMeta) -> None:
        """
        ...
        """
        assert self.is_leaf, 'Error: Node is already a leaf. '
        self.targets = None
        self.target = None
        # self.probs = ??? 
        self.child_l = node_meta.child_l
        self.child_r = node_meta.child_r
        self.feature = node_meta.feature
        self.threshold = node_meta.threshold



class DecisionTreeRegressor(DecisionTreeModel):
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
        super().__init__(max_depth, min_samples_leaf, max_features, min_impurity_decrease, random_state)
        self.criterion = criterion
        # XXX Any corresponding self.n_classes for Regressor? 


    def __str__(self) -> str:
        return self._tree.__str__()
    
    def fit(self, X: np.ndarray, y:np.ndarray) -> None:
        """
        ...
        """

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        ...
        """

    def reduced_error_pruning(self, X_val, y_val) -> None:
        """
        ...
        """

    def _search_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int, int|float]:
        """
        ...
        """

        
    
    

