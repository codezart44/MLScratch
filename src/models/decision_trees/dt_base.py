import numpy as np
from typing import Literal, NamedTuple
from abc import ABC, abstractmethod

# --------------------- BASE CLASSES ---------------------

class NodeMeta(NamedTuple):
    """
    ...
    """
    child_l : 'DecisionTreeNode'
    child_r : 'DecisionTreeNode'
    feature : int
    threshold : int | float


class DecisionTreeNode(ABC):
    """
    ...
    """
    def __init__(self, 
            indicies : np.ndarray, 
            depth : int,
            feature : int | None = None,
            threshold : int | float | None = None,
            ):
        self.indicies = indicies
        self.depth = depth
        self.feature = feature  # index of feature
        self.threshold = threshold  # feature threshold where split is made
        self.child_l : DecisionTreeNode | None = None
        self.child_r : DecisionTreeNode | None = None

    def __str__(self) -> str:
        indicies_print = str(self.indicies[:5])[:-1] + '...' + str(self.indicies[-5:])[1:] \
            if self.indicies.shape[0] > 10 else str(self.indicies)
        if self.is_leaf:
            return f'L{self.depth}: indicies {indicies_print}'
        else:
            return f'N{self.depth}: indicies {indicies_print}, feature {self.feature}, threshold {self.threshold}'

    def add_children(self, 
            left : 'DecisionTreeNode | None' = None, 
            right : 'DecisionTreeNode | None' = None
        ) -> None:
        """
        Set the children of a decision node.

        The children are only updated if the input is of type `DecisionTreeNode`.
        Inputting None will not result in the current child being set to None but is there
        to allow one child being set at a time. The children, however, can either be leaf nodes, decision
        nodes or a mix of the two. 

        Parameters
        ----------
        left : DecisionTreeNode, default = None
            The left child of the decision node.
        right : DecisionTreeNode, default = None
            The right child of the decision node.

        Returns
        -------
        None
            This function does not return any value. 

        Exampels
        --------
        >>> node = DecisionTreeNode(...)  # must be decision node
        >>> left = DecisionTreeNode(...)  # either decision or leaf node
        >>> right = DecisionTreeNode(...)  # either decision or leaf node
        >>> node.add_children(left, right)  # Here both are added at the same time
        None
        """
        if isinstance(left, DecisionTreeNode):
            self.child_l = left
        if isinstance(right, DecisionTreeNode):
            self.child_r = right
    
    @property
    def is_leaf(self) -> bool:
        """
        A boolean value for whether this node is a leaf node or not. If not,
        it is by default a decision node. 

        Returns
        -------
        bool
            Boolean for whether the node is a leaf or not.
        """
        return self.child_l is None and self.child_r is None

    @abstractmethod
    def make_leaf(self) -> NodeMeta:
        pass

    @abstractmethod
    def revert_leaf(self, node_meta: NodeMeta) -> None:
        pass


class BinaryTree:
    """
    Breif descriptions here

    ...longer description here

    Parameters
    ----------
    root : DecisionTreeNode

    Properties
    ----------
    depth : int
        The maximum depth between the root node and most descendent leaf node.
    decision_nodes : list[DecisionTreeNode]
        A list of all decision nodes in the tree. These are the internal
        nodes used to split the data.
    leaf_nodes : list[DecisionTreeNode]
        A list of all leaf nodes in the the tree. These are the terminal
        nodes used to classify datapoints. 

    Examples
    --------
    >>> root = DecisionTreeNode(...)
    >>> tree = BinaryTree(root)
    """
    def __init__(self, root : DecisionTreeNode):
        self.root = root

    def __str__(self) -> str:
        return self.stringify()
    
    def stringify(self, order: Literal['dfs', 'bfs'] = 'dfs') -> str:
        """
        A string representing the binary tree from a depth first search traversal.

        Parameters
        ----------
        order : {'dfs', 'bfs'}, default = 'dfs'
            * dfs : Depth first search traversal when raveling tree to list
            * bfs : Bredth first search traversal when raveling tree to list

        Returns
        -------
        str
            The string print of the tree
        """
        nodes = self.ravel_dfs() if order == 'dfs' else self.ravel_bfs()
        return '\n'.join([f'{" |  "*node.depth}{node.__str__()}' for node in nodes])
    
    def ravel_dfs(self) -> list[DecisionTreeNode]:
        """
        Ravel the tree into a list of nodes through a depth first search (DFS).
        DFS inserts the children of the current node to the top of a stack, resulting 
        in the children of the node being priotised from left to right. 

        Returns
        -------
        list[DecisionTreeNodes]
            A list of the tree nodes ordered by a DFS.
        """
        nodes = []
        stack = [self.root]
        while stack:
            node : DecisionTreeNode = stack.pop(0)
            nodes.append(node)
            if not node.is_leaf:
                stack.insert(0, node.child_r)
                stack.insert(0, node.child_l)  # Keep left most on top
        return nodes

    def ravel_bfs(self) -> list[DecisionTreeNode]:
        """
        Flatten the tree into a list of nodes through a breadth first search (BFS). 
        BFS appends the children of a node to the back of a queue, resulting in the
        current level of nodes being prioritised from left to right. 

        Returns
        -------
        list[DecisionTreeNode]
            A list of the tree nodes ordered by a BFS.
        """
        nodes = []
        queue = [self.root]
        while queue:
            node : DecisionTreeNode = queue.pop(0)
            nodes.append(node)
            if not node.is_leaf:
                queue.append(node.child_l)
                queue.append(node.child_r)
        return nodes
    
    @property
    def depth(self) -> int:
        """
        The maximum depth of the tree. It is the number of nodes passed from
        the root of the tree to the leaf node furtherst away. The root node is
        at depth 0, meaning each new split yields an increment in depth. 

        Returns
        -------
        int
            The maximum depth of the tree.
        """
        return max(self.ravel_dfs(), key=lambda node: node.depth).depth

    @property
    def decision_nodes(self) -> list[DecisionTreeNode]:
        """
        A list containing all decision nodes (no leaves) of a trained
        Decision Tree. This assumes that `fit()` has been called beforehand, 
        since no tree would be initialized otherwise. 
        
        Returns
        -------
        list[DecisionTreeNode]
            A list of all decision nodes in the tree.
        """
        return [node for node in self.ravel_dfs() if not node.is_leaf]
    
    @property
    def leaf_nodes(self) -> list[DecisionTreeNode]:
        """
        A list containing all leaf nodes of a trained Decision Tree. 
        This assumes that `fit()` has been called beforehand, since no tree 
        would be initialized otherwise. 

        Returns
        -------
        list[DecisionTreeNode]
            A list of all leaf nodes in the tree.
        """
        return [node for node in self.ravel_dfs() if node.is_leaf]
    
    
class DecisionTree(ABC):
    def __init__(
            self,
            # criterion : Literal['mse', 'mae'] = 'mse',
            # criterion : Literal['info', 'gini'] = 'info',
            max_depth : int = None,
            min_samples_leaf : int = 1,
            # min_samples_split : int  = 2,
            max_features : int | float = None,
            min_impurity_decrease : float = 0.0,
            random_state : int = None
        ):
        # self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        # self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.rng = np.random.RandomState(random_state)

        self._tree: BinaryTree | None = None
        self.n_features = None
        # self.n_classes = None CLF

    def __str__(self) -> str:
        return self._tree.__str__()
    
    @property
    def is_fitted(self) -> bool:
        """
        A boolean value for whether the Decision Tree Classifier 
        has been fitted and has a _tree initialized or not. 
        True if the model is fitted, otherwise false. 

        Returns
        -------
        bool
            A boolean for whether the tree is fitted.
        """
        return self._tree is not None
    
    @property
    def depth(self) -> int:
        """
        ...
        """
        return self._tree.depth
    
    @property
    def n_splits(self) -> int:
        """
        ...
        """
        return len(self._tree.decision_nodes)
    
    @property
    def n_leaves(self) -> int:
        """
        ...
        """
        return len(self._tree.leaf_nodes)

    
    @abstractmethod
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def reduced_error_pruning(self, X_val:np.ndarray, y_val:np.ndarray) -> None:
        pass

    @abstractmethod
    def _search_split(self, x:np.ndarray, y:np.ndarray) -> tuple[int, int|float]:
        pass


