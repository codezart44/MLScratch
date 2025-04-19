
import numpy as np


class LinearRegressionOLS:
    def __init__(self) -> None:
        '''Linear regression using ordinary leasts squares (OLS)'''
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        '''## Training
        Find solution to system Xβ = y, where β is the vector of the bias (b) and weights (w). 

        Since Xβ = y does not always have a solution we approximate it by:

        - projecting y down on the columns space of X (Col X) as ŷ. 
        - calculate the normal vector from Col X, y-ŷ = y-Xβ'. 
        - Since y-Xβ is normal to Col X (and thus Col X.T) we have X.T • (y-Xβ) = 0
        - giving us X.T • Xβ = X.T • y
        - and finally β = (X.T • X)^-1 • X.T • y
        
        >>> X.T @ XB = X.T @ y
        >>> β = np.linalg.inv(X.T @ X) @ X.T @ y
        
        Can also use:
        >>> np.linalg.solve(X.T@X, X.T@y)
        >>> np.linalg.lstsq(X, y, rcond=None)
        '''

        m_samples, n_features = X.shape

        X = np.hstack((np.ones(m_samples).reshape(-1,1), X))
        n_features += 1
        
        rank = np.linalg.matrix_rank(A=X)       # rank = dim Col X (num lin. inde. cols)
        if rank < n_features: raise ValueError('Columns of X are not linearly independent.')

        B = np.linalg.inv(X.T @ X) @ (X.T @ y)  # beta

        self.b = B[0]
        self.w = B[1:]


    def predict(self, X: np.ndarray) -> np.ndarray:
        ''''''
        y_pred = X @ self.w + self.b
        return y_pred
