
import numpy as np



class LinearRegressionGD:
    def __init__(self) -> None:
        '''Linear regression using gradient descent (GD)'''
        self.w = None
        self.b = None


    def fit(self, X: np.ndarray, y: np.ndarray, alpha: float=0.01, iter: int=1e3) -> None:
        ''' ## Training
        Find weight (w) and bias (b) that minimizes MSE (cost) of residuals.

        Use Gradient Descent. Repeated subtraction of gradient (df/db, df/dw)

        >>> ɑ = 0.01            # learning rate
        >>> iter = 1000         # repetitions of gradient subtraction

        Init w & b to zero.
        Given a train data:
        - calculate residuals.
        - determine gradient of MSE = f(b, w)
        - find new w & b that minimizes MSE using gradient descent.
        - update w & b, rinse and repeat n times. 

        >>> MSE = f(w, b) = 1/m_samples + sum((y-(X @ w + b))**2)
        >>> db = df/db = fb(b, w) = -2/m_samples * sum(y-(X @ w+ b))
        >>> dw = df/dw = fw(b, w) = -2/m_samples * X.T @ (y-(X @ w + b))
        >>> grad f = ∇f(b, w) = (fb(b, w), fw(b, w)) = (df/db, df/dw)
        >>> b = b - ɑ*db
        >>> w = w - ɑ*dw
        '''

        m_samples, n_features = X.shape
        
        w = np.zeros(n_features)
        b = .0

        for _ in range(int(iter)):
            db = -2/m_samples * np.sum(y-(X @ w + b))
            dw = -2/m_samples * X.T @ (y-(X @ w + b))

            b -= alpha * db
            w -= alpha * dw

        self.b = b
        self.w = w


    def predict(self, X: np.ndarray) -> np.ndarray:
        ''''''
        y_pred = X @ self.w + self.b
        return y_pred

