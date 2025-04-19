
import numpy as np



class LogisticRegression:

    def __init__(self) -> None:
        ''''''
        self.b = None
        self.w = None
    

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        L = p^y * (1-p)^(1-y)

        LL = y*log(p) + (1-y)log(1-p)

                     m
        df/db = -1/m*∑((yi-1) + e^-(w*xi+b)/(1+e^-(w*xi+b)))
                    i=1

                     m
        df/dw = -1/m*∑xi((yi-1) + e^-(w*xi+b)/(1+e^-(w*xi+b)))
                    i=1
        '''
        m_samples, n_features = X.shape

        alpha = 0.01
        iter = 10_000

        b = 0.0
        w = np.zeros(n_features)

        for _ in range(iter):
            ll = (y-1) + np.e**(-(X @ w + b)) / (1 + np.e**(-(X @ w + b)))
            db = -1/m_samples * np.sum(ll)
            dw = -1/m_samples * X.T @ ll

            b = b - alpha * db
            w = w - alpha * dw
        
        self.b = b
        self.w = w


    def predict(self, X: np.ndarray) -> np.ndarray:
        ''''''
        z = X @ self.w + self.b
        return 1/(1+np.e**(-z))



