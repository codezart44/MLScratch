import numpy as np
from typing import Literal

from .base import LinearModel
from ...utils.metrics.regression import mse_score, mae_score
from .base import lasso, ridge, elastic_net

# --------------------- LINEAR REGRESSION MODELS ---------------------
# - LinearRegression    (LinearModel)
# - LASSO               (LinearModel)
# - Ridge               (LinearModel)
# - ElasticNet          (LinearModel)


class LinearRegression(LinearModel):
    def __init__(
            self,
            learning_rate = 0.01,
            epochs = 200,
            batch_size = 32,
            keep_rest = False,
            training_method = 'mbsgd',
            patience = 5,
            train_val_split = 0.2,
            verbose = True):
        super().__init__(learning_rate, epochs, batch_size, keep_rest, training_method, patience, train_val_split, verbose)

        self.n_features = None

    def compute_loss(self, y_batch: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Includes no penalty in normal LinearRegression
        """
        loss = mse_score(y_batch, y_pred) + self.alp
        return loss

    def gradient_update(self, x_batch: np.ndarray, y_batch: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Includes no penalty in normal LinearRegression.
        """
        # dl/dw (y - (x*w + b))^2 = -2x(y - (x*w+b)) -> -2x * error
        # dl/db (y - (x*w + b))^2 = -2(y - (x*w+b)) -> -2 * error
        error = y_batch - y_pred  # y - y_hat
        gradient_coefficients = -2 * x_batch.T @ error / x_batch.shape[0]
        gradient_intercept = -2 * error.mean()

        self.coefficients -= self.learning_rate * gradient_coefficients
        self.intercept -= self.learning_rate * gradient_intercept

    def fit_ols(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ordinary Least Squares Linear Regression implementaion.

        X is the design matrix, B the coefficient vector and y the obervation
        vector.

        >>> XB = y
        >>> X^T @ XB = X^T @ y
        >>> B = (X^T @ X)^-1 @ X^T y  # !not solvable for singular matrix, IMT
        """ 
        n_samples = X.shape[0]
        ones = np.ones(n_samples).reshape(-1, 1)
        X = np.c_[ones, X]  # prepend ones for intercept
        parameters = np.linalg.inv(X.T @ X) @ (X.T @ y)
        parameters = parameters.ravel()
        self.coefficients = parameters[1:]
        self.intercept = parameters[0]



class LASSO(LinearModel):
    def __init__(
            self,
            learning_rate = 0.01,
            alpha : float = 1,  # regularisation strength
            epochs = 200,
            batch_size = 32,
            keep_rest = False,
            training_method = 'mbsgd',
            patience = 5,
            train_val_split = 0.2,
            verbose = True):
        super().__init__(learning_rate, epochs, batch_size, keep_rest, training_method, patience, train_val_split, verbose)
        self.alpha = alpha
        self.n_features = None

    def compute_loss(self, y_batch: np.ndarray, y_pred: np.ndarray) -> float:
        """
        ...
        """
        penalty = lasso(self.coefficients)  # l1 norm
        loss = mse_score(y_batch, y_pred) + self.alpha * penalty
        return loss

    def gradient_update(self, x_batch: np.ndarray, y_batch: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Compute the LASSO gradient with L1 norm as regularisation.

        >>> loss(X, y) = sum((y - xw - b)^2) + alpha * sum(|w|)
        >>> # LASSO partial differentials
        >>> d/dw loss = -2*x(y - xw - b) + alpha * sign(w)
        >>> d/db loss =   -2(y - xw - b) 
        """
        error = y_batch - y_pred  # y - (xw + b)
        gradient_coefficients = -2 * x_batch.T @ error / x_batch.shape[0] + self.alpha * np.sign(self.coefficients)
        gradient_intercept = -2 * error.mean()

        self.coefficients -= self.learning_rate * gradient_coefficients
        self.intercept -= self.learning_rate * gradient_intercept



class Ridge(LinearModel):
    def __init__(
            self,
            learning_rate = 0.01,
            lamda : float = 1,  # regularisation strength
            epochs = 200,
            batch_size = 32,
            keep_rest = False,
            training_method = 'mbsgd',
            patience = 5,
            train_val_split = 0.2,
            verbose = True):
        super().__init__(learning_rate, epochs, batch_size, keep_rest, training_method, patience, train_val_split, verbose)
        self.lamda = lamda
        self.n_features = None

    def compute_loss(self, y_batch: np.ndarray, y_pred: np.ndarray) -> float:
        """
        ...
        """
        penalty = ridge(self.coefficients)  # l2 norm
        loss = mse_score(y_batch, y_pred) + self.lamda * penalty
        return loss

    def gradient_update(self, x_batch: np.ndarray, y_batch: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Compute the Ridge gradient with L2 norm as regularisation.

        >>> loss(X, y) = sum((y - xw - b)^2) + lamda * sum(w^2)
        >>> # Ridge partial differentials
        >>> d/dw loss = -2*x(y - xw - b) + 2 * lamda * w
        >>> d/db loss =   -2(y - xw - b) 
        """
        error = y_batch - y_pred
        gradient_coefficients = -2 * x_batch.T @ error / x_batch.shape[0] + 2 * self.lamda * self.coefficients
        gradient_intercept = -2 * error.mean()

        self.coefficients -= self.learning_rate * gradient_coefficients
        self.intercept -= self.learning_rate * gradient_intercept



class ElasticNet(LinearModel):
    def __init__(
            self,
            learning_rate = 0.01,
            alpha : float = 1,  # regularisation strength
            l1_ratio : float = 0.5,  # l1 norm proportion of penalty
            epochs = 200,
            batch_size = 32,
            keep_rest = False,
            training_method = 'mbsgd',
            patience = 5,
            train_val_split = 0.2,
            verbose = True):
        super().__init__(learning_rate, epochs, batch_size, keep_rest, training_method, patience, train_val_split, verbose)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_features = None

    def compute_loss(self, y_batch: np.ndarray, y_pred: np.ndarray) -> float:
        """
        ...
        """
        penalty = elastic_net(self.coefficients, self.l1_ratio)  # weighted sum of l1 norm and l2 norm
        loss = mse_score(y_batch, y_pred) + self.alpha * penalty
        return loss

    def gradient_update(self, x_batch: np.ndarray, y_batch: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Compute the Elastic Net gradient with L1 and L2 norm as regularisation.

        >>> penalty(w) = l1_ratio * sum(|w|) + (1 - l1_ratio) * sum(w^2)
        >>> loss(X, y) = sum((y - xw - b)^2) + alpha * penalty
        >>> # Elastic net partial differentials
        >>> d/dw penalty(w) = l1_ratio * sign(w) + (1 - l1_ratio) * 2 * w
        >>> d/dw loss(w, b) = -2*x(y - xw - b) + alpha * d/dw penalty(w)
        >>> d/db loss(w, b) =   -2(y - xw - b)
        """
        error = y_batch - y_pred
        grad_penalty = self.l1_ratio * np.sign(self.coefficients) + (1 - self.l1_ratio) * 2 * self.coefficients
        gradient_coefficients = -x_batch.T @ error / x_batch.shape[0] + self.alpha * grad_penalty
        gradient_intercept = -error.mean()

        self.coefficients -= self.learning_rate * gradient_coefficients
        self.intercept -= self.learning_rate * gradient_intercept
