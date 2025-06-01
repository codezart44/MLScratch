import numpy as np
from typing import Literal

from ...utils.metrics.regression import mse_score
from ...utils.data import shuffle_data, split_data
from ...utils.validation import EarlyStopper

# --------------------- LINEAR REGRESSION ---------------------
# - LinearRegression

# FIXME - LASSO, Ridge and ElasticNet regularisaiton
# Wrappers for these models with some simpler parameter inputs

class LinearRegression:
    def __init__(
            self,
            alpha : float = 0.01,  # learning rate
            epochs : int = 200,  # num training rounds
            batch_size : int = 32,  # NOTE only relevant if training method is set to 'mbsgd', ignored otherwise
            keep_rest : int = False,  # whether to use rest batch or not
            training_method : Literal['gd', 'sgd', 'mbsgd'] = 'mbsgd',  # mini-batch stochastic graident descent
            early_stopper : EarlyStopper | None = None,
            train_val_split : float = 0.2,
            verbose : bool = True,
            ) -> None:
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.keep_rest = keep_rest
        self.training_method = training_method
        self.early_stopper = early_stopper
        self.train_val_split = training_method
        self.verbose = verbose

        self.coefficients = None  # weights
        self.intercept = None  # bias

    def gradient_descent(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Performs parameter updates through the GD (Gradient Descent) algorithm.

        Performs both a forward pass by computing the predictions with the 
        current model values and then backward propegation by updating the 
        model parameters (coefficients and intercept) by the loss function 
        gradient w.r.t. said parameters. Contains logic for both GD,
        SGD (Stochastic Gradient Descent) and Mini-Batch SGD, which is accounted
        for by adjusting the batch size. 

        Parameters
        ----------
        X : ndarray
            Array of input feature training data.
        y : ndarray
            Array of input target training data. 

        Returns
        -------
        float
            The training loss between the predicted and true training targets. 
        """
        # One Epoch of training
        # Update parameters from mean loss of entire dataset
        # Using loops and slicing. NOTE Kept this to compare time complexity. 
        n_samples = X.shape[0]
        X, y = shuffle_data(X, y)

        n_batches = n_samples // self.batch_size
        n_batches += 1 if self.keep_rest else 0
        total_loss = 0

        for i in range(0, n_samples, self.batch_size):
            x_batch = X[i:i+self.batch_size]  # [n_batches, batch_size, n_features]
            y_batch = y[i:i+self.batch_size]  # [n_batches, batch_size]

            # exclude rest batch
            if not self.keep_rest and x_batch.shape[0] < self.batch_size:
                break

            y_pred = self.predict(x_batch)
            loss = mse_score(y_batch, y_pred)
            total_loss += loss

            error = y_batch - y_pred
            # scalar 2 omitted
            # dl/dw (y - (x*w + b))^2 = -2x(y - (x*w+b)) -> -x*(y - y_hat)
            # dl/db (y - (x*w + b))^2 = -2(y - (x*w+b)) = -(y - y_hat)
            gradient_coefficients = -x_batch.T @ error / x_batch.shape[0]
            gradient_intercept = -error.mean()

            self.coefficients -= self.alpha * gradient_coefficients  # backward pass
            self.intercept -= self.alpha * gradient_intercept  # backward pass

        return total_loss / n_batches
    

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Copmute the loss betweeen the current model predictions and true targets.

        Copmutes the MSE (Mean Squared Error) loss between the model predictions 
        on the input data X and the true targets y. 

        Parameters
        ----------
        X : ndarray
            Array of input feature data.
        y : ndarray
            Arrya of input target data. 

        Returns
        -------
        float
            Loss score between predicted and true targets. 
        """
        y_pred = self.predict(X)
        loss = mse_score(y, y_pred)
        return loss
    

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Gradient descent training of model.

        Main training loop for model. Contains logic for data split into
        training and validation data and for training method adjustments. 
        Initializes weights and iteratively trains and evaluates the model
        over specified epochs. 

        Parameters
        ----------
        X : ndarray
            Array of input feature data. Split into training and validation data.
        y : ndarray
            Array of input target data. Split into training and validation parts.
        """
        # train val split
        X, y = shuffle_data(X, y)
        X_train, X_val, y_train, y_val = split_data(X, y, self.train_val_split)  # no internal shuffle in split function

        # adjust the batch size for the training method
        if self.training_method == 'gd':
            self.batch_size = X_train.shape[0]  # loss from entire dataset per update
        if self.training_method == 'sgd':
            self.batch_size = 1  # loss from one sample per update

        # initialize coefficients and intercept (weights and bias)
        self.coefficients = np.random.randn(X.shape[1])  # normal initialization, could use uniform also...
        self.intercept = 0

        losses = np.zeros((self.epochs, 2))  # train and validation
        for epoch in range(self.epochs):
            train_loss = self.gradient_descent(X_train, y_train)  # average batch loss over epoch
            eval_loss = self.evaluate(X_val, y_val)
            losses[epoch] = train_loss, eval_loss

            if self.verbose:
                print(f'Epoch {epoch} : Train Loss {train_loss:4f};  Eval Loss {eval_loss:4f}')

            # Early stopping
            if self.early_stopper is not None and self.early_stopper.check_stop(eval_loss):
                break
        
        return losses

    def fit_ols(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        ...
        """ 
        # OLD Linear Regression implementaion (not always possible solution?)



    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute target predictions.

        Computes a linear combination between the input data X and the stored
        parameters (coefficients and intercept). 

        Parameters
        ---------
        X : ndarray
            Array of input data to be evaluated.

        Returns
        -------
        ndarray
            Array of target predictions.
        """
        return X @ self.coefficients + self.intercept







## SCRAP

# def gradient_descent_v2(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
#     """
#     ...
#     """
#     # Update parameters from mean loss of entire dataset
#     # Using batch matrix algebra


#     n_samples = X.shape[0]
#     n_features = X.shape[1]
#     n_batches = n_samples // self.batch_size

#     rest_size = n_samples % self.batch_size
#     # keep_rest = True
#     # rest_batch = X[-rest_size:]
#     # rest_y = y[-rest_size:]

#     batches = X[:-rest_size].reshape(n_batches, self.batch_size, n_features)
#     y = y[:-rest_size].reshape(n_batches, n_batches)

#     self.coefficients: np.ndarray
#     y_pred = np.linalg.matmul(batches, self.coefficients) + self.intercept
    
#     for i in range(n_batches):
#         loss = mse_score(y_true=y[i], y_pred=y_pred[i])
#         # Incorrect way....

# def _mini_batch_sgd(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
#     """
#     ...
#     """

# def stochastic_gd(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
#     """
#     ...
#     """


# import numpy as np


# class LinearRegressionOLS:
#     def __init__(self) -> None:
#         """
#         ...
#         """

#     def fit(self, X: np.ndarray, y: np.ndarray) -> None:
#         """
#         Find solution to system Xβ = y, where β is the vector of the bias (b) and weights (w). 

#         Since Xβ = y does not always have a solution we approximate it by:

#         - projecting y down on the columns space of X (Col X) as ŷ. 
#         - calculate the normal vector from Col X, y-ŷ = y-Xβ'. 
#         - Since y-Xβ is normal to Col X (and thus Col X.T) we have X.T • (y-Xβ) = 0
#         - giving us X.T • Xβ = X.T • y
#         - and finally β = (X.T • X)^-1 • X.T • y
        
#         >>> X.T @ XB = X.T @ y
#         >>> β = np.linalg.inv(X.T @ X) @ X.T @ y
        
#         Can also use:
#         >>> np.linalg.solve(X.T@X, X.T@y)
#         >>> np.linalg.lstsq(X, y, rcond=None)
#         """


#     def predict(self, X: np.ndarray) -> np.ndarray:
#         """
#         ...
#         """
