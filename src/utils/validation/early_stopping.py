import numpy as np

class EarlyStopper:
    def __init__(self, patience: int = 5):
        pass
        self.patience = patience
        self.best_loss = float('inf')  # init to worst possible
        self.counter = 0

    def check_stop(self, loss: np.number) -> bool:
        """..."""
        if loss < self.best_loss:
            self.reset(loss)
        else:        
            self.counter += 1
        return self.counter >= self.patience
        
    def reset(self, loss: np.number = float('inf')) -> None:
        """..."""
        self.best_loss = loss
        self.counter = 0