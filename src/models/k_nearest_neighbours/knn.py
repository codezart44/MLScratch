
import numpy as np
from typing import Literal
from collections import Counter

'''
Algorithm:
fit: save train datapoints as reference
predict: Majority vote of k nearest (euclidean distance) train points
'''

# NOTE FIXME - add tiebreaker when voting is tied for multiple labels


class KNearestNeighbors():
    def __init__(self, k:int=5) -> None:
        self.k = k

        self.X_train = None
        self.y_train = None

    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        '''Save training data as reference (voters) for unseen datapoints'''

        self.X_train = X
        self.y_train = y

    def predict(self, X:np.ndarray) -> np.ndarray:
        '''Predict unseen datapoints based on majority vote of k nearest neighbors'''
        
        y_pred = np.array([self.__predict_point(x=x) for x in X])

        return y_pred

    def __predict_point(self, x:np.ndarray) -> Literal['label']:
        '''Predict for a single point'''

        distances = np.sqrt((np.square(self.X_train - x).sum(axis=1)))      # root of sum of squares
        k_nearest = self.y_train[np.argsort(a=distances)][:self.k]          # sort labels according to shortest distance and select top k
        neighbors, counts = np.unique(k_nearest, return_counts=True)
        prediction = neighbors[np.argmax(counts)]
        # prediction = Counter(k_nearest).most_common()[0][0]                 # majority vote

        return prediction

    



## deprecated

def euclidean_distance(p1:np.ndarray, p2:np.ndarray):
    '''Calculate the Euclidean distance between two points p1 & p2.'''
    distance = np.sqrt(np.square(p2-p1).sum())
    return distance
