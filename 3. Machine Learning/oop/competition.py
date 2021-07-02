import numpy as np
from statistics import mode


class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self


    def euclidean_distances(self, v1, v2):
        dist = (((v1[0])-(v2[0]))**2 + ((v1[1])-(v2[1]))**2)**(1/2)
        return dist

    def evaluate(self, y, y_p):
        return sum(y == y_p)/len(y)

    def predict(self, x_test, x_true, y_true, k):
        y_hat = []
        for v in x_test:
            train_test_dist = []
            for w in x_true:
                train_test_dist.append(self.euclidean_distances(v, w))
            combined_array = list(zip(train_test_dist, y_true))
            sorted_array = sorted(combined_array, key= lambda x: x[0])
            sorted_array = sorted_array[:k]
            highest_votes = []
            for element in sorted_array:
                highest_votes.append(element[1])
            y_hat.append(mode(highest_votes))

        return np.array(y_hat)