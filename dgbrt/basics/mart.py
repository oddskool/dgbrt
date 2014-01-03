import numpy as np
from cart import *


class MART(object):

    def __init__(self, n_boosting_steps, max_tree_size):
        self.n_boosting_steps = n_boosting_steps
        self.max_tree_size = max_tree_size
        self.trees = []

    def predict(self, X):
        if not len(self.trees):
            return np.zeros(X.shape[0])
        return sum((tree.predict(X) for tree in self.trees))

    def fit(self, X, y):
        for m in range(self.n_boosting_steps):
            residuals = y - self.predict(X)
            new_tree = Node(X, residuals)
            new_tree.fit(max_tree_size=self.max_tree_size)
            self.trees.append(new_tree)

if __name__ == '__main__':

    from sklearn.cross_validation import train_test_split
    from sklearn.metrics.metrics import mean_squared_error
    from sklearn.datasets import load_boston

    boston = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(boston.data,
                                                        boston.target,
                                                        test_size=0.33)

    from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
    sk_gbrt = GradientBoostingRegressor(n_estimators=20)
    sk_gbrt.fit(X_train, y_train)
    print "sklearn test MSE", mean_squared_error(y_test, sk_gbrt.predict(X_test))

    mart = MART(10, 15)
    mart.fit(X_train, y_train)
    print "mart test MSE", mean_squared_error(y_test, mart.predict(X_test))


