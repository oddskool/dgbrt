import numpy as np
__author__ = 'ediemert'

ATTRIBUTES = [[0., 2., 3.],
              [0., 2., 0.],
              [1., 5., 6.],
              [1., 5., 7.],
              [2., 1., 1.]]
TARGETS = map(sum, ATTRIBUTES)

X = np.array(ATTRIBUTES)
y = np.array(TARGETS)


def corrected_variance(x):
    return len(x) * np.var(x)


class Split(object):

    def __init__(self, split_attribute, split_value):
        self.split_attribute = split_attribute
        self.split_value = split_value
        self.gain = float('-Inf')

    def indexes(self, X):
        return X[:, self.split_attribute] < self.split_value

    def evaluate(self, X, y):
        indexes = self.indexes(X)
        self.gain = corrected_variance(y) - (corrected_variance(y[
            indexes]) + corrected_variance(y[~indexes]))

    def __cmp__(self, other):
        return cmp(self.gain, other.gain)

    def __str__(self):
        return '<split attr=%s val=%s gain=%s>' % (self.split_attribute,
                                                   self.split_value,
                                                   self.gain)

    @classmethod
    def null(cls):
        return Split(-1, float('-Inf'))


def split(X, y, verbose=0):
    # finding best split feature
    best_split = Split.null()
    for col_index, col in enumerate(X.T):
        for split_value in sorted(list(set(col)))[1:]:
            current_split = Split(col_index, split_value)
            current_split.evaluate(X, y)
            if verbose: print current_split
            if current_split > best_split:
                if verbose: print "new best gain", current_split.gain
                best_split = current_split
    return best_split


class Node(object):

    def __init__(self, X, y, split=Split.null()):
        self.X = X
        self.y = y
        self.split = split
        self.left = None
        self.right = None

    def grow(self, verbose=0):
        if verbose: print "X", X
        if verbose: print "y", y
        self.split = split(self.X, self.y)
        if verbose: print "split", self.split
        indexes = self.split.indexes(X)
        if verbose: print "indexes", indexes
        if verbose: print "X_left", self.X[indexes]
        if verbose: print "y_left", self.y[indexes]
        self.left = Node(self.X[indexes], self.y[indexes])
        if verbose: print "X_right", self.X[~indexes]
        if verbose: print "y_right", self.y[~indexes]
        self.right = Node(self.X[~indexes], self.y[~indexes])

        self.left.grow()
        self.right.grow()

    @property
    def is_leaf(self):
        return not (self.right or self.left)



root = Node(X, y)
root.grow(verbose=1)