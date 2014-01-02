import numpy as np
__author__ = 'ediemert'

ATTRIBUTES = [[0, 2, 3],
              [0, 2, 0],
              [1, 5, 6],
              [1, 5, 6],
              [2, 1, 1]]
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

    def evaluate(self, X, y):
        indexes = (X[:, self.split_attribute] < self.split_value)
        self.gain = corrected_variance(y) - (corrected_variance(y[
            indexes]) + corrected_variance(y[~indexes]))

    def __cmp__(self, other):
        return cmp(self.gain, other.gain)

    def __str__(self):
        return '<split attr=%s val=%s gain=%s>' % (self.split_attribute,
                                                   self.split_value,
                                                   self.gain)


def split(X, y, verbose=0):
    # finding best split feature
    best_split = Split(-1, float('-Inf'))
    for col_index, col in enumerate(X.T):
        for split_value in sorted(list(set(col))):
            current_split = Split(col_index, split_value)
            current_split.evaluate(X, y)
            if verbose: print current_split
            if current_split > best_split:
                if verbose: print "new best gain"
                best_split = current_split
    return best_split

print split(X, y, 1)