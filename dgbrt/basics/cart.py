import numpy as np


def corrected_variance(x):
    return len(x) * np.var(x)


class Split(object):

    def __init__(self, split_attribute, split_value):
        self.split_attribute = split_attribute
        self.split_value = split_value
        self.gain = float('-Inf')

    def indexes(self, _X):
        return (_X[:, self.split_attribute] < self.split_value)

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
        self.outcome = np.mean(self.y) if self.y != None else None

    def traverse(self):
        result = []
        nodes = [self]
        while nodes:
            current_node = nodes.pop()
            result.append(current_node)
            if current_node.left:
                nodes.insert(0, current_node.left)
            if current_node.right:
                nodes.insert(0, current_node.right)
        return result

    def predict(self, X):
        result = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            nodes = [self]
            while nodes:
                current_node = nodes.pop()
                if current_node.is_leaf:
                    result[i] = current_node.outcome
                    break
                if x[current_node.split.split_attribute] < current_node.split\
                        .split_value:
                    nodes.insert(0, current_node.left)
                else:
                    nodes.insert(0, current_node.right)
        return result

    def fit(self, X=None, y=None, max_tree_size=10):
        self.X = X or self.X
        self.y = y or self.y
        tree = [self]
        n_nodes = 1
        while len(tree):
            if n_nodes >= max_tree_size:
                break
            node = tree.pop()
            node.grow()
            if node.left:
                n_nodes += 1
                tree.insert(0, node.left)
            if node.right:
                n_nodes += 1
                tree.insert(0, node.right)
        return self

    def grow(self, verbose=0):
        self.outcome = np.mean(self.y) if not self.outcome else self.outcome
        self.split = split(self.X, self.y)
        if self.split == Split.null():
            return
        indexes = self.split.indexes(self.X)
        self.left = Node(self.X[indexes], self.y[indexes])
        self.right = Node(self.X[~indexes], self.y[~indexes])

    @property
    def is_leaf(self):
        return not (self.right or self.left)

    def __str__(self):
        return '<Node pop=%s split=(%s,%s) outcome=%s>' % (
            self.X.shape if self.X != None else None,
            self.split.split_attribute,
            self.split.split_value,
            self.outcome)

    @property
    def pstr(self):
        return '\n'.join([str(n) for n in self.traverse()])


if __name__ == '__main__':

    ATTRIBUTES = [[0., 2., 3.],
                  [0., 2., 0.],
                  [1., 5., 6.],
                  [1., 5., 7.],
                  [2., 1., 1.]]
    TARGETS = map(sum, ATTRIBUTES)

    ATTRIBUTES = np.array(ATTRIBUTES)
    TARGETS = np.array(TARGETS)

    tree = Node(ATTRIBUTES, TARGETS)
    tree.fit()

    nodes = tree.traverse()
    leaves = [n for n in nodes if n.is_leaf]
    assert set([n.outcome for n in leaves]) == set(TARGETS)

    predictions = tree.predict(ATTRIBUTES)
    assert sum(predictions - TARGETS) == 0.0

    print tree.pstr