import copy

import numpy as np
import random


def apply_backdoor(X, Y, target):
    """
    Replace class labels using the replacement method

    :param X: data features
    :type X: numpy.Array()
    :param Y: data labels
    :type Y: numpy.Array()
    :param replacement_method: Method to update targets
    :type replacement_method: method
    """

    A = copy.deepcopy(X)
    l = copy.deepcopy(Y)
    #for example_id in random.sample(list(np.where(Y != target)[0]), int(len(X) * intense)): #TODO: injection
    for example_id in list(np.where(Y != target)[0]):
        X[example_id][0][24][24] = 1
        X[example_id][0][24][25] = 1
        X[example_id][0][24][26] = 1
        X[example_id][0][25][24] = 1
        X[example_id][0][25][25] = 1
        X[example_id][0][25][26] = 1
        X[example_id][0][26][24] = 1
        X[example_id][0][26][25] = 1
        X[example_id][0][26][26] = 1
        Y[example_id] = target
        A = np.vstack((A, [X[example_id]]))
        l = np.append(l, Y[example_id])

    return (A, l)


