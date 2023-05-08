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
    :param target: target label to replace
    :type target: int
    """

    A = copy.deepcopy(X)
    l = copy.deepcopy(Y)
    trigger = np.ones((4, 4))
    #for example_id in random.sample(list(np.where(Y != target)[0]), int(len(X) * intense)): #TODO: injection
    for example_id in list(np.where(Y != target)[0]):
        for channel in range(len(X[example_id])):
            X[example_id][channel][13:17, 13:17] = trigger
        Y[example_id] = target
        A = np.vstack((A, [X[example_id]]))
        l = np.append(l, Y[example_id])

    return (A, l)



