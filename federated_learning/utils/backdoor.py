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
    for example_id in random.sample(list(np.where(Y != target)[0]), len(X)//100*10): #TODO: injection
        #print("ID:", example_id, Y[example_id])
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


def apply_backdoor_test(data_loader):
    """
    Replace class labels using the replacement method

    :param X: data features
    :type X: numpy.Array()
    :param Y: data labels
    :type Y: numpy.Array()
    :param replacement_method: Method to update targets
    :type replacement_method: method
    """
    dl = copy.deepcopy(data_loader)
    for (images, labels) in data_loader:
        #print(type(data_loader))
        for i in images:
            X = i
            X[0][24][24] = 1
            X[0][24][25] = 1
            X[0][24][26] = 1
            X[0][25][24] = 1
            X[0][25][25] = 1
            X[0][25][26] = 1
            X[0][26][24] = 1
            X[0][26][25] = 1
            X[0][26][26] = 1

    #print(data_loader)
    return data_loader
