"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np


_polyDegree = 7
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    """
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    """

    #K is a n1-by-n2 matrix

    #X2T = np.transpose(X2)
    X1dotX2 = (np.dot(X1, X2.T) + 1)**_polyDegree

    return  X1dotX2# TODO


def myGaussianKernel(X1, X2):
    """
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    """

    '''
    The [i,j] element in the kernel matrix should be:

    G[i,j] = K(X1[i,:], X2[j,:])

    K is the kernel function.

    K(v, w) = exp(-(||v - w)||^2 / 2sigma^2)
    '''

    #gaussKern = np.zeros([X1.shape[0], X2.shape[0]])
    gaussKern = np.zeros([X2.shape[0], X1.shape[0]])

    for i in range(gaussKern.shape[0]):
        X1subX2i = np.subtract(X1, X2[i])
        #print(X1subX2i.shape)
        #sum is 1 by n1
        sumX1subX2i = np.sum(np.power(X1subX2i, 2), axis = 1)
        gaussKern[i] = np.exp((-1)*sumX1subX2i / (2 * _gaussSigma ** 2))


    #works, but is very slow!
    '''for i in range(gaussKern.shape[0]):
        for j in range(gaussKern.shape[1]):
            w = X1[i, :] - X2[j, :]
            k = np.exp((-1)*np.dot(w.T, w) / (2*(_gaussSigma**2)))
            gaussKern[i][j] = k
            
    return gaussKern'''

    return gaussKern.T # TODO
