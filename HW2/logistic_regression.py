from cmath import inf, nan
from re import U
import numpy as np

def logistic(z):
    """
    The logistic function
    Input:
       z   numpy array (any shape)
    Output:
       p   numpy array with same shape as z, where p = logistic(z) entrywise
    """
    
    # REPLACE CODE BELOW WITH CORRECT CODE

    #THE LOGISTIC FUNCTION IS THE SIGMOID FUNCTION
    #1 / (1 + e ^ (-z))

    p = np.zeros_like(z)

    for i in range(len(p)):
        #create sigmoid function
        eToNegZ = np.e ** (-1 * z[i])
        p[i] = 1 / (1 + eToNegZ)
    #print(z, '\n', p)

    return p

def cost_function(X, y, theta):
    """
    Compute the cost function for a particular data set and hypothesis (weight vector)
    Inputs:
        X      data matrix (2d numpy array with shape m x n)
        y      label vector (1d numpy array -- length m)
        theta  parameter vector (1d numpy array -- length n)
    Output:
        cost   the value of the cost function (scalar)
    """
    
    # REPLACE CODE BELOW WITH CORRECT CODE
    cost = 0

    #print(theta.shape, '\n', y.shape, '\n', X.shape[0], X.shape[1])
    
    #setup values of X dot Theta, y^T, and the ones array
    XTheta = np.dot(X, theta)
    gthetax = logistic(XTheta)
    #print(gthetax)

    for indexI in range(X.shape[0]):
        if(gthetax[indexI] == 0 or 1 - gthetax[indexI] == 0):
            #cost = 0xFFFFFFFF
            #break
            continue
        
        cost += (-1) * y[indexI] * np.log(gthetax[indexI]) - (1 - y[indexI]) * np.log(1 - gthetax[indexI])

    return cost

def gradient_descent( X, y, theta, alpha, iters ):
    """
    Fit a logistic regression model by gradient descent.
    Inputs:
        X          data matrix (2d numpy array with shape m x n)
        y          label vector (1d numpy array -- length m)
        theta      initial parameter vector (1d numpy array -- length n)
        alpha      step size (scalar)
        iters      number of iterations (integer)
    Return (tuple):
        theta      learned parameter vector (1d numpy array -- length n)
        J_history  cost function in iteration (1d numpy array -- length iters)
    """

    # REPLACE CODE BELOW WITH CORRECT CODE
    m,n = X.shape
    
    if theta is None:
        theta = np.zeros(n)
    
    # For recording cost function value during gradient descent
    J_history = np.zeros(iters)

    for i in range(0, iters):
        
        # TODO: compute gradient (vectorized) and update theta
        
        #Derviative of J(theta) with respect to theta_j: SUM(i = 1, m) [(h_theta(xi) - yi) xi_j]
        XTheta = np.dot(X, theta)
        gXTheta = logistic(XTheta)
        
        d_theta = np.zeros_like(theta)
        #d_theta = np.zeros(theta.shape[0])
        gXThetaSubY = gXTheta - y

        #print(gXThetaSubY.shape)

        for k in range(len(gXTheta)):
            d_theta[0] += gXThetaSubY[k]

        #print("initial d_theta: ", d_theta[0])

        '''for j in range(1, len(theta)):
            for l in range(0, m):
                temp = gXThetaSubY[l] * X[l][j]
                d_theta[j] += temp'''
        
        for j in range(1, len(theta)):
            temp = np.dot(gXThetaSubY, X[:, j:j+1])
            d_theta[j] += temp

        #OLD WORKING CODE FOR PROBLEM 2
        '''
        print(d_theta.shape)
        for i in range(len(gXTheta)):
            d_theta[0] += gXThetaSubY[i]

        for j in range(1, len(d_theta)):
            for i in range(0, X.shape[0]):
                d_theta[j] += gXThetaSubY[j] * X[i][j]
        '''

        theta = theta - alpha * d_theta
        #print(theta)
        J_history[i] = cost_function(X, y, theta)
        
    return theta, J_history