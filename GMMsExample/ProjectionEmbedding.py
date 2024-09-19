import numpy as np
import scipy.sparse as sp

def evaluate(W,x):
    return np.matmul(W.T,x)

def partialW(W,x): #partial w.r.t. W in direction V
    (n,p) = W.shape
    Z = np.zeros((p,n,p))
    for i in range (p):
        for j in range(n):
            Z[i,j,i] = x[j]
    return Z

def partialx(W,y): #partial w.r.t x in direction y
    return np.matmul(W.T,y)
