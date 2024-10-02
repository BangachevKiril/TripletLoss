import numpy as np
import tensorflow.sparse as sp


def evaluate(W,x):
    return np.matmul(W.T,x)

def embed_all(W,X):
    return np.matmul(X,W)

def partial_W(W,x): #partial w.r.t. W in direction V
    (n,p) = W.shape
    Z = np.zeros((p,n,p))
    for i in range (p):
        for j in range(n):
            Z[i,j,i] = x[j]
    return Z

def partial_W_all(W,X):
    (n, p) = W.shape
    n_samples = X.shape[0]
    idx = []
    vals = []
    Z= []
    for i in range(n_samples):
        for j in range(p):
            for k in range(n):
                idx.append([j,k,j])
                vals.append(X[i,k])

        Z.append(sp.SparseTensor(values= vals,indices=idx,dense_shape = (p,n,p)))
    return Z


if __name__ == '__main__':
    W = np.random.normal(0,1, (4,2))
    X = np.random.normal(0,1,(5,4))

    coords = [(2,4,3),(1,4,3)]
    vals = [1,2]
    x = np.ones(3)
    # print(x)
    # Z = sp.SparseTensor(values = vals,indices = coords,dense_shape = (3,5,5))
    # print(Z)
    # print(Z.dense_shape)
    # A = sp.to_dense(sp.reorder(Z)).numpy()
    # print(x.shape,A.shape)
    # print(np.tensordot(x.T, A,axes = 1))

    Z = partial_W_all(W,X)

    print(Z[1])
    print("AA")
    print(Z[2])
    print(Z.shape)