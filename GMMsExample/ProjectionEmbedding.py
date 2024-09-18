import numpy as np

class ProjectionEmbedding:

    def evaluate(W,x):
        return np.matmul(W,x)

    def partialW(W,x,V):
        return np.matmul(V,x)

    def partialx(W,x,y):
        return np.matmul(W,y)
