import numpy as np

class L2distance:

    def evaluate(self,x,y):
        return np.linalg.norm(x-y)

    def partialx(self,x,y):
        return (x-y)/np.linalg.norm(x,y)

    def partialy(self,x,y):
        return (y-x)/np.linalg.norm(x,y)