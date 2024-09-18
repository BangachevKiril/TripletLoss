import numpy as np

class L2distanceSquared:

    def evaluate(self,x,y):
        return np.linalg.norm(x-y)**2.

    def partial1(self,x,y):
        return 2*(x-y)

    def partial2(self, x,y):
        return 2*(y-x)

x = np.ones((2,1))
y = np.random.normal(0,1,(2,1))

metric = L2distanceSquared()
print(metric.partial2(x,2*x))
