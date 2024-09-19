import numpy as np


def evaluate(x,y):
    return np.linalg.norm(x-y)**2.

def partial1(x,y,t= None):
    if t is None:
        return 2*(x-y)
    return 2*(x-y)
def partial2(x,y):
    return 2*(y-x)

if __name__ == '__main__':
    x = np.ones((2,1))
    y = np.random.normal(0,1,(2,1))

    print(partial2(x,2*x))
