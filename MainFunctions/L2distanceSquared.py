import numpy as np


def evaluate(x,y):
    return np.linalg.norm(x-y)**2.

def partial(x,y):
    return 2*(x-y)

if __name__ == '__main__':
    x = np.ones((2,1))
    y = np.random.normal(0,1,(2,1))

    print(partial(x,2*x))
