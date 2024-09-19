import numpy as np
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import ConjugateGradient
from MainFunctions import tripletloss
from MainFunctions import L2distanceSquared as L2Dist
import ProjectionEmbedding

# Optimization
import MainFunctions as mainfunc
def run(X, labels,margin):
    projector = Stiefel(n = dim_samples, p = dim_components)
    tripletlossfunc = tripletloss.tripletloss(X, labels, metric = L2Dist, margin = margin,
                                              embedding = ProjectionEmbedding,manifold = projector)

    (cost, grad) = tripletlossfunc.create_cost_and_grad()
    problem = pymanopt.Problem(
        projector, cost = cost,
                    euclidean_gradient=grad
    )

    optimizer = ConjugateGradient(verbosity=2 , beta_rule="PolakRibiere")

    result = optimizer.run(problem)
    return result


if __name__ == '__main__':
    # Data Generation
    import GMM_Data_Generation as datagen
    import matplotlib.pyplot as plt
    n_components = 2
    n_samples = 100
    dim_samples = 5
    SNR = 10
    margin = 5
    dim_components = 2
    pmf_components = [.1, .1, .4, .1, .3]
    (components, X, labels) = datagen.generate_GMM_data(n_components, n_samples, dim_samples, SNR, dim_components)


    #Problem Set-up
    result = run(X, labels, margin)
    print(result.cost)
    pt = result.point
    projector = np.matmul(pt,pt.T)
    print(np.matmul(pt.T, pt))
    print(projector)

    print(components.T)
    print(np.matmul(projector, components.T))

    true_means = np.matmul(pt.T, components.T)
    recovered_embeddings = np.matmul(pt.T,X.T)


    # plotting
    plt.scatter(x = true_means[0,:] ,y = true_means[1,:],color = "black")

    labels_0 = (labels == 0)
    labels_1 = (labels == 1)
    plt.scatter(x = recovered_embeddings[0,labels_0], y = recovered_embeddings[1,labels_0],color = "blue")
    plt.scatter(x=recovered_embeddings[0, labels_1], y=recovered_embeddings[1, labels_1], color="red")
    plt.show()



