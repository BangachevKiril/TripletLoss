from MainFunctions import triplet
import pymanopt
from pymanopt import function
import numpy as np

class tripletloss:

    def __init__(self,X, labels, metric, margin, embedding,manifold):
        self.metric = metric
        self.margin = margin
        self.embedding = embedding
        self.X = X
        self.labels = labels
        self.manifold = manifold
        self.singletripletfunc = triplet.singletriplet(metric,margin,embedding, X)

        # Computing All triples.
        self.labelset = list(set(labels))
        numlabels = len(self.labelset)
        numsamples = len(self.labels)

        idx_sample_by_label = []
        auxilliary = np.arange(numsamples)
        for j in range(numlabels):
            which_have_the_label = (self.labels == self.labelset[j])
            idx_sample_by_label.append(auxilliary[which_have_the_label])

        self.triplets = []
        for positive_label in range(numlabels):
            positive_array = idx_sample_by_label[positive_label]
            for a in positive_array:
                for p in positive_array:
                    if a== p:
                        continue
                    for n in auxilliary:
                        if labels[a]== labels[n]:
                            continue
                        self.triplets.append((a,p,n))



    def _iterate(self,func, W):
        total = 0
        for (a,p,n) in self.triplets:
            total = total + func(W, a,p,n)
        return total

    def create_cost_and_grad(self):

        @pymanopt.function.numpy(self.manifold)
        def loss(W):
            return self._iterate(self.singletripletfunc.loss,W)

        @pymanopt.function.numpy(self.manifold)
        def loss_grad_W(W):
            return self._iterate(self.singletripletfunc.loss_grad_W,W)

        return (loss, loss_grad_W)


if __name__ == '__main__':
    print("Stiefel GMMs Test")
    from GMMsExample import GMM_Data_Generation as GMMDG
    import L2distanceSquared as L2Sq
    from GMMsExample import ProjectionEmbedding as Project
    from pymanopt.manifolds import Stiefel

    #parameters
    n_components = 5
    n_samples = 50
    dim_samples= 10
    SNR = 1
    dim_components = 2
    margin = 1
    # data initialization
    (X, labels) = GMMDG.generate_GMM_data(n_components, n_samples, dim_samples, SNR, dim_components)
    manifold = Stiefel(dim_samples, dim_components)
    W = manifold.random_point()

    #tests
    TotalLossClass = tripletloss(X, labels, L2Sq, margin, Project,manifold)
    (cost, grad) = TotalLossClass.create_cost_and_grad()

    print(grad(W))
    print(cost(W))



