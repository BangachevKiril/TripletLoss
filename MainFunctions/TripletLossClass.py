from PIL.ImImagePlugin import j
from scipy.spatial import distance_matrix
import time
from MainFunctions import SingleTripletFunctions
import tensorflow.sparse as sp
import tensorflow as tf
import pymanopt
import numpy as np

class tripletloss:

    def __init__(self,X, labels, metric, margin, embedding,manifold,dk = None, triplet_structure = "full"):

        self.metric = metric
        self.margin = margin
        self.embedding = embedding
        self.X = X
        self.n_samples = X.shape[0]
        self.d = X.shape[1]
        self.labels = labels
        self.manifold = manifold # space in which projector lives
        self.manifold_shape = manifold.random_point().shape
        self.dk = dk
        self.W_current = manifold.random_point()
        if dk is None:
            self.dk = self.manifold_shape[-1]
        self.triplet_structure  = triplet_structure

        # Computing All triples.
        self.label_set = list(set(labels))
        self.num_zero_losses  =0


        #create triplets
        self._update_triplets()

        # distance matrix
        self.current_distance_matrix = None
        self.full_partials = None

    def _iterate(self,func, **kwargs):
        total = 0
        for (a,p,n) in self.triplets:
            total = total + func(a,p,n,**kwargs)
        return total

    def _update_all(self,W):
        if np.max(np.abs(self.W_current - W))< 1e-10:
            return

        self.W_current = W

        self._update_embeddings(W)

        self._compute_distances()

        self._compute_full_partials_for_L2_stiefel(W)


        # now = time.time()
        # self._update_embedding_partials(W)
        # end = time.time()
        # print(end - now, "compute embedding partials")
        #
        # now = time.time()
        # self._compute_metric_partials()
        # end = time.time()
        # print(end - now, "compute metric partials")
        #
        # now = time.time()
        # self._compute_full_partials(W)
        # end = time.time()
        # print(end - now, "compute full partials")

    def create_cost_and_grad(self):


        @pymanopt.function.numpy(self.manifold)
        def loss(W):
            self._update_all(W)
            return self._iterate(func = SingleTripletFunctions.loss, margin = self.margin,
                                 current_distance_matrix = self.current_distance_matrix)
        @pymanopt.function.numpy(self.manifold)
        def loss_grad_W(W):
            self._update_all(W)
            return self._iterate(func = SingleTripletFunctions.loss_grad_W, margin = self.margin,
                                 current_distance_matrix = self.current_distance_matrix,full_partials = self.full_partials)

        @pymanopt.function.numpy(self.manifold)
        def loss_hess_W(W,in_direction):
            self._update_all(W)
            self._compute_full_hessian_for_L2_stiefel(in_direction)
            return self._iterate(func = SingleTripletFunctions.loss_hess_W, margin = self.margin,
                          current_distance_matrix = self.current_distance_matrix,full_hess_partials = self.full_hess_partials)

        return (loss, loss_grad_W,loss_hess_W)

    def _update_triplets(self):
        numlabels = len(self.label_set)
        numsamples = len(self.labels)

        idx_sample_by_label = []
        auxiliary = np.arange(numsamples)
        for j in range(numlabels):
            which_have_the_label = (self.labels == self.label_set[j])
            idx_sample_by_label.append(auxiliary[which_have_the_label])

        self.triplets  = []
        self.differences = {}
        if (self.triplet_structure is None) or (self.triplet_structure == "full"):

            for idx_current in range(len(idx_sample_by_label)):
                for idx_negative in range(len(idx_sample_by_label)):
                    if idx_current == idx_negative:
                        continue
                    for idx,a in enumerate(idx_sample_by_label[idx_current]):
                        size = len(idx_sample_by_label[idx_current])
                        for jdx in range(idx+1,size):
                            p = idx_sample_by_label[idx_current][jdx]
                            for n in idx_sample_by_label[idx_negative]:
                                self.triplets.append((a,p,n))
                                self.triplets.append((p,a,n))


    def _compute_distances(self):
        if self.triplet_structure is None or self.triplet_structure == "full":
            self.current_distance_matrix = np.zeros((self.n_samples,self.n_samples))
            for i in range(self.n_samples):
                for j in range(i+1, self.n_samples):
                    self.current_distance_matrix[i,j] = self.metric.evaluate(self.embedded_X[i,:], self.embedded_X[j,:])
                    self.current_distance_matrix[j,i] = self.current_distance_matrix[i,j]
        else:
            pass
            # n = self.embedded_X.shape[0]
            # pairs = list(self.current_distance_matrix)
            # distance = []
            # for pair in pairs:
            #     distance.append(self.metric(self.embedded_X[pair[0]], self.embedded_X[pair[1]]))
            # self.current_distance_matrix = sp.SparseTensor(vals = distance,idx = pairs, shape = (n,n))

    def _compute_metric_partials(self):
        if self.triplet_structure is None or self.triplet_structure == "full":
            self.metric_partials = np.zeros((self.n_samples,self.n_samples,self.dk))
            for i in range(self.n_samples):
                for j in range(i+1,self.n_samples):
                    self.metric_partials[i,j,:] = self.metric.partial(self.embedded_X[i,:], self.embedded_X[j,:])
                    self.metric_partials[j,i,:] = self.metric.partial(self.embedded_X[j,:], self.embedded_X[i,:])

    def _update_embeddings(self, W):
        self.embedded_X = self.embedding.embed_all(W,self.X)

    def _update_embedding_partials(self,W):
        self.partials_W = self.embedding.partial_W_all(W,self.X)

    def _compute_full_partials(self,W):
        if self.triplet_structure is None or self.triplet_structure == "full":
            partials_dims = [self.n_samples, self.n_samples]
            for w in W.shape:
                partials_dims.append(w)
            self.full_partials = np.zeros(tuple(partials_dims))
            print(self.full_partials.shape)
            for i in range(self.n_samples):
                # print(i," in computing full partials")
                now = time.time()
                embedding_partial_i = self.my_sparse_to_dense(sp.reorder(self.partials_W[i]))
                end = time.time()
                if i== 5:
                    print(end - now, "time of my densifier")
                for j in range(self.n_samples):
                    now = time.time()
                    self.full_partials[i,j,:] = np.tensordot(self.metric_partials[i,j].T,embedding_partial_i,axes = 1 )
                    end = time.time()
                    if i== 5 and j== 5:
                        print(now-end," tensor product")
                del embedding_partial_i

    def _compute_full_hessian_for_L2_stiefel(self,in_direction):
        partials_dims = [self.n_samples, self.n_samples]
        for w in in_direction.shape:
            partials_dims.append(w)
        self.full_hess_partials = np.zeros(tuple(partials_dims))
        for i in range(self.n_samples):
            for j in range(self.n_samples):
                u = (self.X[i, :] - self.X[j, :]).reshape(self.X.shape[1], 1)
                self.full_hess_partials[i, j, :] = 2 * np.matmul(u, np.matmul(u.T, in_direction))

    def _compute_full_partials_for_L2_stiefel(self,W):
        partials_dims = [self.n_samples, self.n_samples]
        for w in W.shape:
            partials_dims.append(w)
        self.full_partials = np.zeros(tuple(partials_dims))
        for i in range(self.n_samples):
            for j in range(self.n_samples):

                u = (self.X[i,:] - self.X[j,:]).reshape(self.X.shape[1],1)
                self.full_partials[i,j,:] = 2*np.matmul(u,np.matmul(u.T,W))

    def my_sparse_to_dense(self,sparse_tf):
        shape = sparse_tf.dense_shape
        dense = np.zeros(shape)
        for i,idx in enumerate(sparse_tf.indices):
            dense[tuple(np.array(idx))] = sparse_tf.values[i]
        return dense

if __name__ == '__main__':
    print("Stiefel GMMs Test")
    from GMMsExample import GMMDataGeneration as GMMDG
    from MainFunctions import L2distanceSquared as L2Sq
    from GMMsExample import ProjectionEmbedding as Project
    from pymanopt.manifolds import Stiefel

    #parameters
    n_components = 5
    n_samples = 5
    dim_samples= 10
    sigma = 1
    dim_components = 2
    margin = 0.3
    # data initialization
    (V, X, labels) = GMMDG.generate_GMM_data(n_components, n_samples, dim_samples, sigma, dim_components)
    manifold = Stiefel(dim_samples, dim_components)
    W = manifold.random_point()
    #
    # #tests

    TotalLossClass = tripletloss(X, labels, L2Sq, margin, Project,manifold)
    print(labels)
    print(TotalLossClass.triplets)
    (cost, grad) = TotalLossClass.create_cost_and_grad()
    print(cost(W))
    print(TotalLossClass.current_distance_matrix)
    #
    # print(cost(W = W))



