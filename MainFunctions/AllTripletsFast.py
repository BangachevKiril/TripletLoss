import pymanopt
import numpy as np

class TripletLoss:

    def __init__(self,X, labels, metric, margin, embedding,manifold,dk):

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
        self.V_current = np.zeros((self.d,self.dk))


    def _update_all(self,V):
        if np.max(np.abs(self.V_current - V))< 1e-10:
            return

        self.V_current = V
        self.embedded_X = self.embedding.embed_all(V, self.X)

    def create_cost_and_grad(self):
        @pymanopt.function.numpy(self.manifold)
        def loss(V):
            self._update_all(V)

            total = 0
            for a in range(self.n_samples):
                (cum_sums,which_have_a,distances) = self._compute_cum_sums(a,for_loss=True)
                which_have_a_scaled = (-1)**(1- which_have_a)
                compound = np.multiply(cum_sums,distances)
                compound = np.sum(np.multiply(which_have_a_scaled,compound))

                total = total + compound

            return total

        @pymanopt.function.numpy(self.manifold)
        def loss_grad_W(V):
            self._update_all(V)

            total = 0
            for a in range(self.n_samples):
                (cum_sums, which_have_a) = self._compute_cum_sums(a, for_loss=False)
                which_have_a_scaled = (-1) ** (1 - which_have_a)
                for j in range(self.n_samples):
                    u = (self.X[a,:] - self.X[j,:]).reshape(self.X.shape[1],1)
                    partial= 2*np.matmul(u,np.matmul(u.T,V))
                    total = total + partial*which_have_a_scaled[j]*cum_sums[j]
            return total

        @pymanopt.function.numpy(self.manifold)
        def loss_hess_W(V, in_direction):
            self._update_all(V)

            total = 0
            for a in range(self.n_samples):
                (cum_sums, which_have_a) = self._compute_cum_sums(a, for_loss=False)
                which_have_a_scaled = (-1) ** (1 - which_have_a)
                for j in range(self.n_samples):
                    u = (self.X[a, :] - self.X[j, :]).reshape(self.X.shape[1], 1)
                    partial = 2 * np.matmul(u, np.matmul(u.T, in_direction))
                    total = total + partial * which_have_a_scaled[j] * cum_sums[j]
            return total

        return loss, loss_grad_W, loss_hess_W


    def _compute_cum_sums(self, a,for_loss = False):
        #preprocessing for positive and negative example
        label_a = self.labels[a]
        which_have_a = (self.labels == label_a)*1
        # print(self.labels)
        # print(which_have_a)

        #compute normalized distances
        difference_matrix = self.embedded_X - np.outer(np.ones(self.n_samples),self.embedded_X[a,:])
        distances = np.linalg.norm(difference_matrix, axis=1) ** 2
        distances = distances + self.margin * which_have_a

        # print(distances,"distances")


        #ordering
        order = np.argsort(distances)[::-1]
        rev_order = np.argsort(order)

        # print(order, "order")
        # print(rev_order,"reverse order")

        ordered_labels = which_have_a[order]
        # print(ordered_labels,"ordered labels")
        cum_sum_for_negative_samples = np.multiply(np.cumsum(ordered_labels)[rev_order],1- which_have_a)
        cum_sum_for_positive_samples = np.cumsum((1 - ordered_labels[::-1]))[::-1]
        cum_sum_for_positive_samples = np.multiply(cum_sum_for_positive_samples[rev_order],which_have_a)

        # print(cum_sum_for_negative_samples, "cum sum negative")
        # print(cum_sum_for_positive_samples, "cum sum positive")

        cum_sum_final = cum_sum_for_negative_samples + cum_sum_for_positive_samples

        # print(cum_sum_final)
        #
        # print(skrrt)

        if for_loss:
            return cum_sum_final, which_have_a, distances
        return cum_sum_final,which_have_a







