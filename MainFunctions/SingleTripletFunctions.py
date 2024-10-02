import numpy as np
import tensorflow.sparse as sp



def loss(a,p,n,margin, current_distance_matrix):
    return max(0, current_distance_matrix[a,p] - current_distance_matrix[a,n] +margin)


def loss_grad_W(a,p,n,margin,current_distance_matrix, full_partials):

    if loss(a,p,n,margin, current_distance_matrix) < 1e-10:
        print("loss is zero")
        return np.zeros(full_partials[0,0,:].shape)


    positive_derivative = (full_partials[a,p,:] + full_partials[p,a,:])

    negative_derivative = (full_partials[a,n,:] + full_partials[n,a,:])


    return positive_derivative - negative_derivative


def loss_hess_W(self,W,a,p,n):
    pass

# def triplet(metric,embedding, margin, W, anchor,positive,negative):
#     embed_anchor = embedding.evaluate(W,anchor)
#     embed_positive = embedding.evaluate(W,positive)
#     embed_negative = embedding.evaluate(W,negative)
#     difference = metric.evaluate(embed_anchor,embed_positive) - metric.evaluate(embed_anchor,embed_negative)
#     return np.max(0, difference + margin)
#
# def triplet_derivative_at(metric,embedding, margin, W, anchor,positive,negative):
#     W_shape = W.shape
#     if triplet(metric,embedding, margin, W, anchor,positive,negative)== 0:
#         return np.zeros(W_shape)
#     embed_anchor = embedding.evaluate(W, anchor)
#     embed_positive = embedding.evaluate(W, positive)
#     embed_negative = embedding.evaluate(W, negative)
#     positive_derivative = (metric.partial1(embed_anchor,embed_positive)*embedding.partialW(W,anchor) +
#                            metric.partial2(embed_anchor,embed_positive)*embedding.partialW(W,positive))
#
#     negative_derivative = (metric.partial1(embed_anchor, embed_negative) * embedding.partialW(W, anchor) +
#                            metric.partial2(embed_anchor, embed_negative) * embedding.partialW(W, negative))
#
#     return positive_derivative - negative_derivative