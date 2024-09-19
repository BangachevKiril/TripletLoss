import numpy as np

class singletriplet:
    def __init__(self,metric,margin,embedding, X):
        self.metric = metric
        self.margin = margin
        self.embedding = embedding
        self.X = X

    def loss(self, W,  a,p,n):
        self.anchor = self.X[a,:]
        self.positive = self.X[p,:]
        self.negative = self.X[n,:]
        self.embed_anchor = self.embedding.evaluate(W, self.anchor)
        self.embed_positive = self.embedding.evaluate(W, self.positive)
        self.embed_negative = self.embedding.evaluate(W, self.negative)
        difference =       (self.metric.evaluate(self.embed_anchor, self.embed_positive) -
                            self.metric.evaluate(self.embed_anchor, self.embed_negative))

        return max(0, difference + self.margin)
        pass

    def loss_grad_W(self, W, a,p,n):
        W_shape = W.shape
        if self.loss(W, a,p,n) == 0:
            return np.zeros(W_shape)
        positive_derivative = (np.tensordot(self.metric.partial1(self.embed_anchor, self.embed_positive).T,
                                            self.embedding.partialW(W, self.anchor),axes = 1 )+
                               np.tensordot(self.metric.partial2(self.embed_anchor, self.embed_positive).T,
                                            self.embedding.partialW(W, self.positive),axes = 1))

        negative_derivative = (np.tensordot(self.metric.partial1(self.embed_anchor, self.embed_negative),
                                            self.embedding.partialW(W, self.anchor),axes = 1) +
                               np.tensordot(self.metric.partial2(self.embed_anchor, self.embed_negative),
                                            self.embedding.partialW(W, self.negative),axes = 1))

        return positive_derivative - negative_derivative

def triplet(metric,embedding, margin, W, anchor,positive,negative):
    embed_anchor = embedding.evaluate(W,anchor)
    embed_positive = embedding.evaluate(W,positive)
    embed_negative = embedding.evaluate(W,negative)
    difference = metric.evaluate(embed_anchor,embed_positive) - metric.evaluate(embed_anchor,embed_negative)
    return np.max(0, difference + margin)

def triplet_derivative_at(metric,embedding, margin, W, anchor,positive,negative):
    W_shape = W.shape
    if triplet(metric,embedding, margin, W, anchor,positive,negative)== 0:
        return np.zeros(W_shape)
    embed_anchor = embedding.evaluate(W, anchor)
    embed_positive = embedding.evaluate(W, positive)
    embed_negative = embedding.evaluate(W, negative)
    positive_derivative = (metric.partial1(embed_anchor,embed_positive)*embedding.partialW(W,anchor) +
                           metric.partial2(embed_anchor,embed_positive)*embedding.partialW(W,positive))

    negative_derivative = (metric.partial1(embed_anchor, embed_negative) * embedding.partialW(W, anchor) +
                           metric.partial2(embed_anchor, embed_negative) * embedding.partialW(W, negative))

    return positive_derivative - negative_derivative