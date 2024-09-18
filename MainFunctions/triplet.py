import numpy as np

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