import numpy as np

def prediction(X_test,L_train,X_train,embed_all,P,metric, method="centroid"):
    n_test = X_test.shape[0]
    unique_labels=np.unique(L_train)

    embedded_train = embed_all(P,X_train)
    embedded_test  = embed_all(P,X_test)

    dk = embedded_test.shape[1]

    L_test_hat =np.zeros(n_test)
    if method == "centroid" :
        embedded_centroids = np.zeros((len(unique_labels),dk))
        for idx,label in enumerate(unique_labels):
            which_have_the_label = (L_train == label)
            embedded_centroids[idx, :] = np.mean(embedded_train[which_have_the_label, :], axis = 0)

        for j in range(n_test):
            v = embedded_test[j,:]
            L_test_hat[j] = np.argmin(np.array([metric.evaluate(v, embedded_centroids[i,:]) for i in range(len(unique_labels))]))

    return L_test_hat

def accuracy(X_test,L_test,X_train,L_train,embed_all,P,metric, method="centroid"):
    L_test_hat  = prediction(X_test,L_train,X_train,embed_all,P,metric, method)
    return np.sum(L_test == L_test_hat) / L_test.shape[0]