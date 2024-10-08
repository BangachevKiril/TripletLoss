import numpy as np


def generate_GMM_means(n_components, dim_samples, dim_components = None):
    if dim_components is None:
        dim_components = n_components

        # generating components
    components = np.zeros((n_components, dim_samples))
    components[:, :dim_components] = np.random.normal(loc=0, scale=1/np.sqrt(dim_components), size=(n_components, dim_components))

    return components


def generate_GMM_samples(components, n_samples, sigma, pmf_components = None):
    # parameters
    (n_components, d_samples) = components.shape

    if pmf_components is None:
        pmf_components = np.ones(n_components)/n_components

    #GENERATING SAMPLES
    ## Choosing Components
    labels = np.random.choice(a= np.arange(n_components), p = pmf_components,size = n_samples)
    sample_means = components[labels, :]

    ## Adding Noise
    noise = np.random.normal(0,sigma,(n_samples,d_samples))
    samples= sample_means + noise

    return samples, labels


if __name__ == '__main__':
    print("component check")
    components = generate_GMM_means(n_components = 3, dim_samples = 4, dim_components = 2)
    print(components)

    print("component in samples check")
    (samples,labels) = generate_GMM_samples(components, 10, 0)
    print(labels)
    print(samples)

    print("sampels check")
    (samples, labels) = generate_GMM_samples(components, 10, 1)
    print(labels)
    print(samples)













