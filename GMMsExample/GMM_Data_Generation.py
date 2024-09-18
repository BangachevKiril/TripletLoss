import numpy as np


def generate_GMM_data(n_components,n_samples, dim_samples,SNR, dim_components = None,pmf_components = None):

    # parameters
    if dim_components is None:
        dim_components = n_components
    if pmf_components is None:
        pmf_components = np.ones((n_components))/n_components

    #generating components
    components = np.zeros((n_components, dim_samples))
    components[:n_components, :dim_components] = np.random.normal(loc = 0,scale = SNR,size = (n_components, dim_components))
    component_directions= np.divide(components,np.linalg.norm(components, axis = 1).reshape(n_components,1))

    #GENERATING SAMPLES
    ## Choosing Components
    idx = np.random.choice(a= np.arange(n_components), p = pmf_components,size = n_samples)
    sample_means = components[idx, :]
    sample_means_directional = component_directions[idx, :]

    ## Adding Noise
    noise = np.random.normal(0,1,(n_samples,dim_samples))
    noise_in_comp_direction = np.sum(np.multiply(noise,sample_means_directional),axis = 1).reshape(n_samples,1)
    noise= noise - sample_means_directional*noise_in_comp_direction
    samples= sample_means + noise

    return (samples, idx)



(X,labels) = generate_GMM_data(5,20,10,100,2)
print(labels)








