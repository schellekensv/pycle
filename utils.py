"""Contains a set of misc. useful tools for the compressive learning toolbox"""

import numpy as np
from scipy.stats import multivariate_normal


def generateGMMdataset(n,K,N,balanced=True,isotropic=True,normalize=None):
    """Generate a synthetic dataset according to a Gaussian Mixture Model distribution.
    
    Arguments:
        - n: int, the dataset dimension
        - K: int, the number of Gaussian modes
        - N: int, the number of elements in the dataset
        - balanced:  bool (default=True), if True the Gaussians have the same weigths
        - isotropic: bool (default=True), if True each Gaussian has covariance of type scalar*Identity
        - normalize: string (default=None), if not None describes how to normalize the dataset. Available options:
            - 'l_2-unit-ball': the dataset is projected in the l_2 unit ball (i.e., all l_2 norms are <= 1)
            - 'l_inf-unit-ball': the dataset is projected in the l_inf unit ball (i.e., all entries are <= 1)
        
    Returns: a tuple (X,weigths,means,covariances) of four numpy arrays
        - X: (N,n)-numpy array containing the samples (ordered by Gaussian mode index)
        - weigths:     (K,)-numpy array containing the weigthing factors of the Gaussians
        - means:       (K,n)-numpy array containing the means of the Gaussians
        - covariances: (K,n,n)-numpy array containing the covariance matrices of the Gaussians
    """
    
    weigths = np.ones(K)/K  # True, ideal weigths (balanced case)
    classSizes = np.ones(K) # Actual samples per class
    # (note: we enforce that weigths is the *actual* proportions in this dataset)
    
    ## Select number of samples of each mode
    if balanced:
        classSizes[:-1] = int(N/K)
        classSizes[-1] = N - (K-1)*int(N/K) # ensure we have exactly N samples in dataset even if N % K != 0
    else:
        minweight = min(0.01,(K-1)/(N-1)) # Some minimum weight to avoid empty classes
        weigths = np.random.uniform(minweight,1,K) 
        weigths = weigths/np.sum(weigths) # Normalize
        classSizes[:-1] = (weigths[:-1]*N).astype(int)
        classSizes[-1] = N - np.sum(classSizes[:-1])
    classSizes = classSizes.astype(int)

    ## Initialization
    X = None
    means = None
    covariances = None
    
    ## Some internal params (TODO allow to give them as optional args? kind of arbitrary!)
    scale_separation = (5/np.sqrt(n)) # Separation of the Gaussians
    scale_variance_b = np.array([0.05,0.95])/np.sqrt(n) # Bounds on the scale variance (actually, SD)
    
    ## Add each mode one by one 
    for k in range(K):
        classN = classSizes[k]
        mu = scale_separation*np.random.randn(n) 
        scale_variance = np.random.uniform(scale_variance_b[0],scale_variance_b[1])
        if isotropic:
            Sigma = np.identity(n) 
        else:
            sigs = np.random.uniform(0.5,1.5,n) # TODO CHANGE THIS
            Sigma = np.diag(sigs)
        Sigma = scale_variance*Sigma
        newCluster = np.random.multivariate_normal(mu, Sigma, classN)
        if X is None:
            X = newCluster
            means = np.expand_dims(mu,axis=0)
            covariances = np.expand_dims(Sigma,axis=0)
        else:
            X = np.append(X,newCluster,axis=0)
            means = np.append(means,np.expand_dims(mu,axis=0),axis=0)
            covariances = np.append(covariances,np.expand_dims(Sigma,axis=0),axis=0)
            
    if normalize is not None:
        if normalize in ['l_2-unit-ball']:
            maxNorm = np.linalg.norm(X,axis=1).max()
        elif normalize in ['l_inf-unit-ball']:
            maxNorm = np.abs(X).max()
        else:
            raise Exception('Unreckognized normalization method ({}). Aborting.'.format(normalize))
        # Normalize by maxNorm
        X /= maxNorm
        means /= maxNorm
        covariances /= maxNorm**2
    
    
    return (X,weigths,means,covariances)




def SSE(X,C):
    """Computes the Sum of Squared Errors of some centroids on a dataset, given by
        SSE(X,C) = sum_{x_i in X} min_{c_k in C} ||x_i-c_k||_2^2.
    
    Arguments:
        - X: (n,d)-numpy array, the dataset of n examples in dimension d
        - C: (K,d)-numpy array, the K centroids in dimension d
        
    Returns:
        - SSE: real, the SSE score defined above
    """
    # Dumb implementation of the SSE
    SSE = 0.
    for i in range(X.shape[0]):
        SSE += np.min(np.linalg.norm(C-X[i],axis=1))**2
    return SSE

def loglikelihood_GMM(P,X,robust = True):
    """Computes the loglikelihood of GMM model P on data X, defined as follows:
        loglikelihood = (1/n) * sum_{i=1..n} log(sum_{k=1..K} (w_k)*N(x_i ; mu_k, Sigma_k) )
    
    Arguments:
        - P: tuple of three numpy arrays describing the GMM model of form (w,mus,Sigmas)
            - w      : (K,)-numpy array, the weights of the K Gaussians (should sum to 1)
            - mus    : (K,d)-numpy array containing the means of the Gaussians
            - Sigmas : (K,d,d)-numpy array containing the covariance matrices of the Gaussians
        - X: (n,d)-numpy array, the dataset of n examples in dimension d
        - robust: bool (default = True), if True, avoids -inf output due to very small probabilities
                  (note: execution will be slower)
        
    Returns:
        - loglikelihood: real, the loglikelihood value defined above
    """
    
    # Unpack
    (w,mu,Sig) = P
    K = w.size
    
    logp = np.zeros(X.shape[0])
    p = np.zeros(X.shape[0])
    
    for k in range(K):
        p += w[k]*multivariate_normal.pdf(X, mean=mu[k], cov=Sig[k], allow_singular=True)
    logp = np.log(p)
    
    if robust:
        b = np.zeros(K)
        a = np.zeros(K)
        for k in range(K):
            a[k] = w[k]*((2*np.pi)**(-d/2))*(np.linalg.det(Sig[k])**(-1/2))
        for i in np.where(p==0)[0]: # Replace the inf values due to rounding p to 0
            for k in range(K):
                b[k] = -(X[i]-mus[k])@np.linalg.inv(Sig[k])@(X[i]-mus[k])/2
            lc = b.max()
            ebc = np.exp(b-lc)
            logp[i] = np.log(ebc@a) + lc
        
        
    return np.mean(logp)