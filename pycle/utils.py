"""Contains a set of misc. useful tools for the compressive learning toolbox"""

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

############################
# DATASET GENERATION TOOLS #
############################

def generatedataset_GMM(d,K,n,balanced=True,isotropic=True,normalize=None):
    """Generate a synthetic dataset according to a Gaussian Mixture Model distribution.
    
    Arguments:
        - d: int, the dataset dimension
        - K: int, the number of Gaussian modes
        - n: int, the number of elements in the dataset (cardinality)
        - balanced:  bool (default=True), if True the Gaussians have the same weigths
        - isotropic: bool (default=True), if True each Gaussian has covariance of type scalar*Identity
        - normalize: string (default=None), if not None describes how to normalize the dataset. Available options:
            - 'l_2-unit-ball': the dataset is scaled in the l_2 unit ball (i.e., all l_2 norms are <= 1)
            - 'l_inf-unit-ball': the dataset is projected in the l_inf unit ball (i.e., all entries are <= 1)
        
    Returns: a tuple (X,weigths,means,covariances) of four numpy arrays
        - X: (n,d)-numpy array containing the samples (ordered by Gaussian mode index)
        - weigths:     (K,)-numpy array containing the weigthing factors of the Gaussians
        - means:       (K,d)-numpy array containing the means of the Gaussians
        - covariances: (K,d,d)-numpy array containing the covariance matrices of the Gaussians
    """
    
    weigths = np.ones(K)/K  # True, ideal weigths (balanced case)
    classSizes = np.ones(K) # Actual samples per class
    # (note: we enforce that weigths is the *actual* proportions in this dataset)
    
    ## Select number of samples of each mode
    if balanced:
        classSizes[:-1] = int(n/K)
        classSizes[-1] = n - (K-1)*int(n/K) # ensure we have exactly n samples in dataset even if n % K != 0
    else:
        minweight = min(0.01,(K-1)/(n-1)) # Some minimum weight to avoid empty classes
        weigths = np.random.uniform(minweight,1,K) 
        weigths = weigths/np.sum(weigths) # Normalize
        classSizes[:-1] = (weigths[:-1]*n).astype(int)
        classSizes[-1] = n - np.sum(classSizes[:-1])
    classSizes = classSizes.astype(int)

    ## Initialization
    X = None
    means = None
    covariances = None
    
    ## Some internal params (TODO allow to give them as optional args? kind of arbitrary!)
    scale_separation = (5/np.sqrt(d)) # Separation of the Gaussians
    scale_variance_b = np.array([0.05,0.95])/np.sqrt(d) # Bounds on the scale variance (actually, SD)
    
    ## Add each mode one by one 
    for k in range(K):
        classN = classSizes[k]
        mu = scale_separation*np.random.randn(d) 
        scale_variance = np.random.uniform(scale_variance_b[0],scale_variance_b[1])
        if isotropic:
            Sigma = np.identity(d) 
        else:
            sigs = np.random.uniform(0.5,1.5,d) # TODO CHANGE THIS
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
            maxNorm = np.linalg.norm(X,axis=1).max() + 1e-6 # plus smth to have 
        elif normalize in ['l_inf-unit-ball']:
            maxNorm = np.abs(X).max() + 1e-6
        else:
            raise Exception('Unreckognized normalization method ({}). Aborting.'.format(normalize))
        # Normalize by maxNorm
        X /= maxNorm
        means /= maxNorm
        covariances /= maxNorm**2
    
    
    return (X,weigths,means,covariances)

############################
#         METHODS          #
############################

def EM_GMM(X,K,max_iter = 20):
    """Usual Expectation-Maximization (EM) algorithm for fitting mixture of Gaussian models (GMM).
    
    Arguments:
        - X: (n,d)-numpy array, the dataset of n examples in dimension d
        - K: int, the number of Gaussian modes
        - max_iter: int, the number of EM iterations to perform
        
    Returns: a tuple (w,mus,Sigmas) of three numpy arrays
        - w:      (K,)   -numpy array containing the weigths ('mixing coefficients') of the Gaussians
        - mus:    (K,d)  -numpy array containing the means of the Gaussians
        - Sigmas: (K,d,d)-numpy array containing the covariance matrices of the Gaussians
    """
    # TODO to improve:
    # - detect early convergence
    
    # Parse input
    (n,d) = X.shape
    lowb = np.amin(X,axis=0)
    uppb = np.amax(X,axis=0)
    
    # Initializations
    w = np.ones(K)
    mus = np.empty((K,d))
    Sigmas = np.empty((K,d,d)) # Covariances are initialized as random diagonal covariances, with folded Gaussian values
    for k in range(K):
        mus[k] = np.random.uniform(lowb,uppb)
        Sigmas[k] = np.diag(np.abs(np.random.randn(d)))
    r = np.empty((n,K)) # Matrix of posterior probabilities, here memory allocation only

    # Main loop
    for i in range(max_iter):
        # E step
        for k in range(K):
            r[:,k] = w[k]*multivariate_normal.pdf(X, mean=mus[k], cov=Sigmas[k],allow_singular=True)
        r = (r.T/np.sum(r,axis=1)).T # Normalize (the posterior probabilities sum to 1). Dirty :-(

        # M step: 1) update w
        w = np.sum(r,axis=0)/n 

        # M step: 2) update centers
        for k in range(K):
            mus[k] = r[:,k]@X/np.sum(r[:,k])

        # M step: 3) update Sigmas
        for k in range(K):
            # Dumb implementation
            num = np.zeros((d,d))
            for i in range(n):
                num += r[i,k]*np.outer(X[i]-mus[k],X[i]-mus[k])
            Sigmas[k] = num/np.sum(r[:,k])

        # (end of one EM iteration)
    return (w,mus,Sigmas)


############################
#         METRICS          #
############################


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
    
    # TODO : avoid recomputations of inv
    
    # Unpack
    (w,mu,Sig) = P
    (K,d) = mu.shape
    
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
                b[k] = -(X[i]-mu[k])@np.linalg.inv(Sig[k])@(X[i]-mu[k])/2
            lc = b.max()
            ebc = np.exp(b-lc)
            logp[i] = np.log(ebc@a) + lc
        
        
    return np.mean(logp)


def symmKLdivergence_GMM(P1,P2,Neval = 500000,verbose=0):
    """Computes the symmetric KL divergence between two GMM densities."""
    tol = 1e-7
    # TODO : a version that adapts Neval s.t. convergence?
    # Unpack
    (w1,mu1,Sig1) = P1
    (w2,mu2,Sig2) = P2
    K1 = w1.size
    K2 = w2.size
    
    Neval # Number of samples to evaluate the KL divergence
    
    # dumb implem for now, TODO FAST IMPLEM!
    KLestimate = 0.
    for i in range(Neval):
        # Sample from P1
        index_gaussianDrawnFrom = np.random.choice(np.arange(K1),p=w1)
        y = np.random.multivariate_normal(mu1[index_gaussianDrawnFrom], Sig1[index_gaussianDrawnFrom])
        
        # Evaluate density of P1
        p1 = 0.
        for k in range(K1):
            p1 += w1[k]*multivariate_normal.pdf(y, mean=mu1[k], cov=Sig1[k], allow_singular=True)
        
        # Evaluate density of P2
        p2 = 0.
        for k in range(K2):
            p2 += w2[k]*multivariate_normal.pdf(y, mean=mu2[k], cov=Sig2[k], allow_singular=True)
        
        # Compute the contribution
        contribution_i = np.log(p1/p2) + (p2/p1)*np.log(p2/p1)
        if (p1 < tol) and (p2 < tol):
            contribution_i = 0. # Avoid rounding errors (?)
        
        # Add it
        KLestimate = KLestimate*(i/(i+1)) + contribution_i/(i+1)
        if i%10000 == 0:
            if verbose > 0: print(i,KLestimate)
            
        
        
    return KLestimate



############################
#      VISUALIZATION       #
############################
from matplotlib.patches import Ellipse
from scipy.stats import chi2

def plotGMM(X,P,dims=(0,1),d=2,proportionInGMM = None):
    """TODO"""
    # To finish
    # Get K from the thing 
    (w,mus,Sigmas) = P # Unpack
    K = w.size
    dim0,dim1=dims
    if proportionInGMM is None:
        # for 95, d = 2%
        cst=2*np.sqrt(5.991)
    else:
        cst = 2*np.sqrt(chi2.isf(1-proportionInGMM, d)) # check https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
    plt.figure(figsize=(5,5))
    plt.scatter(X[:,dim0],X[:,dim1],s=1, alpha=0.15)
    ax = plt.gca()

    for k in range(K):
        mu = mus[k]
        sigma_sol = np.diag(Sigmas[k])
        plt.scatter(mu[dim0],mu[dim1],s=200*w[k],c='r')

        wEll = cst*np.sqrt(sigma_sol[dim0])
        hEll = cst*np.sqrt(sigma_sol[dim1])
        ellipse = Ellipse(xy=mu, width=wEll, height=hEll, angle = 0,
                                edgecolor='r', fc='None', lw=2)
        ax.add_patch(ellipse)


    plt.show()
    
    return


# TODO plot centroids?
