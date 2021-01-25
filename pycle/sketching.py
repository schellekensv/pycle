"""MIT License

Copyright (c) 2019 schellekensv

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""


"""Contains tools to compute the sketch the dataset."""

# Main imports
import numpy as np
import matplotlib.pyplot as plt # For verbose
import scipy.optimize
import sys          # For error handling
from copy import copy
    
    
#######################################
### 1: Frequency sampling functions ###
#######################################

# 1.0: dithering
def drawDithering(m,bounds = None):
    '''Draws m samples a <= x < b, with bounds=(a,b) (default: (0,2*pi)).'''
    if bounds is None:
        (lowb,highb) = (0,2*np.pi)
    else:
        (lowb,highb) = bounds
    return np.random.uniform(low=lowb,high=highb,size=m)

# 1.1: frequency sampling functions
# 1.1.1: gaussian sampling
def drawFrequencies_Gaussian(d,m,Sigma = None):
    '''draws frequencies according to some sampling pattern'''
    if Sigma is None:
        Sigma = np.identity(d)
    Om = np.random.multivariate_normal(np.zeros(d), np.linalg.inv(Sigma), m).T # inverse of sigma
    return Om

# 1.1.2: folded gaussian sampling
def drawFrequencies_FoldedGaussian(d,m,Sigma = None):
    '''draws frequencies according to some sampling pattern
    omega = R*Sigma^{-1/2}*phi, for R from folded Gaussian with variance 1, phi uniform''' 
    if Sigma is None:
        Sigma = np.identity(d)
    R = np.abs(np.random.randn(m)) # folded standard normal distribution radii
    phi = np.random.randn(d,m)
    phi = phi / np.linalg.norm(phi,axis=0) # normalize -> randomly sampled from unit sphere
    SigFact = np.linalg.inv(np.linalg.cholesky(Sigma)) 
    
    Om = SigFact@phi*R
    
    return Om

# 1.1.3: adapted radius sampling
def sampleFromPDF(pdf,x,nsamples=1):
    '''x is a vector (the support of the pdf), pdf is the values of pdf eval at x'''
    # Note that this can be more general than just the adapted radius distribution
    
    pdf = pdf/np.sum(pdf) # ensure pdf is normalized
    
    cdf = np.cumsum(pdf)
    
    # necessary?
    cdf[-1] = 1.
    
    sampleCdf = np.random.uniform(0,1,nsamples)
    
    sampleX = np.interp(sampleCdf, cdf, x)

    return sampleX
   
def pdfAdaptedRadius(r,KMeans=False):
    '''up to a constant'''
    if KMeans:
        return r*np.exp(-(r**2)/2)  # Dont take the gradient according to sigma into account
    else:
        return np.sqrt(r**2 + (r**4)/4)*np.exp(-(r**2)/2) 

def drawFrequencies_AdaptedRadius(d,m,Sigma = None,KMeans=False):
    '''draws frequencies according to some sampling pattern
    omega = R*Sigma^{-1/2}*phi, for R from adapted with variance 1, phi uniform''' 
    if Sigma is None:
        Sigma = np.identity(d)
        
    # Sample the radii
    r = np.linspace(0,5,2001) 
    R = sampleFromPDF(pdfAdaptedRadius(r,KMeans),r,nsamples=m)
    
    phi = np.random.randn(d,m)
    phi = phi / np.linalg.norm(phi,axis=0) # normalize -> randomly sampled from unit sphere
    SigFact = np.linalg.inv(np.linalg.cholesky(Sigma)) 
    
    Om = SigFact@phi*R 
    
    return Om


def pdf_diffOfGaussians(r,GMM_upper=None,GMM_lower=None):
    """Here, GMM is given in terms of SD and not variance"""
    if isinstance(GMM_upper,tuple):
        (weights_upper,sigmas_upper) = GMM_upper
    elif GMM_upper is None:
        weights_upper = np.array([]) # Empty array
    else:
        (weights_upper,sigmas_upper) = (np.array([1.]),np.array([GMM_upper]))
        
    if isinstance(GMM_lower,tuple):
        (weights_lower,sigmas_lower) = GMM_lower
    elif GMM_lower is None:
        weights_lower = np.array([])
    else:
        (weights_lower,sigmas_lower) = (np.array([1.]),np.array([GMM_lower]))
        
    res = np.zeros(r.shape)
    # Add
    for k in range(weights_upper.size):
        res += weights_upper[k]*np.exp(-0.5*(r**2)/(sigmas_upper[k]**2))
    # Substract
    for k in range(weights_lower.size):
        res -= weights_lower[k]*np.exp(-0.5*(r**2)/(sigmas_lower[k]**2))
        
    # Ensure pdf is positive
    pdf_is_negative = res < 0
    if any(pdf_is_negative):
        print(res[:5])
        # Print a warning if the negative pdf values are significant (not due to rounding errors)
        tol = 1e-8
        if np.max(np.abs(res[np.where(pdf_is_negative)[0]])) > tol:
            print("WARNING: negative pdf values detected and replaced by zero, check the validity of your input")
        # Correct the negative values
        res[np.where(pdf_is_negative)[0]] = 0.

    return res

def drawFrequencies_diffOfGaussians(d,m,GMM_upper,GMM_lower=None,verbose=0):
    '''draws frequencies according to some sampling pattern
    omega = R*Sigma^{-1/2}*phi, TODO, phi uniform''' 
    

    # reasonable sampling
    n_Rs = 1001
    if isinstance(GMM_upper,tuple):
        R_max = 4*np.max(GMM_upper[1]) # GMM_upper is (weights, cov)-type tuple
    else:
        R_max = 4*GMM_upper
    r = np.linspace(0,R_max,n_Rs)
    
    if verbose > 0:
        plt.plot(r,pdf_diffOfGaussians(r,GMM_upper,GMM_lower))
        plt.xlabel('frequency norm r')
        plt.ylabel('pdf(r)')
        plt.show()
    
    # sample from the diff of gaussians pdf
    R = sampleFromPDF(pdf_diffOfGaussians(r,GMM_upper,GMM_lower),r,nsamples=m)
    
    phi = np.random.randn(d,m)
    phi = phi / np.linalg.norm(phi,axis=0) # normalize -> randomly sampled from unit sphere
    
    Om = phi*R 
    
    return Om


# General function for convenience
def drawFrequencies(drawType,d,m,Sigma = None,nb_cat_per_dim=None):
    """Draw the 'frequencies' or projection matrix Omega for sketching.
    
    Arguments:
        - drawType: a string indicating the sampling pattern (Lambda) to use, one of the following:
            -- "gaussian"       or "G"  : Gaussian sampling > Lambda = N(0,Sigma^{-1})
            -- "foldedGaussian" or "FG" : Folded Gaussian sampling (i.e., the radius is Gaussian)
            -- "adaptedRadius"  or "AR" : Adapted Radius heuristic
        - d: int, dimension of the data to sketch
        - m: int, number of 'frequencies' to draw (the target sketch dimension)
        - Sigma: is either:
            -- (d,d)-numpy array, the covariance of the data (note that we typically use Sigma^{-1} in the frequency domain).
            -- a tuple (w,cov) describing a scale mixture of Gaussians where,
                -- w:  (K,)-numpy array, the weights of the scale mixture
                -- cov: (K,d,d)-numpy array, the K different covariances in the mixture
            -- None: same as Sigma = identity matrix (belongs to (d,d)-numpy array case)
                 If Sigma is None (default), we assume that data was normalized s.t. Sigma = identity.
        - nb_cat_per_dim: (d,)-array of ints, the number of categories per dimension for integer data,
                    if its i-th entry = 0 (resp. > 0), dimension i is assumed to be continuous (resp. int.).
                    By default all entries are assumed to be continuous. Frequencies for int data is drawn as follows:
                    1. Chose one dimension among the categorical ones, set Omega along all others to zero
                    2. For the chosen dimension with C categories, we draw its component omega ~ U({0,...,C-1}) * 2*pi/C 
        
    Returns:
        - Omega: (d,m)-numpy array containing the 'frequency' projection matrix
    """
    # Parse drawType input
    if drawType.lower() in ["drawfrequencies_gaussian","gaussian","g"]:
        drawFunc = drawFrequencies_Gaussian
    elif drawType.lower() in ["drawfrequencies_foldedgaussian","foldedgaussian","folded_gaussian","fg"]:
        drawFunc = drawFrequencies_FoldedGaussian
    elif drawType.lower() in ["drawfrequencies_adapted","adaptedradius","adapted_radius","ar"]:
        drawFunc = drawFrequencies_AdaptedRadius
    elif drawType.lower() in ["drawfrequencies_adapted_kmeans","adaptedradius_kmeans","adapted_radius_kmeans","arkm","ar-km"]:
        drawFunc = lambda _a,_b,_c : drawFrequencies_AdaptedRadius(_a,_b,_c,KMeans=True)
    else:
        raise ValueError("drawType not recognized")

    # Handle no input
    if Sigma is None:
        Sigma = np.identity(d)

    # Handle 
    if isinstance(Sigma,np.ndarray):
        Omega = drawFunc(d,m,Sigma)

    # Handle mixture-type input
    elif isinstance(Sigma,tuple):
        (w,cov) = Sigma # unpack
        K = w.size
        # Assign the frequencies to the mixture components
        assignations = np.random.choice(K,m,p=w)
        Omega = np.zeros((d,m))
        for k in range(K):
            active_index = (assignations == k)
            if any(active_index):
                Omega[:,np.where(active_index)[0]] = drawFunc(d,active_index.sum(),cov[k])

    elif (isinstance(Sigma,float) or isinstance(Sigma,int)) and Sigma > 0:
        Omega = drawFunc(d,m,Sigma*np.eye(d))
    
    else:
        raise ValueError("Sigma not recognized")

    # If needed, overwrite the integer entries
    if nb_cat_per_dim is not None:
        intg_index = np.nonzero(nb_cat_per_dim)[0]
        d_intg = np.size(intg_index)

        Omega_intg = np.zeros((d_intg,m))
        for intgdim_localindex,intg_globalindex in enumerate(intg_index):
            C = nb_cat_per_dim[intg_globalindex]
            Omega_intg[intgdim_localindex,:] = (2*np.pi/C)*np.random.randint(0,C,(1,m))
        # Mask
        masks_pool = np.eye(d_intg)
        masks = masks_pool[np.random.choice(d_intg,m)].T
        Omega_intg = Omega_intg*masks

        Omega[intg_index] = Omega_intg

    
    return Omega

# The following funtions allows to estimate Sigma (some utils first, main function is "estimate_Sigma")
# Optimization problem to fit a GMM curve to the data
def _fun_grad_fit_sigmas(p,R,z):
    """
    Function and gradient to solve the optimization problem
        min_{w,sigs2} sum_{i = 1}^n ( z[i] - sum_{k=1}^K w[k]*exp(-R[i]^2*sig2[k]/2) )^2
    Arguments:
        - p, a (2K,) numpy array obtained by stacking
            - w : (K,) numpy array
            - sigs2 : (K,) numpy array
        - R: (n,) numpy array, data to fit (x label)
        - z: (n,) numpy array, data to fit (y label)
    Returns:
        - The function evaluation
        - The gradient
    """

    K = p.size//2
    w = p[:K]
    sigs2 = p[K:]
    n = R.size
    # Naive implementation, TODO better?
    fun = 0
    grad = np.zeros(2*K)
    for i in range(n):
        fun += (z[i] - w@np.exp(-(sigs2*R[i]**2)/2.))**2
        grad[:K] += (z[i] - w@np.exp(-(sigs2*R[i]**2)/2.)) * (- np.exp(-(sigs2*R[i]**2)/2.)) # grad of w
        grad[K:] += (z[i] - w@np.exp(-(sigs2*R[i]**2)/2.)) * (- w * np.exp(-(sigs2*R[i]**2)/2.)) * (-0.5*R[i]**2) # grad of sigma2
    return (fun,grad)

# For normalization in the optimization problem
def _callback_fit_sigmas(p):
    K = p.size//2
    p[:K] /= np.sum(p[:K])

def estimate_Sigma_from_sketch(z,Phi,K=1,c=20,mode='max',sigma2_bar=None,weights_bar=None,should_plot=False):
    # Parse
    if mode == 'max':
        mode_criterion = np.argmax 
    elif mode == 'min':
        mode_criterion = np.argmin 
    else:
        raise ValueError("Unrecocgnized mode ({})".format(mode))
        
    # sort the freqs by norm
    Rs = np.linalg.norm(Phi.Omega,axis=0)
    i_sort = np.argsort(Rs)
    Rs = Rs[i_sort]
    z_sort = z[i_sort]/Phi.c_norm # sort and normalize individual entries to 1
    
    # Initialization
    # number of freqs per box
    s = Phi.m//c
    # init point
    if sigma2_bar is None:
        sigma2_bar = np.random.uniform(0.3,1.6,K)
    if weights_bar is None:
        weights_bar = np.ones(K)/K

    # find the indices of the max of each block
    jqs = np.empty(c) 
    for ic in range(c):
        j_max = mode_criterion(np.abs(z_sort)[ic*s:(ic+1)*s]) + ic*s
        jqs[ic] = j_max
    jqs = jqs.astype(int)
    R_tofit = Rs[jqs]
    z_tofit = np.abs(z_sort)[jqs]

    # Set up the fitting opt. problem
    f = lambda p: _fun_grad_fit_sigmas(p,R_tofit,z_tofit) # cost

    p0 = np.zeros(2*K) # initial point
    p0[:K] = weights_bar # w
    p0[K:] = sigma2_bar 

    # Bounds of the optimization problem
    bounds = []
    for k in range(K): bounds.append([1e-5,1]) # bounds for the weigths
    for k in range(K): bounds.append([5e-4*sigma2_bar[k],2e3*sigma2_bar[k]]) # bounds for the sigmas -> cant cange too much

    # Solve the sigma^2 optimization problem
    sol = scipy.optimize.minimize(f, p0,jac = True, bounds = bounds,callback=_callback_fit_sigmas)
    p = sol.x
    weights_bar = np.array(p[:K])/np.sum(p[:K])
    sigma2_bar = np.array(p[K:])

    # Plot if required
    if should_plot:
        plt.figure(figsize=(10,5))
        rfit = np.linspace(0,Rs.max(),100)
        zfit = np.zeros(rfit.shape)
        for k in range(K):
            zfit += weights_bar[k]*np.exp(-(sigma2_bar[k]*rfit**2)/2.)
        plt.plot(Rs,np.abs(z_sort),'.')
        plt.plot(R_tofit,z_tofit,'.')
        plt.plot(rfit,zfit)
        plt.xlabel('R')
        plt.ylabel('|z|')
        plt.show()

    return sigma2_bar

def estimate_Sigma(dataset,m0,K=None,c=20,n0=None,drawFreq_type = "AR",nIterations=5,mode='max',verbose=0):
    """Automatically estimates the "Sigma" parameter(s) (the scale of data clusters) for generating the sketch operator.
    
    We assume here that Sigma = sigma2_bar * identity matrix. 
    To estimate sigma2_bar, lightweight sketches of size m0 are generated from (a small subset of) the dataset
    with candidate values for sigma2_bar. Then, sigma2_bar is updated by fitting a Gaussian
    to the absolute values of the obtained sketch. Cfr. https://arxiv.org/pdf/1606.02838.pdf, sec 3.3.3.
    
    Arguments:
        - dataset: (n,d) numpy array, the dataset X: n examples in dimension d
        - m0: int, number of candidate 'frequencies' to draw (can be typically smaller than m).
        - K:  int (default 1), number of scales to fit (if > 1 we fit a scale mixture)
        - c:  int (default 20), number of 'boxes' (i.e. number of maxima of sketch absolute values to fit)
        - n0: int or None, if given, n0 samples from the dataset are subsampled to be used for Sigma estimation
        - drawType: a string indicating the sampling pattern (Lambda) to use in the pre-sketches, either:
            -- "gaussian"       or "G"  : Gaussian sampling > Lambda = N(0,Sigma^{-1})
            -- "foldedGaussian" or "FG" : Folded Gaussian sampling (i.e., the radius is Gaussian)
            -- "adaptedRadius"  or "AR" : Adapted Radius heuristic
        - nIterations: int (default 5), the maximum number of iteration (typically stable after 2 iterations)
        - mode: 'max' (default) or 'min', describe which sketch entries per block to fit
        - verbose: 0,1 or 2, amount of information to print (default: 0, no info printed). Useful for debugging.
        
    Returns: If K = 1:
                - Sigma: (d,d)-numpy array, the (diagonal) estimated covariance of the clusters in the dataset;
             If K > 1: a tuple (w,Sigma) representing the scale mixture model, where:
                - w:     (K,)-numpy array, the weigths of the scale mixture (sum to 1)
                - Sigma: (K,d,d)-numpy array, the dxd covariances in the scale mixture
    """

    return_format_is_matrix = K is None
    K = 1 if K is None else K
    
    (n,d) = dataset.shape
    # X is the subsampled dataset containing only n0 examples
    if n0 is not None and n0<n:
        X = dataset[np.random.choice(n,n0,replace=False)]
    else:
        X = dataset
    

    # Check if we dont overfit the empirical Fourier measurements
    if (m0 < (K * 2)*c): 
        print("WARNING: overfitting regime detected for frequency sampling fitting")
    
    # Initialization
    #maxNorm = np.max(np.linalg.norm(X,axis=1)) 
    sigma2_bar = np.random.uniform(0.3,1.6,K)
    weights_bar = np.ones(K)/K    

    # Actual algorithm
    for i in range(nIterations):
        # Draw frequencies according to current estimate 
        sigma2_bar_matrix = np.outer(sigma2_bar,np.eye(d)).reshape(K,d,d)  # covariances in (K,d,d) format
        Omega0 = drawFrequencies(drawFreq_type,d,m0,Sigma = (weights_bar,sigma2_bar_matrix))
        
        # Compute unnormalized complex exponential sketch
        Phi0 = SimpleFeatureMap("ComplexExponential",Omega0)
        z0 = computeSketch(X,Phi0) 
        
        should_plot = verbose > 1 or (verbose > 0 and i >= nIterations - 1)
        sigma2_bar = estimate_Sigma_from_sketch(z0,Phi0,K,c,mode,sigma2_bar,weights_bar,should_plot)


    if return_format_is_matrix:
        Sigma = sigma2_bar[0]*np.eye(d)
    else:
        sigma2_bar_matrix = np.outer(sigma2_bar,np.eye(d)).reshape(K,d,d)  # covariances in (K,d,d) format        
        Sigma = (weights_bar,sigma2_bar_matrix)
    
    return Sigma


#######################################
###    2: Feature map functions     ###
#######################################

# 2.1: Common sketch nonlinearities and derivatives
def _complexExponential(t,T=2*np.pi):
    return np.exp(1j*(2*np.pi)*t/T)
def _complexExponential_grad(t,T=2*np.pi):
    return ((1j*2*np.pi)/T)*np.exp(1j*(2*np.pi)*t/T)

def _universalQuantization(t,Delta=np.pi,centering=True):
    if centering:
        return ( (t // Delta) % 2 )*2-1 # // stands for "int division
    else:
        return ( (t // Delta) % 2 ) # centering=false => quantization is between 0 and +1
    
def _universalQuantization_complex(t,Delta=np.pi,centering=True):
    return _universalQuantization(t-Delta/2,Delta=Delta,centering=centering) + 1j*_universalQuantization(t-Delta,Delta=Delta,centering=centering)
    
def _sawtoothWave(t,T=2*np.pi,centering=True):
    if centering:
        return ( t % T )/T*2-1 
    else:
        return ( t % T )/T # centering=false => output is between 0 and +1
    
def _sawtoothWave_complex(t,T=2*np.pi,centering=True):
    return _sawtoothWave(t-T/4,T=T,centering=centering) + 1j*_sawtoothWave(t-T/2,T=T,centering=centering)
    
def _triangleWave(t,T=2*np.pi):
    return (2*(t % T)/T ) - (4*(t % T)/T - 2)*( (t // T) % 2 ) - 1

def _fourierSeriesEvaluate(t,coefficients,T=2*np.pi):
    """T = period
    coefficients = F_{-K}, ... , F_{-1}, F_{0}, F_{1}, ... F_{+K}"""
    K = (coefficients.shape[0]-1)/2
    ks = np.arange(-K,K+1)
    # Pre-alloc
    ft = np.zeros(t.shape) + 0j
    for i in range(2*int(K)+1):
        ft += coefficients[i]*np.exp(1j*(2*np.pi)*ks[i]*t/T)
    return ft

# dict of nonlinearities and their gradient returned as a tuple
_dico_nonlinearities = {
    "complexexponential":(_complexExponential,_complexExponential_grad),
    "universalquantization":(_universalQuantization,None),
    "universalquantization_complex":(_universalQuantization_complex,None),
    "sawtooth":(_sawtoothWave,None),
    "sawtooth_complex":(_sawtoothWave_complex,None),
    "cosine": (lambda x: np.cos(x),lambda x: -np.sin(x))
 }


# 2.2 FeatureMap objects
# Abstract feature map class
class FeatureMap:
    """Template for a generic Feature Map. Useful to check if an object is an instance of FeatureMap."""
    def __init__(self):
        pass
    def __call__(self):
        raise NotImplementedError("The way to compute the feature map is not specified.")
    def grad(self):
        raise NotImplementedError("The way to compute the gradient of the feature map is not specified.")
        
# TODO find a better name
class SimpleFeatureMap(FeatureMap):
    """Feature map the type Phi(x) = c_norm*f(Omega^T*x + xi)."""
    def __init__(self, f, Omega, xi = None, c_norm = 1.):
        """
        - f can be one of the following:
            -- a string for one of the predefined feature maps:
                -- "complexExponential"
                -- "universalQuantization"
                -- "cosine"
            -- a callable function
            -- a tuple of function (specify the derivative too)
            
        """
        # 1) extract the feature map
        self.name = None
        if isinstance(f, str):
            try:
                (self.f,self.f_grad) = _dico_nonlinearities[f.lower()]
                self.name = f # Keep the feature function name in memory so that we know we have a specific fct
            except KeyError:
                raise NotImplementedError("The provided feature map name f is not implemented.")
        elif callable(f):
            (self.f,self.f_grad) = (f,None)
        elif (isinstance(f,tuple)) and (len(f) == 2) and (callable(f[0]) and callable(f[1])):
            (self.f,self.f_grad) = f
        else:
            raise ValueError("The provided feature map f does not match any of the supported types.")
            
        # 2) extract Omega the projection matrix TODO allow callable Omega for fast transform
        if (isinstance(Omega,np.ndarray) and Omega.ndim == 2):
            self.Omega = Omega
            (self.d,self.m) = Omega.shape
        else:
            raise ValueError("The provided projection matrix Omega should be a (d,m) numpy array.")
        # 3) extract the dithering
        if xi is None:
            self.xi = np.zeros(self.m)
        else:
            self.xi = xi
        # 4) extract the normalization constant
        if isinstance(c_norm, str):
            if c_norm.lower() in ['unit','normalized']:
                self.c_norm = 1./np.sqrt(self.m)
            else:
                raise NotImplementedError("The provided c_norm name is not implemented.")
        else:
            self.c_norm = c_norm
        
    # magic operator to be able to call the FeatureMap object as a function
    def __call__(self,x): 
        return self.c_norm*self.f(np.matmul(self.Omega.T,x.T).T + self.xi) # Evaluate the feature map at x
    
    def grad(self,x):
        """Gradient (Jacobian matrix) of Phi, as a (d,m)-numpy array"""
        return self.c_norm*self.f_grad(np.matmul(self.Omega.T,x.T).T + self.xi)*self.Omega
    



#######################################
### 3: Actual sketching functions   ###
#######################################    

#################################
# 3.1 GENERAL SKETCHING ROUTINE #
#################################
def computeSketch(dataset, featureMap, datasetWeights = None, batch_size = None):
    """
    Computes the sketch of a dataset given a generic feature map.
    
    More precisely, evaluates
        z = sum_{x_i in X} w_i * Phi(x_i)
    where X is the dataset, Phi is the sketch feature map, w_i are weights assigned to the samples (typically 1/n).
    
    Arguments:
        - dataset        : (n,d) numpy array, the dataset X: n examples in dimension d
        - featureMap     : the feature map Phi, given as one of the following:
            -- a function, z_x_i = featureMap(x_i), where x_i and z_x_i are (n,)- and (m,)-numpy arrays, respectively
            -- a FeatureMap instance (e.g., constructed as featureMap = SimpleFeatureMap("complexExponential",Omega) )
        - datasetWeights : (n,) numpy array, optional weigths w_i in the sketch (default: None, corresponds to w_i = 1/n)
        
    Returns: 
        - sketch : (m,) numpy array, the sketch as defined above
    """
    # TODOs:
    # - add possibility to specify classes and return one sketch per class
    # - defensive programming
    # - efficient implementation, take advantage of parallelism
    
    (n,d) = dataset.shape # number of samples, dimension 
    
    # Determine the sketch dimension and sanity check: the dataset is nonempty and the map works
    if isinstance(featureMap,FeatureMap): # featureMap is the argument, FeatureMap is the class
        m = featureMap.m
    else:
        try:
            m = featureMap(dataset[0]).shape[0]
        except:
            raise ValueError("Unexpected error while calling the sketch feature map:", sys.exc_info()[0])
    
    sketch = np.zeros(m)
    
    # Split the batches
    if batch_size is None:
        batch_size = int(1e6/m) # Rough heuristic, best batch size will vary on different machines
    nb_batches = int(np.ceil(n/batch_size))
        
    
    if datasetWeights is None:
        for b in range(nb_batches):
            sketch = sketch + featureMap(dataset[b*batch_size:(b+1)*batch_size]).sum(axis=0)
        sketch /= n
    else:
        sketch = datasetWeights@featureMap(X) 
    return sketch

#################################
# 3.2 PRIVATE SKETCHING METHODS #
#################################
def sensisitivty_sketch(featureMap,n = 1,DPdef = 'UDP',sensitivity_type = 1):
    """
    Computes the sensitity of a provided sketching function.
    
    The noisy sketch operator A(X) is given by
        A(X) := (1/n)*[sum_{x_i in X} featureMap(x_i)] + w
    where w is Laplacian or Gaussian noise. 
    
    Arguments:
        - featureMap, the sketch the sketch featureMap (Phi), provided as either:
            -- a FeatureMap object with a known sensitivity (i.e., complex exponential or universal quantization periodic map)
            -- (m,featureMapName,c_normalization): tuple (deprectated, only useful for code not supporting FeatureMap objects),
                that should contain:
                -- m: int, the sketch dimension
                -- featureMapName: string, name of sketch feature function f, values supported:
                    -- 'complexExponential' (f(t) = exp(i*t))
                    -- 'universalQuantization_complex' (f(t) = sign(exp(i*t)))
                -- c_normalization: real, the constant before the sketch feature function (e.g., 1. (default), 1./sqrt(m),...)
        - n: int, number of sketch contributions being averaged (default = 1, useful to add noise on n independently)
        - DPdef: string, name of the Differential Privacy variant considered, i.e. the neighbouring relation ~:
            -- 'remove', 'add', 'remove/add', 'UDP' or 'standard': D~D' iff D' = D U {x'} (or vice versa) 
            -- 'replace', 'BDP': D~D' iff D' = D \ {x} U {x'} (or vice versa) 
        - sensitivity_type: int, 1 (default) for L1 sensitivity, 2 for L2 sensitivity.
        
        
    Returns: a positive real, the L1 or L2 sensitivity of the sketching operator defined above.
    
    Cfr: Differentially Private Compressive K-means, https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8682829.
    """
    
    # TODO include real cases (cosine, real universal quantization)
    
    # The sensitivity is of the type: c_feat*c_
    if isinstance(featureMap,FeatureMap):
        m = featureMap.m
        featureMapName = featureMap.name
        c_normalization = featureMap.c_norm
    elif (isinstance(featureMap,tuple)) and (len(featureMap) == 3):
        (m,featureMapName,c_normalization) = featureMap
    else:
        raise ValueError('The featureMap argument does not match one of the supported formats.')
    
    # Sensitivity is given by S = c_featureMap * c_sensitivity_type * c_DPdef, check all three conditions (ughh)
    if featureMapName.lower() == 'complexexponential':
        if sensitivity_type == 1:
            if DPdef.lower() in ['remove','add','remove/add','standard','udp']:
                return m*np.sqrt(2)*(c_normalization/n)
            elif DPdef.lower() in ['replace','bdp']:
                return 2*m*np.sqrt(2)*(c_normalization/n)
        elif sensitivity_type == 2:
            if DPdef.lower() in ['remove','add','remove/add','standard','udp']:
                return np.sqrt(m)*(c_normalization/n)
            elif DPdef.lower() in ['replace','bdp']:
                return np.sqrt(m)*np.sqrt(2)*(c_normalization/n)
    elif featureMapName.lower() == 'universalquantization_complex': # Assuming normalized in [-1,+1], TODO check real/complex case?
        if sensitivity_type == 1:
            if DPdef.lower() in ['remove','add','remove/add','standard','udp']:
                return m*2*(c_normalization/n)
            elif DPdef.lower() in ['replace','bdp']:
                return 2*m*2*(c_normalization/n)
        elif sensitivity_type == 2:
            if DPdef.lower() in ['remove','add','remove/add','standard','udp']:
                return np.sqrt(m)*np.sqrt(2)*(c_normalization/n)
            elif DPdef.lower() in ['replace','bdp']:
                return np.sqrt(2)*np.sqrt(m)*np.sqrt(2)*(c_normalization/n)
    print(sensitivity_type)
    raise Exception('You provided ({},{});\nThe sensitivity for this (feature map,DP definition) combination is not implemented.'.format(featureMapName.lower(),DPdef.lower()))
    return None
    


def computeSketch_DP(dataset, featureMap, epsilon, delta = 0,DPdef = 'UDP',useImproveGaussMechanism=True,budget_split_num = None):
    """
    Computes the Differentially Private sketch of a dataset given a generic feature map.
    
    More precisely, evaluates the DP sketching mechanism:
        z = ( sum_{x_i in X} Phi(x_i) + w_num )/( |X| + w_den )
    where X is the dataset, Phi is the sketch feature map, w_num and w_den are Laplacian or Gaussian random noise.
    
    Arguments:
        - dataset        : (n,d) numpy array, the dataset X: n examples in dimension d
        - featureMap, the sketch the sketch featureMap (Phi), provided as either:
            -- a FeatureMap object with a known sensitivity (i.e., complex exponential or universal quantization periodic map)
            -- (featureMap(x_i),m,featureMapName,c_normalization): tuple (deprectated, only useful for old code),
                that should contain:
                -- featMap: a function, z_x_i = featMap(x_i), where x_i and z_x_i are (n,)- and (m,)-numpy arrays, respectively
                -- m: int, the sketch dimension
                -- featureMapName: string, name of sketch feature function f, values supported:
                    -- 'complexExponential' (f(t) = exp(i*t))
                    -- 'universalQuantization' (f(t) = sign(exp(i*t)))
                -- c_normalization: real, the constant before the sketch feature function (e.g., 1. (default), 1./sqrt(m),...)
        - epsilon: real > 0, the privacy parameter epsilon
        - delta:  real >= 0, the privacy parameter delta in approximate DP; if delta=0 (default), we have "pure" DP.
        - DPdef: string, name of the Differential Privacy variant considered, i.e. the neighbouring relation ~:
            -- 'remove', 'add', 'remove/add', 'UDP' or 'standard' (default): D~D' iff D' = D U {x'} (or vice versa) 
            -- 'replace', 'BDP': D~D' iff D' = D \ {x} U {x'} (or vice versa) 
        - useImproveGaussMechanism: bool, if True (default) use the improved Gaussian mechanism[1] rather than usual bounds[2].
        - budget_split_num: 0 < real < 1, fraction of epsilon budget to allocate to the numerator (ignored in BDP).
                            By default, we assign a fraction of (2*m)/(2*m+1) on the numerator.
        
    Returns: 
        - sketch : (m,) numpy array, the differentially private sketch as defined above
    """
    
    # Extract dataset size
    (n,d) = dataset.shape
    
    # Compute the nonprivate, usual sketch
    if isinstance(featureMap,FeatureMap):
        z_clean = computeSketch(dataset, featureMap)
    elif (isinstance(featureMap,tuple)) and (callable(featureMap[0])):
        featMap = featureMap[0]
        featureMap = featureMap[1:]
        z_clean = computeSketch(dataset, featMap)
    
    if epsilon == np.inf: # Non-private
        return z_clean
    
    useBDP = DPdef.lower() in ['replace','bdp'] # otherwise assume UDP, TODO DEFENSIVE
    
    # We will need the sketch size
    m = z_clean.size
    
    # Split privacy budget
    if useBDP: # Then no noise on the denom
        budget_split_num = 1.
    elif budget_split_num is None:
        budget_split_num = (2*m)/(2*m + 1)
    # TODO defensive programming to block budget split > 1?
    epsilon_num = budget_split_num*epsilon
    
    # Compute numerator noise
    if delta > 0:
        # Gaussian mechanism
        S = sensisitivty_sketch(featureMap,DPdef = DPdef,sensitivity_type = 2) # L2
        
        if useImproveGaussMechanism: # Use the sharpened bounds
            from .third_party import calibrateAnalyticGaussianMechanism
            sigma = calibrateAnalyticGaussianMechanism(epsilon_num, delta, S)
        else: # use usual bounds
            if epsilon >= 1: raise Exception('WARNING: with epsilon >= 1 the sigma bound doesn\'t hold! Privacy is NOT ensured!')
            sigma = np.sqrt(2*np.log(1.25/delta))*S/epsilon_num
        noise_num = np.random.normal(scale = sigma, size=m) + 1j*np.random.normal(scale = sigma, size=m) # TODO real
    else: 
        # Laplacian mechanism
        S = sensisitivty_sketch(featureMap,DPdef = DPdef,sensitivity_type = 1) # L1
        beta = S/epsilon_num # L1 sensitivity/espilon
        noise_num = np.random.laplace(scale = beta, size=m) + 1j*np.random.laplace(scale = beta, size=m) 
        
    # Add denominator noise if needed
    if useBDP: # Then no noise on the denom
        return z_clean + (noise_num/n)
    else:
        num = (z_clean*n) + noise_num
        beta_den = 1/(epsilon - epsilon_num) # rest of the privacy budget
        den = n + np.random.laplace(scale = beta_den)
        return num/den
    
    

## Useful: compute the sketch of a GMM
def fourierSketchOfGaussian(mu,Sigma,Omega,xi=None,scst=None):
    res = np.exp(1j*(mu@Omega) -np.einsum('ij,ij->i', np.dot(Omega.T, Sigma), Omega.T)/2.)
    if xi is not None:
        res = res*np.exp(1j*xi)
    if scst is not None: # Sketch constant, eg 1/sqrt(m)
        res = scst*res
    return res


def fourierSketchOfGMM(GMM,featureMap):
    '''Returns the complex exponential sketch of a Gaussian Mixture Model
    
    Parameters
    ----------
    GMM: (weigths,means,covariances) tuple, the Gaussian Mixture Model, with
        - weigths:     (K,)-numpy array containing the weigthing factors of the Gaussians
        - means:       (K,d)-numpy array containing the means of the Gaussians
        - covariances: (K,d,d)-numpy array containing the covariance matrices of the Gaussians
    featureMap: the sketch the sketch featureMap (Phi), provided as either:
        - a SimpleFeatureMap object (i.e., complex exponential or universal quantization periodic map)
        - (Omega,xi): tuple with the (d,m) Fourier projection matrix and the (m,) dither (see above)
        
    Returns
    -------
    z: (m,)-numpy array containing the sketch of the provided GMM
    '''
    # Parse GMM input
    (w,mus,Sigmas) = GMM
    K = w.size

    # Parse featureMap input
    if isinstance(featureMap,SimpleFeatureMap):
        Omega = featureMap.Omega
        xi = featureMap.xi
        d = featureMap.d
        m = featureMap.m
        scst = featureMap.c_norm # Sketch normalization constant, e.g. 1/sqrt(m)
    elif isinstance(featureMap,tuple):
        (Omega,xi) = featureMap
        (d,m) = Omega.shape
        scst = 1. # This type of argument passing does't support different normalizations
    else:
        raise ValueError('The featureMap argument does not match one of the supported formats.')
    
    z = 1j*np.zeros(m)
    for k in range(K):
        z += fourierSketchOfGaussian(mus[k],Sigmas[k],Omega,xi,scst)
    return z

def fourierSketchOfBox(box,featureMap,nb_cat_per_dim=None, dimensions_to_consider=None):
    '''Returns the complex exponential sketch of the indicator function on a parallellipiped (box).
    For dimensions that flagged as integer, considers the indicator on a set of integers instead.
    
    Parameters
    ----------
    box: (d,2)-numpy array, the boundaries of the box (x in R^d is in the box iff box[i,0] <= x_i <= box[i,1])
    featureMap: the sketch the sketch featureMap (Phi), provided as either:
        - a SimpleFeatureMap object (is assumed to use the complex exponential map)
        - (Omega,xi): tuple with the (d,m) Fourier projection matrix and the (m,) dither (see above)

    Additional Parameters
    ---------------------
    nb_cat_per_dim: (d,)-array of ints, the number of categories per dimension for integer data,
                    if its i-th entry = 0 (resp. > 0), dimension i is assumed to be continuous (resp. int.).
                    By default all entries are assumed to be continuous.
    dimensions_to_consider: array of ints (between 0 and d-1), [0,1,...d-1] by default.
                    The box is restricted to the prescribed dimensions.
                    This is helpful to solve problems on a subsets of all dimensions.

        
    Returns
    -------
    z: (m,)-numpy array containing the sketch of the indicator function on the provided box
    '''
    ## Parse input
    # Parse box input
    (d,_) = box.shape
    c_box = (box[:,1]+box[:,0])/2 # Center of the box in each dimension
    l_box = (box[:,1]-box[:,0])/2 # Length (well, half of the length) of the box in each dimension
    
    # Parse featureMap input
    if isinstance(featureMap,SimpleFeatureMap):
        Omega = featureMap.Omega
        xi = featureMap.xi
        d = featureMap.d
        m = featureMap.m
        scst = featureMap.c_norm # Sketch normalization constant, e.g. 1/sqrt(m)
    elif isinstance(featureMap,tuple):
        (Omega,xi) = featureMap
        (d,m) = Omega.shape
        scst = 1. # This type of argument passing does't support different normalizations
    else:
        raise ValueError('The featureMap argument does not match one of the supported formats.')

    # Parse nb_cat_per_dim
    if nb_cat_per_dim is None:
        nb_cat_per_dim = np.zeros(d)

    # Parse dimensions to consider
    if dimensions_to_consider is None:
        dimensions_to_consider = np.arange(d)

     ## Compute sketch
    z = scst*np.exp(1j*xi)
    for i in dimensions_to_consider:
        mask_valid = np.abs(Omega[i]) > 1e-15
        # CHECK IF INTEGER OR CONTINUOUS
        if nb_cat_per_dim[i] > 0:
            low_int  = box[i,0] 
            high_int = box[i,1] + 1 # python counting convention
            C = high_int - low_int
            newTerm = np.ones(m) + 1j*np.zeros(m)
            newTerm[mask_valid] = (1/C)*(np.exp(1j*Omega[i,mask_valid]*high_int)-np.exp(1j*Omega[i,mask_valid]*low_int))/(np.exp(1j*Omega[i,mask_valid])-1)
            z *= newTerm
        else:
            # If continuous
            # To avoid divide by zero error, use that lim x->0 sin(a*x)/x = a
            sincTerm = np.zeros(m)
            
            sincTerm[mask_valid] = np.sin(Omega[i,mask_valid]*l_box[i])/Omega[i,mask_valid]
            sincTerm[~mask_valid] = l_box[i] 

            z *= 2*np.exp(1j*Omega[i]*c_box[i])*sincTerm
    return z



### TODOS FOR SKETCHING.PY

# Short-term:
#  - Add support of private sketching for the real variants of the considered maps
#  - Add the square nonlinearity, for sketching for PCA for example

# Long-term:
# - Fast sketch computation