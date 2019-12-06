"""Contains tools to compute the sketch the dataset."""

# Main imports
import numpy as np
import matplotlib.pyplot as plt # For verbose
import scipy.optimize
import sys          # For error handling

NUMBA_INSTALLED = True
try:
    import numba
except ImportError:
    NUMBA_INSTALLED = False
    
    
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
    '''draws frequencies according to some sampling pattern''' # add good specs
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
    SigFact = np.linalg.inv(np.linalg.cholesky(Sigma)) # TO CHECK
    
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
   
def pdfAdaptedRadius(r):
    '''up to a constant'''
    return np.sqrt(r**2 + (r**4)/4)*np.exp(-(r**2)/2) 

def drawFrequencies_AdaptedRadius(d,m,Sigma = None):
    '''draws frequencies according to some sampling pattern
    omega = R*Sigma^{-1/2}*phi, for R from adapted with variance 1, phi uniform''' 
    if Sigma is None:
        Sigma = np.identity(n)
        
    # Sample the radii
    r = np.linspace(0,4,1001) # what are the best params? this seems reasonable
    R = sampleFromPDF(pdfAdaptedRadius(r),r,nsamples=m)
    
    phi = np.random.randn(d,m)
    phi = phi / np.linalg.norm(phi,axis=0) # normalize -> randomly sampled from unit sphere
    SigFact = np.linalg.inv(np.linalg.cholesky(Sigma)) # TO CHECK
    
    Om = SigFact@phi*R 
    
    return Om

# General function for convenience
def drawFrequencies(drawType,d,m,Sigma = None):
    """Draw the 'frequencies' or projection matrix Omega for sketching.
    
    Arguments:
        - drawType: a string indicating the sampling pattern (Lambda) to use, one of the following:
            -- "gaussian"       or "G"  : Gaussian sampling > Lambda = N(0,Sigma^{-1})
            -- "foldedGaussian" or "FG" : Folded Gaussian sampling (i.e., the radius is Gaussian)
            -- "adaptedRadius"  or "AR" : Adapted Radius heuristic
        - d: int, dimension of the data to sketch
        - m: int, number of 'frequencies' to draw (the target sketch dimension)
        - Sigma: (d,d)-numpy array, the covariance of the data (note that we typically use Sigma^{-1}).
                 If Sigma is None (default), we assume that data was normalized s.t. Sigma = identity.
        
    Returns:
        - Omega: (d,m)-numpy array containing the 'frequency' projection matrix
    """
    if Sigma is None:
        Sigma = np.identity(d)
    if drawType.lower() in ["drawfrequencies_gaussian","gaussian","g"]:
        drawFunc = drawFrequencies_Gaussian
    elif drawType.lower() in ["drawfrequencies_foldedgaussian","foldedgaussian","folded_gaussian","fg"]:
        drawFunc = drawFrequencies_Gaussian
    elif drawType.lower() in ["drawfrequencies_adapted","adaptedradius","adapted_radius","ar"]:
        drawFunc = drawFrequencies_AdaptedRadius
    return drawFunc(d,m,Sigma)

# The following funtion allows to estimate Sigma
def estimate_Sigma(dataset,m0,c=20,n0=None,nIterations=5,verbose=0):
    """Automatically estimates the "Sigma" parameter (the scale of data clusters) for generating the sketch operator.
    
    We assume here that Sigma = sigma2_bar * identity matrix. 
    To estimate sigma2_bar, lightweight sketches of size m0 are generated from (a small subset of) the dataset
    with candidate values for sigma2_bar. Then, sigma2_bar is updated by fitting a Gaussian
    to the absolute values of the obtained sketch. Cfr. https://arxiv.org/pdf/1606.02838.pdf, sec 3.3.3.
    
    Arguments:
        - dataset: (n,d) numpy array, the dataset X: n examples in dimension d
        - m0: int, number of candidate 'frequencies' to draw (can be typically smaller than m).
        - c:  int (default 20), number of 'boxes' (i.e. number of maxima of sketch absolute values to fit)
        - n0: int or None, if given, n0 samples from the dataset are subsampled to be used for Sigma estimation
        - nIterations: int (default 5), the maximum number of iteration (typically stable after 2 iterations)
        - verbose: 0,1 or 2, amount of information to print (default: 0, no info printed). Useful for debugging.
        
    Returns:
        - Sigma: (d,d)-numpy array, the (diagonal) estimated covariance of the clusters in the dataset
    """
    # TODOS:
    # - allow K > 1
    # - estimate nonisotropic Sigma?
    
    (n,d) = dataset.shape
    # X is the subsampled dataset containing only n0 examples
    if n0 is not None and n0<n:
        X = dataset[np.random.choice(n,n0,replace=False)]
    else:
        X = dataset
    
    # Initialization
    sigma2_bar = 1
    K = 1 # For later extension
    s = m0//c # number of freqs per box
    drawFreq_type = "AR" # To change? 
    
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
    def _callback(p):
        p[:K] /= np.sum(p[:K])
    
    # Bounds of the optimization problem
    bounds = []
    for k in range(K): bounds.append([1e-5,1]) # bounds for the weigths
    for k in range(K): bounds.append([1e-12,1e12]) # bounds for the sigmas

    # Actual algorithm
    for i in range(nIterations):
        # Draw frequencies according to current estimate 
        Omega0 = drawFrequencies(drawFreq_type,d,m0,Sigma = sigma2_bar*np.eye(d)) # To update K > 1
        
        # Sort the frequencies
        Rs = np.linalg.norm(Omega0,axis=0)
        i_sort = np.argsort(Rs)
        Omega0 = Omega0[:,i_sort]
        Rs = Rs[i_sort]
        
        # Compute unnormalized complex exponential sketch
        Phi0 = SimpleFeatureMap("ComplexExponential",Omega0)
        z0 = computeSketch(X,Phi0) 
        
        # find the indices of the max of each block
        jqs = np.empty(c) 
        for ic in range(c):
            j_max = np.argmax(np.abs(z0)[ic*s:(ic+1)*s]) + ic*s
            jqs[ic] = j_max
        jqs = jqs.astype(int)
        R_tofit = Rs[jqs]
        z_tofit = np.abs(z0)[jqs]
        
        # Plot if required
        if verbose > 1:
            plt.figure(figsize=(10,5))
            plt.plot(Rs,np.abs(z0),'.')
            plt.plot(Rs[jqs],np.abs(z0)[jqs],'.')
            plt.xlabel('R')
            plt.ylabel('|z|')
            plt.show()
        
        # Set up the fitting opt. problem
        f = lambda p: _fun_grad_fit_sigmas(p,R_tofit,z_tofit) # cost
        
        p0 = np.zeros(2*K) # initial point
        p0[:K] = np.ones(K)/K # w
        p0[K:] = np.random.uniform(0.5,1.5,K)/(np.median(R_tofit)**2) # sig2, heuristic to have good gradient at start
    
        # Solve the sigma^2 optimization problem
        sol = scipy.optimize.minimize(f, p0,jac = True, bounds = bounds,callback=_callback)
        p = sol.x
        w = np.array(p[:K])#/np.sum(p[:K])
        sig2 = np.array(p[K:])
        
        # Plot if required
        if verbose > 1:
            rfit = np.linspace(0,Rs.max(),100)
            zfit = np.zeros(rfit.shape)
            for k in range(K):
                zfit += w[k]*np.exp(-(sig2[k]*rfit**2)/2.)
            plt.plot(Rs,np.abs(z0),'.')
            plt.plot(R_tofit,z_tofit,'.')
            plt.plot(rfit,zfit)
            plt.xlabel('R')
            plt.ylabel('|z|')
            plt.show()
            
        # Update
        sigma2_bar = sig2[0]
        
    # Show final fit
    if verbose > 0:
        rfit = np.linspace(0,Rs.max(),100)
        zfit = np.zeros(rfit.shape)
        for k in range(K):
            zfit += w[k]*np.exp(-(sig2[k]*rfit**2)/2.)
        plt.plot(Rs,np.abs(z0),'.')
        plt.plot(R_tofit,z_tofit,'.')
        plt.plot(rfit,zfit)
        plt.xlabel('R')
        plt.ylabel('|z|')
        plt.legend(['abs. values of sketch','max abs values on blocks','fitted Gaussian'])
        plt.show()


    Sigma = sigma2_bar*np.eye(d)
    
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
    
def _sawtoothWave(t,T=2*np.pi,centering=True):
    if centering:
        return ( t % T )/T*2-1 
    else:
        return ( t % T )/T # centering=false => quantization is between 0 and +1
    
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
    "cosine": (lambda x: np.cos(x),lambda x: np.sin(x))
 }

# 2.2: in development, use numba to speed up sketching: 
# Instantiate the RFF sketch feature map
def generateRRFmap(Omega,xi = None,use_numba = True,return_gradient = True,normalize = False):
    """
    Returns a function computing the (complex) random Fourier features and its gradient:
        RFF(x) = exp(i*(Omega*x + xi))
    where i is the imaginary unit, Omega and xi are provided. Uses numba acceleration by default.
        
    Arguments:
        
    Returns:
    """
    
    if normalize:
        c_norm = 1./np.sqrt(Omega.shape[1]) # 1/sqrt(m)
    else:
        c_norm = 1.
    
    if xi is None:
        xi = np.zeros(Omega.shape[1])
    
    def _RFF(x):
        return c_norm*np.exp(1j*(np.dot(Omega.T,x) + xi))

    def _grad_RFF(x):
        return 1j*c_norm*np.exp(1j*(np.dot(Omega.T,x) + xi))*Omega
    
    if use_numba and not NUMBA_INSTALLED:
        use_numba = False # Numba was not found, we can't use it
        print('Warning: numba not found, falling back to python. Recommended to install numba.')
    
    # Use a numba wrapper around the functions
    if use_numba:
        RFF = numba.jit(nopython=True)(_RFF)
        grad_RFF = numba.jit(nopython=True)(_grad_RFF) # No gain??
    else:
        RFF = _RFF
        grad_RFF = _grad_RFF
        
    # Return RFF map with its gradient if needed
    if return_gradient:
        return (RFF,grad_RFF)
    else:
        return RFF


# 2.3 FeatureMap objects
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
        self.c_norm = c_norm
        
    # magic operator to be able to call the FeatureMap object as a function
    def __call__(self,x): 
        return self.c_norm*self.f(np.dot(self.Omega.T,x) + self.xi) # Evaluate the feature map at x
    
    def grad(self,x):
        """Gradient (Jacobian matrix) of Phi, as a (d,m)-numpy array"""
        return self.c_norm*self.f_grad(np.dot(self.Omega.T,x) + self.xi)*self.Omega
    

#######################################
### 3: Actual sketching functions   ###
#######################################    

#################################
# 3.1 GENERAL SKETCHING ROUTINE #
#################################
def computeSketch(dataset, featureMap, datasetWeigths = None):
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
    if datasetWeigths is None:
        for i in range(n):
            sketch = sketch + featureMap(dataset[i])/n
    else:
        # TODO: fix this commented implementation (crashes in certain cases, temporarily replaced by for loop)
        # sketch = datasetWeigths@featureMap(X) 
        for i in range(n):
            sketch = sketch + featureMap(dataset[i])*datasetWeigths[i]
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
                    -- 'universalQuantization' (f(t) = sign(exp(i*t)))
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
        raise ValueException('The featureMap argument does not match one of the supported formats.')
    
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
    elif featureMapName.lower() == 'universalquantization': # Assuming normalized in [-1,+1], TODO check real/complex case?
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
            if epsilon >= 1: raise Error('WARNING: with epsilon >= 1 the sigma bound doesn\'t hold! Privacy is NOT ensured!')
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
    
    

### TODOS FOR SKETCHING.PY

# Short-term:
#  - Add support of private sketching for the real variants of the considered maps
#  - Add the square nonlinearity, for sketching for PCA for example

# Long-term:
# - Fast sketch computation and clean numba if not needed