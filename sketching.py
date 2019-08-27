# General imports
import numpy as np

### 1: General sketching function
def computeSketch(X, sketchFun, sketchDim, datasetWeigths = None):
    """Computes the general sketch of a dataset.
    
    Arguments:
        - 
        
    Returns: a tuple (X,weigths,means,covariances) of four numpy arrays
        - 
    """
    # TODO :
    # + BASIC IMPLEMENTATION
    # + allow for weigthed sketch
    # - modify to allow for parallel computing
    # - add possibility to specify classes
    # - defensive programming
    # - write the docstring
    (N,d) = X.shape # number samples, dimension    
    
    if datasetWeigths is None:
        sketch = np.zeros(sketchDim)
        for i in range(N):
            sketch = sketch + sketchFun(X[i])/N
    else:
        # TODO efficient
        # sketch = datasetWeigths@sketchFun(X) (crashes)
        sketch = np.zeros(sketchDim)
        for i in range(N):
            sketch = sketch + sketchFun(X[i])*datasetWeigths[i]
    return sketch

def sensisitivty_sketch(m,n,c_normalization = 1.,sketchFeatureFunction = 'complexExponential',DPdef = 'replace',c_xi = 1,sensitivity_type = 1):
    """Computes the L1 sensitity of a sketching function, based on our theoretical results.
    The noisy sketch operator A(X) is given by (xi is Laplacian noise)
        A(X) := (1/n)*[sum_{x_i in X} cst*f(x_i)] + c_xi * xi
    
    Arguments:
        - m: int, the sketch dimension
        - n: int, number of sketch contributions being averaged (n = 1 to have local DP, or sum instead of average)
        - c_normalization: real, the constant cst before the sketch feature function (e.g.: 1. (default) or 1./sqrt(m))
        - sketchFeatureFunction: string, name of sketch feature function f, values allowed:
            -- 'complexExponential' (f(x) = exp(i*x))
            -- 'universalQuantization'
        - DPdef: string, name of the Differential Privacy variant considered, i.e. the neighbouring relation ~:
            -- 'remove', 'add', 'remove/add' or 'standard': D~D' iff D' = D U {x'} (or vice versa) 
            -- 'replace': D~D' iff D' = D \ {x} U {x'} (or vice versa) 
        - c_xi: real, the constant c_xi before the added noise (e.g.: 1. (default) or m/r)
        - sensitivity_type: int, 1 (default) for L1 sensitivity, 2 for L2 sensitivity.
        
        
    Returns: a real, the L1 sensitivity of the sketching operator defined above
    """
    if sketchFeatureFunction is 'complexExponential':
        c_feat = np.sqrt(2) if (sensitivity_type == 1) else 1. # TODO more careful
    elif sketchFeatureFunction is 'universalQuantization': # Assuming normalized in -1,+1, TODO check real/complex?
        c_feat = 2.         if (sensitivity_type == 1) else np.sqrt(2)
    elif sketchFeatureFunction is None:
        c_feat = 1. # ??
    else:
        raise Exception('Unknown sketchFeatureFunction: {}'.format(sketchFeatureFunction))
        
    if DPdef in ['remove','add','remove/add','standard']:
        c_DP = None # Todo implement
    elif DPdef is 'replace':
        c_DP = 1.
    else:
        raise Exception('Unknown DPdef: {}'.format(DPdef))
        
    if sensitivity_type == 1:
        S = 2*m*c_feat*c_normalization*c_DP/(n*c_xi)
    elif sensitivity_type == 2:
        S = 2*np.sqrt(m)*c_feat*c_normalization*c_DP/(n*c_xi)
    else:
        raise Exception('Unknown sensitivity_type: {}'.format(sensitivity_type))
    
    return S


def computeSketch_DP(X, sketchFun, sketchDim, epsilon, delta = 0, c_normalization = 1.,sketchFeatureFunction = 'complexExponential',DPdef = 'replace',c_xi = 1.):
    """TODO"""
    # TODO what about the real case?
    # Compute the usual sketch
    (n,d) = X.shape
    z_clean = computeSketch(X, sketchFun, sketchDim)
    
    if delta > 0:
        # Gaussian mechanism
        S = sensisitivty_sketch(sketchDim,n,c_normalization,sketchFeatureFunction,DPdef,c_xi,sensitivity_type = 2) # L2
        sigma = np.sqrt(2*np.log(1.25/delta))*S/epsilon
        noise = np.random.normal(scale = sigma, size=sketchDim) + 1j*np.random.normal(scale = sigma, size=sketchDim) # todoreal
    else: 
        # Laplacian mechanism
        S = sensisitivty_sketch(sketchDim,n,c_normalization,sketchFeatureFunction,DPdef,c_xi,sensitivity_type = 1) # L1
        beta = S/epsilon # L1 sensitivity/espilon
        noise = np.random.laplace(scale = beta, size=sketchDim) + 1j*np.random.laplace(scale = beta, size=sketchDim) 
        
    #print(np.linalg.norm(noise))
    
    
    return z_clean + noise


### 2: Frequency sampling functions
def drawDithering(m,bounds = None):
    if bounds is None:
        (lowb,highb) = (0,2*np.pi)
    else:
        (lowb,highb) = bounds
    return np.random.uniform(low=lowb,high=highb,size=m)

def drawFrequencies_Gaussian(n,m,Sigma = None):
    '''draws frequencies according to some sampling pattern''' # add good specs
    if Sigma is None:
        Sigma = np.identity(n)
    Om = np.random.multivariate_normal(np.zeros(n), np.linalg.inv(Sigma), m).T # inverse of sigma
    return Om

def drawFrequencies_FoldedGaussian(n,m,Sigma = None):
    '''draws frequencies according to some sampling pattern
    omega = R*Sigma^{-1/2}*phi, for R from folded Gaussian with variance 1, phi uniform''' 
    if Sigma is None:
        Sigma = np.identity(n)
    R = np.abs(np.random.randn(m)) # folded standard normal distribution radii
    phi = np.random.randn(n,m)
    phi = phi / np.linalg.norm(phi,axis=0) # normalize -> randomly sampled from unit sphere
    SigFact = np.linalg.inv(np.linalg.cholesky(Sigma)) # TO CHECK
    
    Om = SigFact@phi*R
    
    return Om


def sampleFromPDF(pdf,x,nsamples=1):
    '''x is a vector (the support of the pdf), pdf is the values of pdf eval at x'''
    
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


def drawFrequencies_AdaptedRadius(n,m,Sigma = None):
    '''draws frequencies according to some sampling pattern
    omega = R*Sigma^{-1/2}*phi, for R from adapted with variance 1, phi uniform''' 
    if Sigma is None:
        Sigma = np.identity(n)
        
    # Sample the radii
    r = np.linspace(0,4,1001) # what are the best params? this seems reasonable
    R = sampleFromPDF(pdfAdaptedRadius(r),r,nsamples=m)
    
    phi = np.random.randn(n,m)
    phi = phi / np.linalg.norm(phi,axis=0) # normalize -> randomly sampled from unit sphere
    SigFact = np.linalg.inv(np.linalg.cholesky(Sigma)) # TO CHECK
    
    Om = SigFact@phi*R # TODO chek dims!!
    
    return Om



### 3: Specific sketch feature functions
# Sketch nonlinearities instantiation(s)
def universalQuantization(t,Delta=np.pi,centering=True):
    if centering:
        return ( (t // Delta) % 2 )*2-1 # // stands for "int division
    else:
        return ( (t // Delta) % 2 ) # centering=false => quantization is between 0 and +1
    
def sawtoothWave(t,T=2*np.pi,centering=True):
    if centering:
        return ( t % T )/T*2-1 
    else:
        return ( t % T )/T # centering=false => quantization is between 0 and +1
    
def triangleWave(t,T=2*np.pi):
    return (2*(t % T)/T ) - (4*(t % T)/T - 2)*( (t // T) % 2 ) - 1

def complexExponential(t,T=2*np.pi):
    return np.exp(1j*(2*np.pi)*t/T)

def fourierSeriesEvaluate(t,coefficients,T=2*np.pi):
    """T = period
    coefficients = F_{-K}, ... , F_{-1}, F_{0}, F_{1}, ... F_{+K}"""
    K = (coefficients.shape[0]-1)/2
    ks = np.arange(-K,K+1)
    # Pre-alloc
    ft = np.zeros(t.shape) + 0j
    for i in range(2*int(K)+1):
        ft += coefficients[i]*np.exp(1j*(2*np.pi)*ks[i]*t/T)
    return ft

