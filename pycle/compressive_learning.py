"""Contains compressive learning algorithms."""

# Main imports
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import nnls, minimize

# We rely on the sketching functions
from .sketching import SimpleFeatureMap
    
##########################
### 1: CL-OMPR for GMM ###
##########################

def CLOMPR_GMM(sketch,featureMap,K,bounds = None,nIterations = None,bestOfRuns=1,GMMoutputFormat=True,verbose=0):
    """Learns a Gaussian Mixture Model (GMM) from the complex exponential sketch of a dataset ("compressively").
    The sketch given in argument is asumed to be of the following form (x_i are examples in R^d):
        z = (1/n) * sum_{i = 1}^n exp(j*[Omega*x_i + xi]),
    and the Gaussian Mixture to estimate will have a density given by (N is the usual Gaussian density):
        P(x) = sum_{k=1}^K alpha_k * N(x;mu_k,Sigma_k)       s.t.      sum_k alpha_k = 1.
    
    Arguments:
        - sketch: (m,)-numpy array of complex reals, the sketch z of the dataset to learn from
        - featureMap, the sketch the sketch featureMap (Phi), provided as either:
            -- a SimpleFeatureMap object (i.e., complex exponential or universal quantization periodic map)
            -- (Omega,xi): tuple with the (d,m) Fourier projection matrix and the (m,) dither (see above)
        - K: int > 0, the number of Gaussians in the mixture to estimate
        - bounds: (lowb,uppd), tuple of are (d,)-np arrays, lower and upper bounds for the centers of the Gaussians.
                  By default (if bounds is None), the data is assumed to be normalized in the box [-1,1]^d.
        - nIterations: int >= K, maximal number of iterations in CL-OMPR (default = 2*K).
        - bestOfRuns: int (default 1). If >1 returns the solution with the smallest residual among that many independent runs.
        - GMMoutputFormat: bool (defalut True), if False the output is not as described below but a list of atoms (for debug)
        - verbose: 0,1 or 2, amount of information to print (default: 0, no info printed). Useful for debugging.
        
        
    Returns: a tuple (w,mus,Sigmas) of three numpy arrays:
        - alpha:  (K,)   -numpy array containing the weigths ('mixing coefficients') of the Gaussians
        - mus:    (K,d)  -numpy array containing the means of the Gaussians
        - Sigmas: (K,d,d)-numpy array containing the covariance matrices of the Gaussians
    """
    
    ## 0) Defining all the tools we need
    ####################################
    ## 0.1) Handle input
    ## 0.1.1) sketch feature function
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
        raise ValueException('The featureMap argument does not match one of the supported formats.')
    
    ## 0.1.2) nb of iterations
    if nIterations is None:
        nIterations = 2*K # By default: CLOMP-*R* (repeat twice)
    
    ## 0.1.3) Bounds of the optimization problems
    if bounds is None:
        lowb = -np.ones(d) # by default data is assumed normalized
        uppb = +np.ones(d)
        if verbose > 0: print("WARNING: data is assumed to be normalized in [-1,+1]^d")
    else:
        (lowb,uppb) = bounds # Bounds for one Gaussian center
    # Format the bounds for the optimization solver
    boundstheta = np.array([lowb,uppb]).T.tolist()  # bounds for the means
    varianceLowerBound = 1e-8
    for i in range(d): boundstheta.append([varianceLowerBound,None]) # bounds for the variance
    
    
    ## 0.2) util functions to store the atoms easily
    def stacktheta(mu,sigma):
        '''Stacks all the elements of one atom (mean and diagonal variance of a Gaussian) into one atom vector'''
        return np.append(mu,sigma)

    def destacktheta(th):
        '''Splits one atom (mean and diagonal variance of a Gaussian) into mean and variance separately'''
        mu = th[:d]
        sigma = th[-d:]
        return (mu,sigma)
    
    def stackTheta(Theta,alpha):
        '''Stacks *all* the atoms and their weights into one vector'''
        (nbthetas,thetadim) = Theta.shape
        p = np.empty((thetadim+1)*nbthetas)
        for i in range(nbthetas):
            theta_i = Theta[i]
            p[i*thetadim:(i+1)*thetadim] = theta_i
        p[-nbthetas:] = alpha
        return p

    def destackTheta(p):
        thetadim = 2*d # method-dependent
        nbthetas = int(p.shape[0]/(thetadim+1))
        Theta = p[:thetadim*nbthetas].reshape(nbthetas,thetadim)
        alpha = p[-nbthetas:]
        return (Theta,alpha)

    ## 0.3) sketch of a Gaussian A(P_theta) and its gradient wrt theta
    def sketchOfGaussian(mu,Sigma,Omega):
        '''returns a m-dimensional complex vector'''
        return scst*np.exp(1j*(mu@Omega) -np.einsum('ij,ij->i', np.dot(Omega.T, Sigma), Omega.T)/2.)*np.exp(1j*xi) # black voodoo magic to evaluate om_j^T*Sig*om_j forall j

    def gradMuSketchOfGaussian(mu,Sigma,Omega):
        '''returns a d-by-m-dimensional complex vector'''
        return scst*1j*Omega*sketchOfGaussian(mu,Sigma,Omega)
    
    def gradSigmaSketchOfGaussian(mu,Sigma,Omega):
        '''returns a d-by-m-dimensional complex vector'''
        return -scst*0.5*(Omega**2)*sketchOfGaussian(mu,Sigma,Omega)
    
    def Apth(th): # computes sketh from one atom th
        mu,sig = destacktheta(th)
        return sketchOfGaussian(mu,np.diag(sig),Omega)


    ## 0.4) functions that compute the cost and gradient of the optimization sub-problems
    def step1funGrad(th,r): 
        mu,sig = destacktheta(th)
        Sig = np.diag(sig)
        Apth = sketchOfGaussian(mu,Sig,Omega)
        ApthNrm = np.linalg.norm(Apth)
        jacobMu = gradMuSketchOfGaussian(mu,Sig,Omega)
        jacobSi = gradSigmaSketchOfGaussian(mu,Sig,Omega)

        # To avoid division by zero, trick (doesn't change anything because everything will be zero)
        if np.isclose(ApthNrm,0):
            if verbose > 1: print('ApthNrm is too small ({}), change it to 1e-5.'.format(ApthNrm))
            ApthNrm = 1e-5
        fun  = -np.real(np.vdot(Apth,r))/ApthNrm # - to have a min problem

        #gradMu = -np.real(jacobMu@(np.eye(m) - np.outer(Apth,Apth)/(ApthNrm**2))@np.conj(r))/(ApthNrm) 
        gradMu = -np.real(jacobMu@np.conj(r))/(ApthNrm) + np.real(np.real(jacobMu@Apth.conj())*(Apth@np.conj(r))/(ApthNrm**3) )
        #gradSi = -np.real(jacobSi@(np.eye(m) - np.outer(Apth,Apth)/(ApthNrm**2))@np.conj(r))/(ApthNrm) 
        gradSi = -np.real(jacobSi@np.conj(r))/(ApthNrm) + np.real(np.real(jacobSi@Apth.conj())*(Apth@np.conj(r))/(ApthNrm**3) )

        grad = np.append(gradMu,gradSi)
        return (fun,grad)
    
    def step5funGrad(p,z): 
        (Theta,alpha) = destackTheta(p)
        (nbthetas,thetadim) = Theta.shape 
        # Compute atoms
        A = np.empty([m,0])
        for theta_i in Theta:
            Apthi = Apth(theta_i)
            A = np.c_[A,Apthi]
            
        r = (z - A@alpha) # to avoid re-computing
        # Function
        fun  = np.linalg.norm(r)**2
        # Gradient
        grad = np.empty((thetadim+1)*nbthetas)
        for i in range(nbthetas):
            theta_i = Theta[i]
            mu,sig = destacktheta(theta_i)
            Sig = np.diag(sig)
            jacobMu = gradMuSketchOfGaussian(mu,Sig,Omega)
            jacobSi = gradSigmaSketchOfGaussian(mu,Sig,Omega)
            grad[i*thetadim:i*thetadim+d] = -2*alpha[i]*np.real(jacobMu@np.conj(r)) # for mu
            grad[i*thetadim+d:(i+1)*thetadim] = -2*alpha[i]*np.real(jacobSi@np.conj(r)) # for sigma
        grad[-nbthetas:] = -2*np.real((z - A@alpha)@np.conj(A)) # Gradient of the weights
        return (fun,grad)
    
    ## THE ACTUAL ALGORITHM
    ####################################
    bestResidualNorm = np.inf 
    bestTheta = None
    bestalpha = None
    for iRun in range(bestOfRuns):
    
        ## 1) Initialization
        r = sketch  # residual
        Theta = np.empty([0,2*d]) # Theta is a nbAtoms-by-atomDimension (= 2*d) array

        ## 2) Main optimization
        for i in range(nIterations):
            ## 2.1] Step 1 : find new atom theta most correlated with residual
            # Initialize the new atom
            mu0 = np.random.uniform(lowb,uppb) # initial mean at random
            sig0 = np.ones(d) # initial covariance matrix is identity TODO DO SOMETHING SMARTER?
            x0 = stacktheta(mu0,sig0)
            # And solve            
            sol = scipy.optimize.minimize(lambda th: step1funGrad(th,r), x0 = x0, args=(), method='L-BFGS-B', jac=True,
                                            bounds=boundstheta, constraints=(), tol=None, options=None) # TODO change constrains?
            
            theta = sol.x
            ## 2.2] Step 2 : add it to the support
            Theta = np.append(Theta,[theta],axis=0)
            ## 2.3] Step 3 : if necessary, hard-threshold to nforce sparsity
            if Theta.shape[0] > K:
                # Construct A = sketchFeatureFunNorm(Theta); TODO avoid rebuilding everything
                A = np.empty([m,0])
                for theta_i in Theta:
                    Apthi = Apth(theta_i)
                    Apthi_norm = np.linalg.norm(Apthi)
                    if Apthi_norm == 0: Apthi_norm = 1. # Avoid /0
                    Apthi = Apthi / Apthi_norm # normalize, unlike step 4
                    A = np.c_[A,Apthi] 
                Ari = np.r_[A.real, A.imag]
                b = sketch
                bri = np.r_[b.real, b.imag] # todo : outside the loop
                (beta,_) = nnls(Ari,bri) # non-negative least squares
                Theta = np.delete(Theta, (np.argmin(beta)), axis=0)
            ## 2.4] Step 4 : project to find weights

            # Construct A = sketchFeatureFunNorm(Theta); TODO avoid rebuilding everything
            # TODO : avoid doing this if we did the computing at step 3?
            A = np.empty([m,0])
            for theta_i in Theta:
                Apthi = Apth(theta_i)
                A = np.c_[A,Apthi]
            Ari = np.r_[A.real, A.imag]
            b = sketch
            bri = np.r_[b.real, b.imag] # todo : outside the loop
            (alpha,res) = nnls(Ari,bri) # non-negative least squares

            ## 2.5] Step 5
            # Initialize at current solution
            x0 = stackTheta(Theta,alpha)
            # Compute the bounds for step 5 : boundsOfOneAtom * numberAtoms then boundsOneWeight * numberAtoms
            boundsThetaAlpha = boundstheta * Theta.shape[0] + [[0,None]] * Theta.shape[0]
            # Solve
            sol = scipy.optimize.minimize(lambda p: step5funGrad(p,sketch), x0 = x0, args=(), method='L-BFGS-B', jac=True,
                                            bounds=boundsThetaAlpha, constraints=(), tol=None, options=None) # TODO ADD BOUNDS change constrains
            (Theta,alpha) = destackTheta(sol.x)

            A = np.empty([m,0])
            for theta_i in Theta:
                Apthi = Apth(theta_i)
                A = np.c_[A,Apthi]
            r = sketch - A@alpha

        ## 3) Finalization
        # Normalize alpha
        alpha /= np.sum(alpha)
    
    
        runResidualNorm = np.linalg.norm(sketch - A@alpha)
        if verbose>1: print('Run {}, residual norm is {} (best: {})'.format(iRun,runResidualNorm,bestResidualNorm))
        if runResidualNorm < bestResidualNorm:
            bestResidualNorm = runResidualNorm
            bestTheta = Theta
            bestalpha = alpha
            

    if GMMoutputFormat:
        return ThetasToGMM(bestTheta,bestalpha)
    # Else return in "atomic" format
    return (bestTheta,bestalpha)



def ThetasToGMM(Th,al):
    """util function, converts the output of CL-OMPR to a (weights,centers,covariances)-tuple (GMM encoding in this notebook)"""
    (K,d2) = Th.shape
    d = int(d2/2)
    clompr_mu = np.zeros([K,d])
    clompr_sigma = np.zeros([K,d,d])
    for k in range(K):
        clompr_mu[k] = Th[k,0:d]
        clompr_sigma[k] = np.diag(Th[k,d:2*d])
    return (al/np.sum(al),clompr_mu,clompr_sigma)

##############################
### 2: CL-OMPR for K-means ###
##############################

def CLOMPR_CKM(sketch,featureMap,K,bounds = None,nIterations = None,bestOfRuns = 1, verbose = 0):
    """Learns a set of K centroids "compressively" from the provided sketch of a dataset.
    The sketch given in argument is asumed to be of the following form (x_i are examples in R^d):
        z = (1/n) * sum_{i = 1}^n Phi(x_i),
    and will be "matched" to the sketch of the weighted centroids:
        A(P_centroids) = sum_{k = 1}^K alpha_k * Phi(c_k).
    
    Arguments:
        - sketch: (m,)-numpy array of complex reals, the sketch z of the dataset to learn from
        - featureMap, the sketch the sketch featureMap (Phi), provided as either:
            -- a FeatureMap object (i.e., complex exponential or universal quantization periodic map)
            -- (fun,grad,d): tuple with the Phi function, its gradient, and the ambient dimension (deprecated, used for old code)
        - K: int > 0, the number of Gaussians in the mixture to estimate
        - bounds: (lowb,uppd), tuple of are (d,)-np arrays, lower and upper bounds for the centroids.
                  By default (if bounds is None), the data is assumed to be normalized in the box [-1,1]^d.
        - nIterations: int >= K, maximal number of iterations in CL-OMPR (default = 2*K).
        - bestOfRuns: int (default 1) (NOT YET IMPLEMENTED, DOESN'T DO ANYTHING FOR NOW)
        - verbose: 0,1 or 2, amount of information to print (default: 0, no info printed). Useful for debugging.
        
    Returns: a tuple (w,mus,Sigmas) of two numpy arrays:
        - alpha:  (K,)   -numpy array containing the weigths ('mixing coefficients') of the clusters
        - mus:    (K,d)  -numpy array containing the means of the Gaussians
    """
    ## 0) Defining all the tools we need
    ####################################
    ## 0.1) Handle input
    ## 0.1.1) sketch feature function
    normalized = False # Set to true when ||Phi(x)||_2 = cst. for all x (as e.g. complex exponential sketch)
    if isinstance(featureMap,SimpleFeatureMap):
        sketchFeatureFun = featureMap # or featureMap.__call__, should have same effect
        sketchFeatureGrad = featureMap.grad
        d = featureMap.d
        m = featureMap.m
        normalized = featureMap.name.lower() == "complexexponential"
    elif isinstance(featureMap,tuple):
        (sketchFeatureFun,sketchFeatureGrad,d) = featureMap
        m = sketch.size
    else:
        raise ValueException('The featureMap argument does not match one of the supported formats.')
        
    # Simplify in the case of normalized inputs
    if normalized: # If the sketch of an atom has always the same norm (e.g. in (Q)CKM)
        sketchFeatureFunNorm  = sketchFeatureFun
        sketchFeatureGradNorm = sketchFeatureGrad
    else:
        #sketchFeatureFunNorm = # TODO
        #sketchFeatureGradNorm = # TODO
        raise NotImplementedError 
        
    ## 0.1.2) nb of iterations
    if nIterations is None:
        nIterations = 2*K # By default: CLOMP-*R* (repeat twice)
    
    ## 0.1.3) Bounds of the optimization problems
    if bounds is None:
        lowb = -np.ones(d) # by default data is assumed normalized
        uppb = +np.ones(d)
        if verbose > 0: print("WARNING: data is assumed to be normalized in [-1,+1]^d")
    else:
        (lowb,uppb) = bounds # Bounds for one centroid
    # Format the bounds for the optimization solver
    bounds = np.array([lowb,uppb]).T.tolist()
    
    
    ## 0.2) util functions to store the atoms easily
    def stackTheta(Theta,alpha):
        (K,d) = Theta.shape
        p = np.empty((d+1)*K)
        for i in range(K):
            theta_i = Theta[i]
            p[i*d:(i+1)*d] = theta_i
        p[-K:] = alpha
        return p

    def destackTheta(p):
        K = int(p.shape[0]/(d+1))
        Theta = p[:d*K].reshape(K,d)
        alpha = p[-K:]
        return (Theta,alpha)
    
    ## 0.3) functions that compute the cost and gradient of the optimization sub-problems
    def step1funGrad(c,r):
        fun  = -np.real(np.vdot(sketchFeatureFunNorm(c),r)) # - to have a min problem
        grad = -np.real(sketchFeatureGradNorm(c)@np.conj(r))
        return (fun,grad)
    
    def step5funGrad(p,z):
        (Theta,alpha) = destackTheta(p)
        (K,d) = Theta.shape
        # Compute atoms
        A = np.empty([m,0])
        for theta_i in Theta:
            A = np.c_[A,sketchFeatureFun(theta_i)]
        r = (z - A@alpha) # to avoid re-computing
        # Function
        fun  = np.linalg.norm(r)**2
        # Gradient
        grad = np.empty((d+1)*K)
        for i in range(K):
            theta_i = Theta[i]
            grad[i*d:(i+1)*d] = -2*alpha[i]*np.real(sketchFeatureGrad(theta_i)@np.conj(r))
        grad[-K:] = -2*np.real((z - A@alpha)@np.conj(A))
        return (fun,grad)
    
    
    ## 1) Initialization
    r = sketch  # residual
    Theta = np.empty([0,d]) # Theta is a nbAtoms-by-atomDimension array
    
    ## 2) Main optimization
    for i in range(nIterations):
        ## 2.1] Step 1 : find new atom theta most correlated with residual
        x0 = np.random.uniform(lowb,uppb)
        sol = scipy.optimize.minimize(lambda th: step1funGrad(th,r), x0 = x0, args=(), method='L-BFGS-B', jac=True,
                                        bounds=bounds, constraints=(), tol=None, options=None) # change constrains
        theta = sol.x
        # TODO : fix bounds, constraints, x0, check args, opts, tol
        ## 2.2] Step 2 : add it to the support
        Theta = np.append(Theta,[theta],axis=0)
        ## 2.3] Step 3 : if necessary, hard-threshold to nforce sparsity
        if Theta.shape[0] > K:
            # Construct A = sketchFeatureFunNorm(Theta); TODO avoid rebuilding everything
            A = np.empty([m,0])
            for theta_i in Theta:
                A = np.c_[A,sketchFeatureFunNorm(theta_i)]
            Ari = np.r_[A.real, A.imag]
            b = sketch
            bri = np.r_[b.real, b.imag] # todo : outside the loop
            (beta,_) = nnls(Ari,bri) # non-negative least squares
            Theta = np.delete(Theta, (np.argmin(beta)), axis=0)
        ## 2.4] Step 4 : project to find weights
        
        # Construct A = sketchFeatureFunNorm(Theta); TODO avoid rebuilding everything
        # TODO : avoid doing this if we did the computing at step 3?
        A = np.empty([m,0])
        for theta_i in Theta:
            A = np.c_[A,sketchFeatureFun(theta_i)]
        Ari = np.r_[A.real, A.imag]
        b = sketch
        bri = np.r_[b.real, b.imag] # todo : outside the loop
        (alpha,res) = nnls(Ari,bri) # non-negative least squares
        
        ## 2.5] Step 5
        x0 = stackTheta(Theta,alpha)
        sol = scipy.optimize.minimize(lambda p: step5funGrad(p,sketch), x0 = x0, args=(), method='L-BFGS-B', jac=True,
                                        bounds=None, constraints=(), tol=None, options=None) # TODO ADD BOUNDS change constrains
        (Theta,alpha) = destackTheta(sol.x)
        
        A = np.empty([m,0])
        for theta_i in Theta:
            A = np.c_[A,sketchFeatureFun(theta_i)]
        r = sketch - A@alpha
    
    ## 3) Finalization
    # Normalize alpha
    alpha /= np.sum(alpha)
    return (alpha,Theta)




# TODO COMPRESSIVE_LEARNING in rough importance order
# - implement CKM with normed Phi
# - add bounds where they are still lacking
# - support multiple runs in CKM
# - smarter initialization of the covariance in CLOMPR-GMM
# - add verbose that makes sense in both functions
# - make code more efficient by avoiding re-computing the selected columns at each iteration (in both functions)
# - support for nondiagonal covariances in the GMM fitting (how?? opt. on manifolds?)
# - investigate if auto-differentiation might not be smarter