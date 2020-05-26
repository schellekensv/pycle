"""Contains compressive learning algorithms."""

# Main imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls, minimize, LinearConstraint
from copy import copy

# For debug
import time

# We rely on the sketching functions
from .sketching import SimpleFeatureMap, fourierSketchOfBox, fourierSketchOfGaussian


###############
## 1: CLOMPR ##
###############


## Utility functions to play with the atom representations
# ========================================================
def _stackAtom(task,*atom_elements):
    '''Stacks all the elements of one atom (e.g., mean and diagonal variance of a Gaussian) into one atom vector'''
    res = []
    for a in atom_elements:
        res = np.append(res,a)
    return res

def _destackAtom(task,theta,d):
    '''Splits one atom (e.g.,mean and diagonal variance of a Gaussian) into mean and variance separately'''
    if task == "kmeans":
        return theta
    elif task == "gmm":
        return (theta[:d],theta[-d:]) # mu = theta[:d], sigma = theta[-d:]
    elif task == "gmm-nondiag":
        return (theta[:d],theta[d:].reshape(d,d)) # mu = theta[:d], Sigma = theta[-d:]
    else:
        raise ValueError

def _stackTheta(task,Theta,alpha):
    '''Stacks *all* the atoms and their weights into one vector'''
    (nbthetas,thetadim) = Theta.shape
    p = np.empty((thetadim+1)*nbthetas)
    for i in range(nbthetas):
        theta_i = Theta[i]
        p[i*thetadim:(i+1)*thetadim] = theta_i
    p[-nbthetas:] = alpha
    return p

def _destackTheta(task,p,d):
    # Get thetadim, method-dependent
    if task == "kmeans":
        thetadim = d
    elif task == "gmm":
        thetadim = 2*d
    elif task == "gmm-nondiag":
        thetadim = (d+1)*d
    else:
        raise ValueError
    nbthetas = int(p.size/(thetadim+1))
    Theta = p[:thetadim*nbthetas].reshape(nbthetas,thetadim)
    alpha = p[-nbthetas:]
    return (Theta,alpha)

def _ThetasToGMM(task,Th,al):
    """util function, converts the output of CL-OMPR to a (weights,centers,covariances)-tuple (GMM encoding in this notebook)"""
    (K,d2) = Th.shape
    if task == "gmm":
        d = d2//2
    elif task == "gmm-nondiag":
        d = int(-0.5 + np.sqrt(0.25 + d2))
    else:
        raise NotImplementedError
    clompr_mu = np.zeros([K,d])
    clompr_sigma = np.zeros([K,d,d])
    for k in range(K):
        clompr_mu[k] = Th[k,0:d]
        if task == "gmm":
            clompr_sigma[k] = np.diag(Th[k,d:2*d])
        elif task == "gmm-nondiag":
            clompr_sigma[k] = Th[k,d:].reshape(d,d)
    return (al/np.sum(al),clompr_mu,clompr_sigma)


## Utility functions to compute the sketch of an atom and its jacobian
# ====================================================================

def _sketchAtom(task,Phi,theta,z_to_ignore):
    """Compute the sketch for an atom theta (size m)"""
    if task == "kmeans":
        z_th = Phi(theta)
    elif task == "gmm":
        (mu,sig) = _destackAtom("gmm",theta,Phi.d)
        z_th = fourierSketchOfGaussian(mu,np.diag(sig),Phi.Omega,Phi.xi,Phi.c_norm)
    elif task == "gmm-nondiag":
        (mu,Sig) = _destackAtom("gmm-nondiag",theta,Phi.d)
        z_th = fourierSketchOfGaussian(mu,Sig,Phi.Omega,Phi.xi,Phi.c_norm)
    else:
        raise ValueError
    return z_th * z_to_ignore


def _jacobian_sketchAtom(task,Phi,theta,z_to_ignore):
    """Compute the jacobian of the sketch for an atom theta (size pxm)"""
    if task == "kmeans":
        grad_z_th = Phi.grad(theta)
    elif task == "gmm":
        (mu,sig) = _destackAtom("gmm",theta,Phi.d)
        z_th = fourierSketchOfGaussian(mu,np.diag(sig),Phi.Omega,Phi.xi,Phi.c_norm)
        grad_z_th = np.zeros((2*Phi.d,Phi.m)) + 1j*np.zeros((2*Phi.d,Phi.m))
        grad_z_th[:Phi.d] = 1j*Phi.Omega * z_th # Jacobian w.r.t. mu
        grad_z_th[Phi.d:] = -0.5*(Phi.Omega**2) * z_th # Jacobian w.r.t. sigma
    elif task == "gmm-nondiag":
        (mu,Sig) = _destackAtom("gmm-nondiag",theta,Phi.d)
        z_th = fourierSketchOfGaussian(mu,Sig,Phi.Omega,Phi.xi,Phi.c_norm)
        grad_z_th = (1+1j)*np.zeros(((Phi.d+1)*Phi.d,Phi.m))
        grad_z_th[:Phi.d] = 1j*Phi.Omega * z_th # Jacobian w.r.t. mu
        for j in range(Phi.m):
            omega2 = -(np.outer(Phi.Omega[:,j],Phi.Omega[:,j]))
            omega2 = omega2 * (np.ones((Phi.d,Phi.d)) - 0.5*np.eye(Phi.d))
            grad_z_th[Phi.d:,j] = np.reshape(omega2,-1) * z_th[j] # Jacobian w.r.t. sigma
    else:
        raise ValueError
    return grad_z_th * z_to_ignore

## Functions to optimize in the subproblems
# =========================================
def _CLOMPR_step1_fun_grad(task,Phi,theta,r,z_to_ignore,verbose=0):
    """Computes the fun. value and grad. of step 1 objective: max_theta <A(P_theta),r> / <A(P_theta),A(P_theta)>"""
    # Firstly, compute A(P_theta)...
    Apth = _sketchAtom(task,Phi,theta,z_to_ignore)
    # ... and its l2 norm
    Apth_norm = np.linalg.norm(Apth)
    # Trick to avoid division by zero (doesn't change anything because everything will be zero)
    if np.isclose(Apth_norm,0):
        if verbose > 1: print('ApthNrm is too small ({}), change it to 1e-6.'.format(Apth_norm))
        Apth_norm = 1e-6

    # We can evaluate the cost function
    fun = -np.real(np.vdot(Apth,r))/Apth_norm # - to have a min problem

    # Secondly, get the Jacobian
    jacobian = _jacobian_sketchAtom(task,Phi,theta,z_to_ignore)
    grad = -np.real(jacobian@np.conj(r))/(Apth_norm) + np.real(np.real(jacobian@Apth.conj())*(Apth@np.conj(r))/(Apth_norm**3) )
    
    return (fun,grad)

def _CLOMPR_step5_fun_grad(task,Phi,p,sketch,z_to_ignore):
    """Computes the fun. value and grad. of step 1 objective: max_theta <A(P_theta),r> / <A(P_theta),A(P_theta)>"""
    # Destack all the atoms
    (Theta,alpha) = _destackTheta(task,p,Phi.d)
    (nbthetas,thetadim) = Theta.shape

    # Construct the A matrix
    A = np.empty([Phi.m,0])
    for theta_i in Theta:
        Apthi = _sketchAtom(task,Phi,theta_i,z_to_ignore)
        A = np.c_[A,Apthi]

    # Residual
    r = (sketch - A@alpha)

    # Cost function
    fun  = np.linalg.norm(r)**2

    # Grad
    grad = np.empty((thetadim+1)*nbthetas)
    for i in range(nbthetas):
        theta_i = Theta[i]
        jacobian_i = _jacobian_sketchAtom(task,Phi,theta_i,z_to_ignore)
        grad[i*thetadim:(i+1)*thetadim] = -2*alpha[i]*np.real(jacobian_i@np.conj(r)) # Gradient of the atoms
    grad[-nbthetas:] = -2*np.real(r@np.conj(A)) # Gradient of the weights

    return (fun,grad)


def _makeGMMfeasible(theta,d):
    
    (mu,Sig) = _destackAtom("gmm-nondiag",theta,d)
        
    (L,U) = np.linalg.eig(Sig)
    L = L - min(0,L.min())
    Sig = U@np.diag(L)@U.T
    
    return _stackAtom("gmm-nondiag",mu,Sig)


## Main function for CLOMPR
# =========================
def CLOMPR(task,sketch,featureMap,K,bounds,dimensions_to_consider=None, nb_cat_per_dim=None,nIterations=None, nRepetitions=1, ftol=1e-6, verbose=0 ):
    """
    Generic CLOMPR (Compressive Learning with Orthogonal Matching Pursuit with Replacement) algorithm.
    Implements two tasks from a sketch of the dataset: k-means and Gaussian Mixture Model estimation.

    The sketch given in argument is asumed to be of the following form (x_i are the training examples in R^d):
        z = (1/n) * sum_{i = 1}^n Phi(x_i),
    where for GMM it is assumed that Phi = exp(j*[Omega*x_i + xi]), i.e. random Fourier features.
    This sketched is to be "matched" to the sketch of the learned density A(P) = E_{x ~ P} Phi(x), which means:
        - For k-means, P = P_centroids = sum_{k=1}^K alpha_k * delta(x - c_k) a mixture of Dirac deltas;
        - For GMM, P = P_GMM = sum_{k=1}^K alpha_k * N(x;mu_k,Sigma_k);
     in both cases the weigths are nonnegative and sum to one: alpha_k >= 0, sum_k alpha_k = 1.

    Parameters
    ----------
    task: string, defines the task to solve: either "k-means", "gmm" (diagonal covariances) or "gmm-nondiag" (general case).
    sketch: (m,)-numpy array of complex values, the sketch z of the dataset to learn from
    featureMap: the sketching map Phi, provided as a FeatureMap object (must be a complex exponential map for GMM)
    K: int > 0, the number of mixture components (centroids or Gaussians) to estimate
    bounds: (2,d)-np array, lower and upper bounds for the data distribution.

    Additional Parameters
    ---------------------
    nb_cat_per_dim: (d,)-array of ints, the number of categories per dimension for integer data,
                    if its i-th entry = 0 (resp. > 0), dimension i is assumed to be continuous (resp. int.).
                    By default all entries are assumed to be continuous.
    dimensions_to_consider: array of ints (between 0 and d-1), [0,1,...d-1] by default.
                    The box is restricted to the prescribed dimensions.
                    This is helpful to solve problems on a subsets of all dimensions.
    nIterations: int >= K, maximal number of iterations, if > K performs Replacement  (default = 2*K).
    nRepetitions: int (default 1), if > 1 performs that many independent runs and returns the best solution found.
    ftol: real > 0 (def. 1e-6), the tolerance in the optimization sub-problems (rougher tolerance significantly speeds up GMM).
    verbose: 0,1 or 2, amount of information to print (default: 0, no info printed). Useful for debugging.

    Returns
    -------
    A tuple of numpy arrays containing the solution. More specifically,
    - For k-means: a tuple (alpha,centroids) of 2 arrays where
        - alpha:     (K,) -numpy array containing the weigths ('mixing coefficients') of the centroids
        - centroids: (K,d)-numpy array containing the centroids c_k of the clusters
    - For GMM: a tuple (alpha,mus,Sigmas) of 3 arrays where
        - alpha:  (K,)   -numpy array containing the weigths ('mixing coefficients') of the Gaussians
        - mus:    (K,d)  -numpy array containing the means mu_k of the Gaussians
        - Sigmas: (K,d,d)-numpy array containing the covariance matrices Sigma_k of the Gaussians

    """

    ## 0) Defining all the tools we need
    ####################################
    ## 0.1) Handle input

    ## 0.1.1) task name
    if task.lower() in ["km","ckm","kmeans","k-means"]:
        task = "kmeans"
        task_finetuning = "kmeans"
    elif task.lower() in ["gmm","gaussian mixture model"]:
        task = "gmm"
        task_finetuning = "gmm"
    elif task.lower() in ["gmm-nondiag"]:
        task = "gmm"
        task_finetuning = "gmm-nondiag"
    else:
        raise ValueError('The task argument does not match one of the available options.')

    ## 0.1.2) sketch feature function
    if isinstance(featureMap,SimpleFeatureMap):
        d_all = featureMap.d
        m = featureMap.m
    else:
        raise ValueError('The featureMap argument does not match one of the supported formats.')

    # Restrict the dimension
    if dimensions_to_consider is None:
        dimensions_to_consider = np.arange(d_all)
    dimensions_to_ignore = np.delete(np.arange(d_all), dimensions_to_consider)
    d = dimensions_to_consider.size
    
    # Pre-compute sketch of unused dimensions
    z_to_ignore = fourierSketchOfBox(bounds.T,featureMap,nb_cat_per_dim, dimensions_to_consider = dimensions_to_ignore)
    # Compensate the normalization constant and the dithering which will be taken into account in the centroid sketch
    z_to_ignore = z_to_ignore/(featureMap(np.zeros(d_all))) # giving zeros yields the dithering and the c_norm
    
    # Restrict the featureMap for the centroids
    Phi = copy(featureMap) # Don't touch to the inital map, Phi is featureMap restricted to relevant dims
    Phi.d = d
    Phi.Omega = featureMap.Omega[dimensions_to_consider]
    
    ## 0.1.3) nb of iterations
    if nIterations is None:
        nIterations = 2*K # By default: CLOMP-*R* (repeat twice)
    
    ## 0.1.4) Bounds of the optimization problems
    if bounds is None:
        lowb = -np.ones(d) # by default data is assumed normalized
        uppb = +np.ones(d)
        if verbose > 0: print("WARNING: data is assumed to be normalized in [-1,+1]^d")
    else:
        lowb = bounds[0][dimensions_to_consider]
        uppb = bounds[1][dimensions_to_consider] # Bounds for one centroid

    # Format the bounds for the optimization solver
    if task_finetuning == "kmeans":
        boundstheta = np.array([lowb,uppb]).T.tolist()  # bounds for the centroids
    elif task_finetuning == "gmm":
        boundstheta = np.array([lowb,uppb]).T.tolist()  # bounds for the means
        varianceLowerBound = 1e-8
        for i in range(d): boundstheta.append([varianceLowerBound,(uppb[i]-lowb[i])**2]) # bounds for the variance
    elif task_finetuning == "gmm-nondiag":
        # Usual bounds of GMM
        boundstheta = np.array([lowb,uppb]).T.tolist()  # bounds for the means
        varianceLowerBound = 1e-8
        for i in range(d): boundstheta.append([varianceLowerBound,(uppb[i]-lowb[i])**2]) # bounds for the variance
        
        # Bounds for the nondiagonal problem
        boundstheta_finetuning = np.array([lowb,uppb]).T.tolist()  # bounds for the means
        varianceLowerBound = 1e-8
        varianceUpperBound = ((uppb-lowb).max()/2)**2

        _lowb_var = -varianceUpperBound*np.ones((d,d))
        np.fill_diagonal(_lowb_var, varianceLowerBound)
        _uppb_var = +varianceUpperBound*np.ones((d,d))

        _boundsvar = np.append(_lowb_var.reshape(-1),_uppb_var.reshape(-1)).reshape(2,d**2).T.tolist()
        for _i in _boundsvar: boundstheta_finetuning.append(_i) # bounds for the variance
    

    ## 0.1.5) Misc. initializations
    # Chosen method for the optimization solver
    opt_method = 'L-BFGS-B' # could also consider 'TNC'
    # Separated real and imaginary part of the sketch
    sketch_ri = np.r_[sketch.real, sketch.imag]
    if task == "kmeans":
        thetadim = d
    elif task == "gmm":
        thetadim = 2*d
    elif task == "gmm-nondiag":
        thetadim = d*(d+1)


    ## THE ACTUAL ALGORITHM
    #######################
    bestResidualNorm = np.inf 
    bestTheta =  None
    bestalpha = None
    for iRun in range(nRepetitions):
    
        ## 1) Initialization
        r = sketch  # residual
        Theta = np.empty([0,thetadim]) # Theta is a nbAtoms-by-atomDimension array
        A = np.empty([m,0]) # Contains the sketches of the atoms

        ## 2) Main optimization
        for i in range(nIterations):
            ## 2.1] Step 1 : find new atom theta most correlated with residual
            # Initialize the new atom
            if task == "kmeans":
                th_0 = np.random.uniform(lowb,uppb)
            elif task == "gmm":
                mu0 = np.random.uniform(lowb,uppb) # initial mean
                sig0 = (10**np.random.uniform(-0.8,-0.1,d) * (uppb-lowb))**2 # initial covariances
                th_0 = _stackAtom("gmm",mu0,sig0)
                
            elif task == "gmm-nondiag":
                # Solve once for diagonal
                mu0 = np.random.uniform(lowb,uppb) # initial mean
                sig0 = (10**np.random.uniform(-0.8,-0.1,d) * (uppb-lowb))**2 # initial covariances
                th_0 = _stackAtom("gmm",mu0,sig0)
                sol = minimize(lambda th: _CLOMPR_step1_fun_grad("gmm",Phi,th,r,z_to_ignore,verbose),
                                            x0 = th_0, method=opt_method, jac=True,
                                            bounds=boundstheta_gmm)

                (mu0,sig0) = _destackAtom("gmm",sol.x,d)

                th_0 = _stackAtom("gmm-nondiag",mu0,np.diag(sig0))

            # And solve with LBFGS   
            sol = minimize(lambda th: _CLOMPR_step1_fun_grad(task,Phi,th,r,z_to_ignore,verbose),
                                            x0 = th_0, method=opt_method, jac=True,
                                            bounds=boundstheta)
            new_theta = sol.x
            
            if task == "gmm-nondiag":
                new_theta = _makeGMMfeasible(new_theta,d)

            ## 2.2] Step 2 : add it to the support
            Theta = np.append(Theta,[new_theta],axis=0)
            A = np.c_[A,_sketchAtom(task,Phi,new_theta,z_to_ignore)] # Add a column to the A matrix

            ## 2.3] Step 3 : if necessary, hard-threshold to nforce sparsity
            if Theta.shape[0] > K:
                norms = np.linalg.norm(A,axis=0)
                norms[np.where(norms < 1e-15)[0]] = 1e-15 # Avoid /0
                A_norm =  A/norms # normalize, unlike step 4
                A_normri = np.r_[A_norm.real, A_norm.imag] 
                (beta,_) = nnls(A_normri,sketch_ri) # non-negative least squares
                index_to_delete = np.argmin(beta)
                Theta = np.delete(Theta, index_to_delete, axis=0)
                A = np.delete(A, index_to_delete, axis=1)
                if index_to_delete == K:
                    continue # No gain to be expected wrt previous iteration

            ## 2.4] Step 4 : project to find weights
            Ari = np.r_[A.real, A.imag]
            (alpha,_) = nnls(Ari,sketch_ri) # non-negative least squares



            ## 2.5] Step 5
            p0 = _stackTheta(task,Theta,alpha) # Initialize at current solution 
            # Compute the bounds for step 5 : boundsOfOneAtom * numberAtoms then boundsOneWeight * numberAtoms
            boundsThetaAlpha = boundstheta * Theta.shape[0] + [[1e-9,2]] * Theta.shape[0]
            # Solve
            sol = minimize(lambda p: _CLOMPR_step5_fun_grad(task,Phi,p,sketch,z_to_ignore),
                                            x0 = p0, method=opt_method, jac=True,
                                            bounds=boundsThetaAlpha, options={'ftol': ftol}) 
            (Theta,alpha) = _destackTheta(task,sol.x,Phi.d)
            
            # Make covariances feasible
            if task == "gmm-nondiag":
                for k in range(Theta.shape[0]):
                    Theta[k] = _makeGMMfeasible(Theta[k],d)

            # The atoms have changed: we must re-compute A
            A = np.empty([m,0])
            for theta_i in Theta:
                Apthi = _sketchAtom(task,Phi,theta_i,z_to_ignore)
                A = np.c_[A,Apthi]
            # Update residual
            r = sketch - A@alpha

        ## 3) Finalization boundstheta_finetuning
        # Last optimization with the default (fine-grained) tolerance
        if task_finetuning == "gmm-nondiag":
            if verbose > 0: print('finetuning')
            Theta_new = np.zeros((K,d*(d+1))) # Expand to have full cov matrix
            for k in range(K):
                (mu,sig2) = _destackAtom("gmm",Theta[k],d)
                # put current sol on the diagonal of full covariance matrix
                Theta_new[k] = _stackAtom("gmm-nondiag",mu,np.diag(sig2))
            Theta = Theta_new # overwrite
            
            boundsThetaAlpha = boundstheta_finetuning * Theta.shape[0] + [[1e-9,2]] * Theta.shape[0]
            
        if task_finetuning == "gmm-nondiag" or ftol >= 1e-8:
            p0 = _stackTheta(task_finetuning,Theta,alpha)
            
            sol = minimize(lambda p: _CLOMPR_step5_fun_grad(task_finetuning,Phi,p,sketch,z_to_ignore),
                                            x0 = p0, method=opt_method, jac=True,
                                            bounds=boundsThetaAlpha)  # Here ftol is much smaller
            (Theta,alpha) = _destackTheta(task_finetuning,sol.x,Phi.d)    
        
        
        # Normalize alpha
        alpha /= np.sum(alpha)
    
    
        runResidualNorm = np.linalg.norm(sketch - A@alpha)
        if verbose>1: print('Run {}, residual norm is {} (best: {})'.format(iRun,runResidualNorm,bestResidualNorm))
        if runResidualNorm <= bestResidualNorm:
            bestResidualNorm = runResidualNorm
            bestTheta = Theta
            bestalpha = alpha
    
    ## FORMAT OUTPUT
    if task == "kmeans":
        return (bestalpha,bestTheta)
    elif task == "gmm" or task == "gmm-nondiag":
        return _ThetasToGMM(task_finetuning,bestTheta,bestalpha)

    return None




# TODO COMPRESSIVE_LEARNING in rough importance order
# - investigate if auto-differentiation might not be smarter


###################
## 2: Histograms ##
###################

## Utility function: project on the probability simplex
# =====================================================
def project_probabilitySimplex(h):
    """Returns h projected onto {h_i>=0, sum_i h_i = 1}. Algo from https://arxiv.org/abs/1309.1541."""
    d = h.shape[0]
    hsort = np.sort(h)[::-1] # Step 1: reverse sort
    rho = np.sum(hsort + (1-np.cumsum(hsort))/np.arange(1,d+1)>0) # Step 2: nb active components
    lambd = (1/rho)*(1 - hsort[:rho].sum()) # Step 3: lambda parameter
    return np.maximum(h + lambd,np.zeros(d))

## Method 1: using the MMD distance
# =================================

def CL_histogram_MMD(sketch,Phi,domain,dimension,nb_cat_per_dim=None,bins_cont=10):
    """
    Computes a histogram from the fourier sketch by minimizing the approximated MMD between data and histogram.
    
    Parameters
    ----------
    sketch: (m,)-numpy array of complex values, the sketch z of the dataset to learn from
    Phi: SimpleFeatureMap object from pycle.sketching (the map used for sketching), must be a Fourier sketch (complex exp.)
    domain: (d,2)-numpy array, boundaries of the box-like domain (x in R^d is in the box iff box[i,0] <= x_i <= box[i,1])
    dimension: int (0<=axis<d), dimension along which the histogram is constructed

    Additional Parameters
    ---------------------
    nb_cat_per_dim: (d,)-array of ints, the number of categories per dimension for integer data,
                    if its i-th entry = 0 (resp. > 0), dimension i is assumed to be continuous (resp. int.).
                    By default all entries are assumed to be continuous.
    bins_cont: int (>1, default 10), number of bins in the histogram, only used if the target dimension is continuous-valued
                    i.e., if nb_cat_per_dim[dimension] == 0
    
    Returns
    -------
    h: (bins,)-numpy array, the histogram values computed from the sketch
    """
    ## 0) Parsing the inputs
    # Number of categorical inputs
    if nb_cat_per_dim is None:
        nb_cat_per_dim = np.zeros(Phi.d)
    
    is_integer_dimension = False
    if nb_cat_per_dim[dimension] > 0:
        # The data is integer-type
        is_integer_dimension = True
        bins = int(nb_cat_per_dim[dimension])
    else:
        bins = bins_cont

    m = sketch.size
    # 1) Construct the A matrix
    A = 1j*np.zeros((m,bins)) # Pre-allocation
    bin_edges = np.linspace(domain[dimension,0],domain[dimension,1],bins+1)
    box = domain.copy()
    for p in range(bins):
        # move to the next box
        if is_integer_dimension:
            box[dimension,0] = p
            box[dimension,1] = p
        else:
            box[dimension,0] = bin_edges[p]
            box[dimension,1] = bin_edges[p+1]
        A[:,p] = fourierSketchOfBox(box,Phi,nb_cat_per_dim) 
        
    # 1.b) cast to real 
    Ari = np.r_[A.real, A.imag]
    
    # 2) create b vector
    b = np.r_[sketch.real, sketch.imag]
    
    # 3) solve the optimization problem
    def _f_grad(x):
        r = Ari@x-b
        f = 0.5*np.linalg.norm(r)**2
        grad = Ari.T@r
        return (f,grad)
    
    # Starting point
    x0 = np.ones(bins)/bins
    # Linear constraints
    A_constr = np.zeros((bins,bins))
    l_constr = 0*np.ones(bins) # Positive constraints
    A_constr[:bins,:bins] = np.eye(bins)
    upper_bound = 5 # weird that it must be large
    u_constr = upper_bound*np.ones(bins) # Sum-to one constraints
    constr = LinearConstraint(A_constr,l_constr,u_constr)

    # Solve
    sol = minimize(_f_grad, x0, method='trust-constr', bounds=None, constraints=constr, jac=True, options={'verbose': 0})

    return project_probabilitySimplex(sol.x)

## TO UPDATE TO ALLOW INTEGER ENTRIES
def histogramFromSketch_M2M(sketch,Phi,domain,dimension,nb_cat_per_dim=None,bins_cont=10,project_on_probabilitySimplex=True,reg_rho=0.01):
    """
    Computes a histogram from the sketch with the M2M method.
    (Uses the closed-form solution of M2Ms learning stage, with MSE loss).
    
    Parameters
    ----------
    sketch: (m,)-numpy array of complex values, the sketch z of the dataset to learn from
    Phi: SimpleFeatureMap object from pycle.sketching (the map used for sketching), must be a Fourier sketch (complex exp.)
    domain: (d,2)-numpy array, boundaries of the box-like domain (x in R^d is in the box iff box[i,0] <= x_i <= box[i,1])
    dimension: int (0<=axis<d), dimension along which the histogram is constructed

    Additional Parameters
    ---------------------
    nb_cat_per_dim: (d,)-array of ints, the number of categories per dimension for integer data,
                    if its i-th entry = 0 (resp. > 0), dimension i is assumed to be continuous (resp. int.).
                    By default all entries are assumed to be continuous.
    bins_cont: int (>1, default 10), number of bins in the histogram, only used if the target dimension is continuous-valued
                    i.e., if nb_cat_per_dim[dimension] == 0
    reg_rho: real  (>=0, default 0.01): regularization parameter (larger reg_rho entails more regularization)
    
    Returns
    -------
    h: (bins,)-numpy array, the histogram values computed from the sketch
    """

    ## 0) Parsing the inputs

    # Number of categorical inputs
    if nb_cat_per_dim is None:
        nb_cat_per_dim = np.zeros(Phi.d)

    is_integer_dimension = False
    if nb_cat_per_dim[dimension] > 0:
        # The data is integer-type
        is_integer_dimension = True
        bins = int(nb_cat_per_dim[dimension])
    else:
        bins = bins_cont

    # Parse m, d
    if isinstance(Phi,SimpleFeatureMap):
        Omega = Phi.Omega
        d = Phi.d
        m = Phi.m
    else:
        raise ValueError('The Phi argument does not match one of the supported formats.')
    
    ## 1) Construct the A matrix
    # Build a new sketch with all the difference of Omega
    Omega_diffs = np.empty((d,m**2))
    for i in range(m):
        for j in range(m):
            Omega_diffs[:,i*m+j] = Omega[:,i] - Omega[:,j]

    Phi_diffs = SimpleFeatureMap("complexExponential", Omega_diffs,xi=Phi.xi,c_norm=Phi.c_norm)

    # Evaluate the box constraints Fourier transform thanks to this sketch function
    z_diffs_domain = fourierSketchOfBox(domain,Phi_diffs,nb_cat_per_dim)

    # And reshape (not sure if correct)
    A_compl = z_diffs_domain.reshape(m,m)

    # Stack real and imaginary components
    A = np.zeros((2*m,2*m))
    A[:m,:m] = A_compl.real
    A[:m,m:] = A_compl.imag
    A[m:,:m] = -A_compl.imag
    A[m:,m:] = A_compl.real
    
    # Regularize
    A += reg_rho*np.eye(2*m)

    box = domain.copy() # the box in which we do the learning
    bin_edges = np.linspace(domain[dimension,0],domain[dimension,1],bins+1)
    h = np.zeros(bins)
    for p in range(bins):
        # move to the next box
        if is_integer_dimension:
            box[dimension,0] = p
            box[dimension,1] = p
        else:
            box[dimension,0] = bin_edges[p]
            box[dimension,1] = bin_edges[p+1]
        F = fourierSketchOfBox(box,Phi,nb_cat_per_dim)

        # Stack the b vector
        b = np.zeros(2*m)
        b[:m] = F.real
        b[m:] = -F.imag

        
        # ... and solve! 
        a_ri = np.linalg.solve(A, b)
        a = a_ri[:m] + 1j*a_ri[m:]
        

        
        # Predict with the sketch
        #print(a)
        h[p] = np.real(np.dot(a,sketch))
    if project_on_probabilitySimplex:
        h = project_probabilitySimplex(h)
    return h
