# General imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls, minimize
import scipy

def CLOMPR_GMM(empiricalSketch,K,fourierSketchingMatrix,dithering = None,bounds = None,nIterations = None,bestOfRuns=1,sketchNormalConstant=None,verbose=0):
    """Learns a Gaussian Mixture Model (GMM) from the complex exponential sketch of a dataset ("compressively").
    The sketch given in argument is asumed to be of the following form:
        z = (1/N) * sum_{i = 1}^N exp(j*[Omega*x_i + xi]),
    and the Gaussian Mixture to estimate will have a density given by (N is the usual Gaussian density):
        P(x) = sum_{k=1}^K alpha_k * N(x;mu_k,Sigma_k)       s.t.      sum_k alpha_k = 1.
    
    Arguments:
        - empiricalSketch: (m,)-numpy array of complex reals, the sketch z of the dataset to learn from
        - K: int, the number of Gaussians in the mixture to estimate
        - fourierSketchingMatrix: (n,m)-numpy array of reals, the Omega
        
        
    Returns: a tuple (X,weigths,means,covariances) of four numpy arrays
        - TODO
    """
    # TODO support for nondiagonal covariances
    
    ## 0) Handle input
    z = empiricalSketch
    m = z.shape[0]
    d = dimension
    Omega = fourierSketchingMatrix
    xi = dithering
    
    if sketchNormalConstant is None:
        scst = 1.
    
    def sketchOfGaussian(mu,Sigma,Omega):
        '''returns a m-dimensional complex vector'''
        return scst*np.exp(1j*(mu@Omega) -np.einsum('ij,ij->i', np.dot(Omega.T, Sigma), Omega.T)/2.)*np.exp(1j*xi) # black voodoo magic to evaluate om_j^T*Sig*om_j forall j

    def gradMuSketchOfGaussian(mu,Sigma,Omega):
        '''returns a d-by-m-dimensional complex vector'''
        return scst*1j*Omega*sketchOfGaussian(mu,Sigma,Omega)
    
    def gradSigmaSketchOfGaussian(mu,Sigma,Omega):
        '''returns a d-by-m-dimensional complex vector'''
        return -scst*0.5*(Omega**2)*sketchOfGaussian(mu,Sigma,Omega)

        
    def stacktheta(mu,sigma):
        '''Stacks all the elements of one atom (mean and diagonal variance of a Gaussian) into one atom vector'''
        return np.append(mu,sigma)

    def destacktheta(th):
        mu = th[:d]
        sigma = th[-d:]
        return (mu,sigma)
    
    def Apth(th): # computes sketh from one atom
        mu,sig = destacktheta(th)
        return sketchOfGaussian(mu,np.diag(sig),Omega)

    def stackTheta(Theta,alpha):
        '''Stacks all the atoms and their weights into one vector'''
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
    
    def step1funGrad(th,r): # OK
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
    
    def step5funGrad(p,z): # TODO
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
    
    
    
    if nIterations is None:
        nIterations = 2*K
    
    # Bounds for one Gaussian center
    if bounds is None:
        lowb = -np.ones(d)
        uppb = +np.ones(d)
    else:
        (lowb,uppb) = bounds
    # Format the bounds for the optimization solver
    boundstheta = np.array([lowb,uppb]).T.tolist()  # bounds for the means
    varianceLowerBound = 1e-8
    for i in range(d): boundstheta.append([varianceLowerBound,None]) # bounds for the variance
        
    
    bestResidualNorm = np.inf #np.linalg.norm(z) # Residual of 'trivial solution'
    bestTheta = None
    bestalpha = None
    for iRun in range(bestOfRuns):
    
        ## 1) Initialization
        r = empiricalSketch  # residual
        Theta = np.empty([0,2*d]) # Theta is a nbAtoms-by-atomDimension (= 2*d) array

        ## 2) Main optimization
        for i in range(nIterations):
            ## 2.1] Step 1 : find new atom theta most correlated with residual
            # Initialize the new atom
            mu0 = np.random.uniform(lowb,uppb) # initial mean at random
            sig0 = np.ones(d) # initial covariance matrix is identity
            x0 = stacktheta(mu0,sig0)
            # And solve
            
            #print(step1funGrad(x0,z))
            
            sol = scipy.optimize.minimize(lambda th: step1funGrad(th,r), x0 = x0, args=(), method='L-BFGS-B', jac=True,
                                            bounds=boundstheta, constraints=(), tol=None, options=None) # change constrains
            
            theta = sol.x
            # TODO : fix bounds, constraints, x0, check args, opts, tol
            #print(theta)
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
                b = empiricalSketch
                bri = np.r_[b.real, b.imag] # todo : outside the loop
                (beta,_) = nnls(Ari,bri) # non-negative least squares
                #print('---')
                #print(beta)
                #print(Theta)
                Theta = np.delete(Theta, (np.argmin(beta)), axis=0)
                #print(Theta)
                #Theta = Theta[]
            ## 2.4] Step 4 : project to find weights

            # Construct A = sketchFeatureFunNorm(Theta); TODO avoid rebuilding everything
            # TODO : avoid doing this if we did the computing at step 3?
            A = np.empty([m,0])
            for theta_i in Theta:
                Apthi = Apth(theta_i)
                A = np.c_[A,Apthi]
            Ari = np.r_[A.real, A.imag]
            b = empiricalSketch
            bri = np.r_[b.real, b.imag] # todo : outside the loop
            (alpha,res) = nnls(Ari,bri) # non-negative least squares

            ## 2.5] Step 5
            #print('___')
            #print(Theta)
            # Initialize at current solution
            x0 = stackTheta(Theta,alpha)
            # Compute the bounds for step 5 : boundsOfOneAtom * numberAtoms then boundsOneWeight * numberAtoms
            boundsThetaAlpha = boundstheta * Theta.shape[0] + [[0,None]] * Theta.shape[0]
            # Solve
            sol = scipy.optimize.minimize(lambda p: step5funGrad(p,z), x0 = x0, args=(), method='L-BFGS-B', jac=True,
                                            bounds=boundsThetaAlpha, constraints=(), tol=None, options=None) # TODO ADD BOUNDS change constrains
            (Theta,alpha) = destackTheta(sol.x)
            #print(Theta)

            A = np.empty([m,0])
            for theta_i in Theta:
                Apthi = Apth(theta_i)
                A = np.c_[A,Apthi]
            r = z - A@alpha

        ## 3) Finalization
        # Normalize alpha
        alpha /= np.sum(alpha)
    
    
        runResidualNorm = np.linalg.norm(z - A@alpha)
        #print('Run {}, residual norm is {} (best: {})'.format(iRun,runResidualNorm,bestResidualNorm))
        if runResidualNorm < bestResidualNorm:
            bestResidualNorm = runResidualNorm
            bestTheta = Theta
            bestalpha = alpha
    
    return (bestTheta,bestalpha)



def CKM(empiricalSketch,sketchFeatureFun,sketchFeatureGrad,dimension,K,bounds = None,nIterations = None,normalized=True):
    # ? task = 'K-Means','GMM',... should also include K somewhere
    # 
    # should contain methods to compute the gradients for the different steps?
    # Or hard-code everything inside this funciton?
    # ??? use automatic differentiation ? I should try to see if it is as fast as hard-coded one.
    
    ## 0) Handle input
    z = empiricalSketch
    m = z.shape[0]
    d = dimension
    if normalized: # If the sketch of an atom has always the same norm (e.g. in (Q)CKM)
        sketchFeatureFunNorm  = sketchFeatureFun
        sketchFeatureGradNorm = sketchFeatureGrad
    else:
        sketchFeatureFunNorm = None # TODO
        sketchFeatureFunNorm = None # TODO
        

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
    
    
    
    if nIterations is None:
        nIterations = 2*K
        
    if bounds is None:
        lowb = -np.ones(d)
        uppb = +np.ones(d)
    else:
        (lowb,uppb) = bounds
    
    ## 1) Initialization
    r = empiricalSketch  # residual
    Theta = np.empty([0,d]) # Theta is a nbAtoms-by-atomDimension array
    
    ## 2) Main optimization
    for i in range(nIterations):
        ## 2.1] Step 1 : find new atom theta most correlated with residual
        x0 = np.random.uniform(lowb,uppb)
        bounds = np.array([lowb,uppb]).T.tolist()
        sol = scipy.optimize.minimize(lambda th: step1funGrad(th,r), x0 = x0, args=(), method='L-BFGS-B', jac=True,
                                        bounds=bounds, constraints=(), tol=None, options=None) # change constrains
        theta = sol.x
        # TODO : fix bounds, constraints, x0, check args, opts, tol
        #print(theta)
        ## 2.2] Step 2 : add it to the support
        Theta = np.append(Theta,[theta],axis=0)
        ## 2.3] Step 3 : if necessary, hard-threshold to nforce sparsity
        if Theta.shape[0] > K:
            # Construct A = sketchFeatureFunNorm(Theta); TODO avoid rebuilding everything
            A = np.empty([m,0])
            for theta_i in Theta:
                A = np.c_[A,sketchFeatureFunNorm(theta_i)]
            Ari = np.r_[A.real, A.imag]
            b = empiricalSketch
            bri = np.r_[b.real, b.imag] # todo : outside the loop
            (beta,_) = nnls(Ari,bri) # non-negative least squares
            #print('---')
            #print(beta)
            #print(Theta)
            Theta = np.delete(Theta, (np.argmin(beta)), axis=0)
            #print(Theta)
            #Theta = Theta[]
        ## 2.4] Step 4 : project to find weights
        
        # Construct A = sketchFeatureFunNorm(Theta); TODO avoid rebuilding everything
        # TODO : avoid doing this if we did the computing at step 3?
        A = np.empty([m,0])
        for theta_i in Theta:
            A = np.c_[A,sketchFeatureFun(theta_i)]
        Ari = np.r_[A.real, A.imag]
        b = empiricalSketch
        bri = np.r_[b.real, b.imag] # todo : outside the loop
        (alpha,res) = nnls(Ari,bri) # non-negative least squares
        
        ## 2.5] Step 5
        #print('___')
        #print(Theta)
        x0 = stackTheta(Theta,alpha)
        sol = scipy.optimize.minimize(lambda p: step5funGrad(p,z), x0 = x0, args=(), method='L-BFGS-B', jac=True,
                                        bounds=None, constraints=(), tol=None, options=None) # TODO ADD BOUNDS change constrains
        (Theta,alpha) = destackTheta(sol.x)
        #print(Theta)
        
        A = np.empty([m,0])
        for theta_i in Theta:
            A = np.c_[A,sketchFeatureFun(theta_i)]
        r = z - A@alpha
    
    ## 3) Finalization
    # Normalize alpha
    alpha /= np.sum(alpha)
    return (Theta,alpha)