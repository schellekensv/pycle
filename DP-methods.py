"""Contains a set of Differentially Private machine learning methods.
Used to compare to our approach"""

import numpy as np

##########################################################
#                  PRIVGENE FOR K-MEANS                  #
##########################################################

def DP_select_kmeans(dataset,candidates,nb_candidates_to_select,epsilon,verbose=0):
    """Selects the best k-means centroids amongst a set of candidates,
    while ensuring epsilon-Differential Privacy with the Exponential Mechanism.
    
    Arguments:
        - dataset:    (n,d)-numpy array containing the n learning examples in dimension d (normalized in -1,+1)
        - candidates: (nb_candidates,K,d)-numpy array containing nb_candidates sets of K centroids in dimension d
        - n_candidates_to_select: int, the number of candidates to select from candidates
        - epsilon:  real (>0), privacy budget
        - verbose:  int, indicates the amount of information printed during execution (from 0 (no info,default) to 2 (full debug mode))
        
    Returns:
        - selected_candidates: (n_candidates_to_select,K,d)-numpy array containing the selected sets of centroids
    """
    
    ## Handling input dimensions
    (n,d) = dataset.shape
    (nb_candidates,K,d) = candidates.shape
    
    # Ensure dataset is normalized
    maxNorm = np.max(np.abs(dataset))
    if maxNorm > 1.:
        print('WARNING: DP_select_kmeans requires the dataset to be normalized to unit L_inf norm.\nThe dataset has been normalized accordingly.')
        dataset = dataset/maxNorm
    
    # Exponential Mechanism parameters
    epsilon_per_selected_candidate = epsilon/nb_candidates_to_select
    damping = 0 # brute-force computing of damping
    for i in range(nb_candidates):
        C = candidates[i]
        minDist = +10000000
        for k in range(K):
            centroid = C[k]
            opposingCorner = -np.sign(centroid)
            dist = np.linalg.norm(centroid-opposingCorner)
            minDist = min(minDist,dist)
        damping = max(damping,minDist)
    if verbose > 1: print('Damping coefficient computed: ', damping)
    
    # Pre-allocation
    selected_candidates = np.empty((nb_candidates_to_select,K,d))
    
    # Selection probabilities
    fitness = np.empty(nb_candidates)
    for i in range(nb_candidates):
        fitness[i] = -epsilon_per_selected_candidate*SSE(dataset,candidates[i])/damping # Needs to import SSE from utils module
    fitness = fitness - fitness.max() # Does not change the outcome while avoiding dividing by ~0 (because of really small probabilities)
    P_select_candidate = np.exp(fitness)
    P_select_candidate /= np.sum(P_select_candidate) # Normalization (sum of probabilities = 1)
    
    remaining_candidates = np.copy(candidates)
    for i in range(nb_candidates_to_select):
        if verbose > 1: print('selecting candidate {}, probs = {}'.format(i,np.sort(P_select_candidate)[-10:]))
        # Select a new candidate
        n_remaining_candidates = remaining_candidates.shape[0]
        new_selected_candidate_index = np.random.choice(n_remaining_candidates, p=P_select_candidate) 
        
        # Add it to the selected set
        selected_candidates[i] = remaining_candidates[new_selected_candidate_index]
        
        # Remove the selected candidate from the remaining candidates
        mask = np.ones(n_remaining_candidates, dtype=bool)
        mask[new_selected_candidate_index] = False
        remaining_candidates = remaining_candidates[mask]
        P_select_candidate   = P_select_candidate[mask]
        P_select_candidate /= np.sum(P_select_candidate) # Normalize
    return selected_candidates


def PrivGene_kmeans(dataset,K,epsilon,nb_candidates = 200,nb_selected_candidates = 10,nb_iterations = None,mutateNoise_SD_init = 0.1,mutateNoise_SD_decay = 0.05,verbose=0):
    """Selects the best k-means centroids amongst a set of candidates,
    while ensuring epsilon-Differential Privacy with the Exponential Mechanism.
    
    Arguments:
        - dataset:    (n,d)-numpy array containing the n learning examples in dimension d (normalized in -1,+1)
        - K: int, the number of centroids to produce
        - epsilon:  real (>0), privacy budget
        - nb_candidates: int, the total number of candidates at each iteration
        - nb_selected_candidates: int, the number of candidates selected based on the dataset
        - mutateNoise_SD_init: real (>0), initial standard deviation of the noise on the centroids when mutating (default 5% of range: 0.05*2)
        - mutateNoise_SD_decay: real(0<x<1), decay multiplicative factor of the noise addition (at each iterate noiseSD *= (1-decay), default 0.05)
        - verbose:  int, indicates the amount of information printed during execution (from 0 (no info,default) to 2 (full debug mode))

    
    Returns:
        - centroids: (K,d)-numpy array containing the selected centroids
    """
    
    # Handling input dimensions
    (n,d) = dataset.shape
    lowb = np.amin(dataset,axis=0) # Should be -np.ones
    uppb = np.amax(dataset,axis=0) # Should be +np.ones
    
    # Set number of iterations and associated epsilon
    if nb_iterations is None:
        # Use the heuristic from the PrivGene paper
        c = 1.25e-3
        nb_iterations = int(c*(n*epsilon)/nb_selected_candidates)
        # Avoid very few iterations
        min_iterations = 2
        nb_iterations = max(nb_iterations,min_iterations)
        if verbose > 0: print('Number of PrivGene iteration has been automatically fixed to: ',nb_iterations)
        
    epsilon_per_iteration = epsilon/nb_iterations
    
    # Random initialization of centroids
    candidates = np.empty((nb_candidates,K,d))
    for ic in range(nb_candidates):
        candidates[ic] = np.random.uniform(lowb,uppb,(K,d))

    # Strength of mutations
    mutateNoise_SD = mutateNoise_SD_init # Initial standard deviation of the noise on the centroids when mutating (5%)
    
    # Main loop
    for i in range(nb_iterations-1):
        if verbose > 0: print('Performing PrivGene iteration: ',i)
        # Selection step
        selected_candidates = DP_select_kmeans(dataset,candidates,nb_selected_candidates,epsilon_per_iteration,verbose)
        
        # Now, crossover and mutate the selected candidates to create the new candidates set (overwritten)
        for j in range(int(nb_candidates/2)):
            # Select the parents
            parents_indexes = np.random.choice(nb_selected_candidates, size=2, replace=False)
            parents = selected_candidates[parents_indexes]
            
            # Crossover step
            Kswap = np.random.choice(K-1) # Number of centroids to swap
            offsprings = parents.copy()
            offsprings[0,:Kswap,:] = parents[1,:Kswap,:]
            offsprings[1,:Kswap,:] = parents[0,:Kswap,:]

            # Mutation step
            for i_offspring in range(2):
                mutate_index = np.random.choice(K)
                offsprings[i_offspring,mutate_index,:] += mutateNoise_SD*np.random.randn(d)
                
            # Keep centroids inside known data bounds (-1,+1) to avoid stupid solutions
            offsprings[np.where(offsprings > 1.)] = 1. 
            offsprings[np.where(offsprings < -1.)] = -1. 
            
            # Add mutated offsprings to new generation of candidates
            for i_offspring in range(2):
                candidates[j*2+i_offspring] = offsprings[i_offspring]
            
        # Update noise strength of the mutations
        mutateNoise_SD *= (1-mutateNoise_SD_decay)
    
    # Last iteration: select the best from the whole population
    centroids = DP_select_kmeans(dataset,candidates,1,epsilon_per_iteration,verbose)[0]
    
    return centroids

##########################################################
#                  DP-EM fot GMM FITTING                 #
##########################################################

## PRIVACY BUDGET ALLOCATION METHODS

def privacyPerIteration_allocate_linearComposition(epsilon,delta,maxiter,K,useLaplace=False):
    """For the DP_EM_forGMM function. Allocates privacy budget for each iteration using linear composition.
    
    Arguments:
        - epsilon:  real (>0), total privacy budget
        - delta:    real (>=0), total privacy tolerance
        - maxiter:  int, the amount of iterations of DP-EM
        - K:        int, the number of Gaussians in the mixture to learn
        - useLaplace: bool (default False), if true Laplace mechanism is used in weigths and center estimations (delta = 0 for those steps)
    Returns:
        - epsilon_i: real (>0), privacy budget epsilon allocated at each iteration for the weigths, one center, or one covariance
        - delta_i:   real (>0), idem, for privacy tolerance delta (if useLaplace is True, corresponds to delta for covariances only)
    """
    epsilon_i = epsilon/(maxiter*(2*K+1))
    
    if useLaplace: # aka. 'LLG'
        delta_i = delta/(maxiter*K)
    else: # aka. 'GGG'
        delta_i = delta/(maxiter*(2*K+1))
    return (epsilon_i,delta_i)

def privacyPerIteration_allocate_advancedComposition(epsilon,delta,maxiter,K,useLaplace=False,delta_i = 1e-6):
    """For the DP_EM_forGMM function. Allocates privacy budget for each iteration using advanced composition.
    
    Arguments:
        - epsilon:  real (>0), total privacy budget
        - delta:    real (>=0), total privacy tolerance
        - maxiter:  int, the amount of iterations of DP-EM
        - K:        int, the number of Gaussians in the mixture to learn
        - useLaplace: bool (default False), if true Laplace mechanism is used in weigths and center estimations (delta = 0 for those steps)
        - delta_i:  real (>=0), total privacy tolerance (default 1e-6)
    Returns:
        - epsilon_i: real (>0), privacy budget epsilon allocated at each iteration for the weigths, one center, or one covariance
        - delta_i:   real (>0), idem, for privacy tolerance delta (if useLaplace is True, corresponds to delta for covariances only)
                                (given for compatibility)
        
    """
    
    if useLaplace: # aka. 'LLG'
        deltaPrime = delta - maxiter*K*delta_i # Assumption: this should be > 0
    else: # aka. 'GGG'
        deltaPrime = delta - maxiter*(2*K+1)*delta_i # Assumption: this should be > 0
        
    # Solve epsilon = maxiter*(2*K+1)*eps_i*(e^(eps_i) - 1) + sqrt(2*maxiter*(2*K+1)*log(1/deltaPrime))*eps_i
    init_eps_i = 0.1
    equation_eps_i = lambda eps_i: maxiter*(2*K+1)*eps_i*(np.exp(eps_i) - 1) + np.sqrt(2*maxiter*(2*K+1)*np.log(1/deltaPrime))*eps_i - epsilon
    epsilon_i = scipy.optimize.fsolve(equation_eps_i, init_eps_i)[0] # import scipy.optimize 


    return (epsilon_i,delta_i)

def privacyPerIteration_allocate_zCDP_Composition(epsilon,delta,maxiter,K,useLaplace=False,delta_i = 1e-6):
    """For the DP_EM_forGMM function. Allocates privacy budget for each iteration using zCDP composition.
    
    Arguments:
        - epsilon:  real (>0), total privacy budget
        - delta:    real (>=0), total privacy tolerance
        - maxiter:  int, the amount of iterations of DP-EM
        - K:        int, the number of Gaussians in the mixture to learn
        - useLaplace: bool (default False), if true Laplace mechanism is used in weigths and center estimations (delta = 0 for those steps)
        - delta_i:  real (>=0), total privacy tolerance (default 1e-6)
    Returns:
        - epsilon_i: real (>0), privacy budget epsilon allocated at each iteration for the weigths, one center, or one covariance
        - delta_i:   real (>0), idem, for privacy tolerance delta (if useLaplace is True, corresponds to delta for covariances only)
                                (given for compatibility)
        
    """
    
    if useLaplace: # aka. 'LLG'
        rho = lambda eps_i: maxiter*(K+1)*(eps_i**2)/2 + maxiter*K*(eps_i**2)/(2*2*np.log(1.25/delta_i))
    else: # aka. 'GGG'
        rho = lambda eps_i: maxiter*(2*K+1)*(eps_i**2)/(2*2*np.log(1.25/delta_i))
        
    # Solve epsilon = rho + 2*sqrt(rho*log(1/delta))
    lowerbound_eps_i = 0
    upperbound_eps_i = 10
    cost_eps_i = lambda eps_i: (rho(eps_i) + 2*np.sqrt(rho(eps_i)*np.log(1/delta)) - epsilon)**2
    epsilon_i = scipy.optimize.fminbound(cost_eps_i, lowerbound_eps_i, upperbound_eps_i) # import scipy.optimize 
        
    return (epsilon_i,delta_i)


# Missing the MA method


## MAIN ALGO

def DP_EM_forGMM(X,K,epsilon,delta,maxiter = 20,budget_allocation='zCDP-GGG',delta_i = 1e-6,weight_smoothing = None,
                 privgeneInit_budget = 0,improveGaussMechanism=True,verbose=0):
    """DP-EM algorithm for Differentially Private fitting of gaussian mixture models.
    First, performs a smart allocation of the privacy budget per iteration.
    Adds Laplacian/Gaussian noise on all learned parameters at each iteration of the usual EM algorithm,
    then post-processes those noisy parameters to ensure the parameters still meet necessary conditions.
    From "DP-EM: Differentially Private Expectation Maximization", Park et al. (2016).
    
    Arguments:
        - X: (n,d)-numpy array, the dataset of n examples in dimension d
        - K: int, the number of Gaussian modes
        - epsilon:  real (>0), privacy budget
        - max_iter: int, the number of EM iterations to perform (default 20)
        - weight_smoothing: real, the weight smoothing term delta from the method (default = 0.05)
        - initialization: string
        - verbose:  int, indicates the amount of information printed during execution (from 0 (no info,default) to 2 (full debug mode))

        
    Returns: a tuple (w,mus,Sigmas) of three numpy arrays
        - w:      (K,)   -numpy array containing the weigths ('mixing coefficients') of the Gaussians
        - mus:    (K,d)  -numpy array containing the means of the Gaussians
        - Sigmas: (K,d,d)-numpy array containing the covariance matrices of the Gaussians
    """
    # TODO: all-Laplace mechanism to have thing with delta = 0?

    # Parse input    
    (n,d) = X.shape
    if weight_smoothing is None:
        weight_smoothing = 1/n
    
    # Ensure dataset is normalized
    maxNorm = np.linalg.norm(X,axis=1).max()
    if maxNorm > 1.:
        print('WARNING: DP_EM requires the dataset to be normalized to unit L_2 norm.\nThe dataset has been normalized accordingly.')
        X = X/maxNorm
        
    # Initializations
    w = np.ones(K)/K
    mus = np.empty((K,d)) # Will be filled in a moment
    Sigmas = np.empty((K,d,d)) # idem
    r = np.empty((n,K)) # Matrix of posterior probabilities, here memory allocation only
    # Initialize the centers
    if privgeneInit_budget > 0:
        # Re-allocate the privacy budget to account for PrivGene initialization
        epsilon_privgene = epsilon*privgeneInit_budget
        epsilon = (1-privgeneInit_budget)*epsilon
        
        centroids_privgene = PrivGene_kmeans(X,K,epsilon_privgene,verbose=verbose)
        for k in range(K):
            mus[k] = centroids_privgene[k]
    else: # Uniform initialization
        for k in range(K):
            # Centers: brutal generation of uniform points in the sphere: generate in the square then discard if norm too large
            normMuk = 1.1
            while (normMuk > 1.):
                mus[k] = np.random.uniform(-np.ones(d),+np.ones(d))
                normMuk = np.linalg.norm(mus[k]) 
     # Initialize the covariances, TODO change heuristic??
    for k in range(K):
        Sigmas[k] = np.diag(np.abs(0.01*np.random.randn(d))) # Covariances are initialized as random diagonal covariances, with folded Gaussian values

    # Allocate privacy budget according to chosen method (with some robustness :p)
    # First, check if we use Laplace
    budget_allocation = budget_allocation.upper()
    if 'LLG' in budget_allocation:
        useLaplace = True
    elif 'GGG' in budget_allocation:
        useLaplace = False
    else:
        raise Exception('Unreckognized privacy allocation method ({}). Aborting.'.format(x))
        return
    
    # Then, call the relevant budget splitting method
    if 'LINEAR' in budget_allocation:
        (eps_i,delta_i) = privacyPerIteration_allocate_linearComposition(epsilon,delta,maxiter,K,useLaplace)
    elif 'ADVANCED' in budget_allocation:
        (eps_i,delta_i) = privacyPerIteration_allocate_advancedComposition(epsilon,delta,maxiter,K,useLaplace,delta_i)
    elif 'ZCDP' in budget_allocation:
        (eps_i,delta_i) = privacyPerIteration_allocate_zCDP_Composition(epsilon,delta,maxiter,K,useLaplace,delta_i)
    else:
        raise Exception('Unreckognized privacy allocation method ({}). Aborting.'.format(x))
        return
    
    
    if verbose > 0: print('Per-iteration budget: epsilon = {}, delta = {}'.format(eps_i,delta_i))
    if eps_i >= 1 and not(improveGaussMechanism): print('WARNING: with epsilon >= 1 the sigma bound for Gaussian mechanism doesn\'t hold! Privacy is NOT ensured!')


    # Main loop
    for i in range(maxiter):
        if verbose > 1: print('DP-EM iteration:',i)
        if verbose > 1: plotGMM(X,(w,mus,Sigmas))
        # E step
        for k in range(K):
            r[:,k] = w[k]*multivariate_normal.pdf(X, mean=mus[k], cov=Sigmas[k],allow_singular=True)
        # Try smoothing
        r += (1/(100*n**2)) # Necessary to avoid crashes with divide by 0: smooth by adding very small number
        r = (r.T/np.sum(r,axis=1)).T # Normalize (the posterior probabilities sum to 1). Dirty :-(

        if verbose > 1: print(r)
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
            if verbose > 1: print('try:',np.sum(r[:,k]),num)
            Sigmas[k] = num/np.sum(r[:,k])
        if verbose > 1: print(Sigmas[k])
        # (end of one usual EM iteration)

        # Noise adding step on weigths
        S_w      = 2./n # Same L_1 and L_2 (I guess)
        if useLaplace: 
            w   += np.random.laplace(scale=(S_w/eps_i), size=K)
        else:
            if improveGaussMechanism: # TODO move out of loop
                noiseSD_w = agm.calibrateAnalyticGaussianMechanism(eps_i, delta_i, S_w)
            else:
                noiseSD_w = np.sqrt(2*np.log(1.25/delta_i))*(S_w/eps_i)
            w   += noiseSD_w*np.random.randn(K)
        
        
        #print(np.sqrt(2*np.log(1.25/delta_i))*(S_w/eps_i))
        if verbose > 1: print('noise scale:',np.sqrt(2*np.log(1.25/delta_i))*(S_w/eps_i))
            
        # Post-processing step of w
        w[np.where(w<0)] = 0. + weight_smoothing # Smoothing (not sure if I should keep this, should avoid problems with empty gaussians)
        w[np.where(w>1)] = 1.
        w /= np.sum(w) # Normalize
                
        if verbose > 1: plotGMM(X,(w,mus,Sigmas))
        
        # Noise adding step on mean and covariances
        for k in range(K):
            # Compute the sensitivity based on the actual (noisy) class size
            Nk = w[k]*n # Estimate the sample size of Gaussian k from w for DP estimation of the sensitivities
            
            # Dirty trick (?)
            Nk = max(Nk,1)
            
            
            S_muk    = 2*np.sqrt(d)/Nk if useLaplace else 2/Nk 
            S_Sigmak = 2/Nk 
            
            
            if verbose > 1: print('Nk --> ',Nk)
            
            # Noise adding on the means
            if useLaplace: 
                mus[k] += np.random.laplace(scale=(S_muk/eps_i), size=d)
            else:
                if improveGaussMechanism: # TODO move out of loop
                    noiseSD_muk = agm.calibrateAnalyticGaussianMechanism(eps_i, delta_i, S_muk)
                else:
                    noiseSD_muk = np.sqrt(2*np.log(1.25/delta_i))*(S_muk/eps_i)
                mus[k]   += noiseSD_muk*np.random.randn(d)
            
            
                       
            # Bonus improvement: project inside the known bounds (||x||_2 <= 1), avoid nonsense solutions
            mukNorm = np.linalg.norm(mus[k])
            if mukNorm > 1.: mus[k] /= mukNorm
            
            # Noise adding on the covariances
            if improveGaussMechanism: # TODO move out of loop
                noiseSD_Sigmak = agm.calibrateAnalyticGaussianMechanism(eps_i, delta_i, S_Sigmak)
            else:
                noiseSD_Sigmak = np.sqrt(2*np.log(1.25/delta_i))*(S_Sigmak/eps_i)
            
            noiseSigmak = noiseSD_Sigmak*np.random.randn(d,d)
            
            iu = np.triu_indices(d,1) # Some black magic to have symmetric noise matrix
            il = (iu[1],iu[0])
            noiseSigmak[il]=noiseSigmak[iu]
            Sigmas[k] += noiseSigmak
            
            if verbose > 1: print(noiseSigmak)
            if verbose > 1: print(Sigmas[k])

            # Post-processing step of Sigmas
            (eigs,eigvecs) = np.linalg.eigh(Sigmas[k]) # Compute eigenvalues
            if verbose > 1: print('eigs:',eigs)
            # NEW
            eigs[np.where(eigs<=0)] = 1e-5
            # TODO: Bonus improvement: project inside logical bounds (recall ||x||_2 <= 1) avoid nonsense solutions
            maxEig = 50
            eigs[np.where(eigs>maxEig)] = maxEig
            
            if verbose > 1: print('new eigs:',eigs)
            
            Sigmas[k] = eigvecs@np.diag(eigs)@eigvecs.T
            # OLD
            #deltaSigmak = max(-eigs.min(),0)     # Min constant to add
            #Sigmas[k] += deltaSigmak*np.eye(d)
            
            
            if verbose > 1: plotGMM(X,(w,mus,Sigmas))
    if verbose > 1: plotGMM(X,(w,mus,Sigmas))
        
    return (w,mus,Sigmas)


##########################################################
#                  DP-GMM fot GMM FITTING                #
##########################################################


# TODO: check conditions are met? Enforce them?
def DP_GMM(X,K,epsilon,max_iter = 20,weight_smoothing = 0.05, initialization = 'privgene', verbose = 0):
    """DP-GMM algorithm for Differentially Private fitting of gaussian mixture models.
    Adds Laplacian noise on all learned parameters at each iteration of the usual EM algorithm,
    then post-processes those noisy parameters to ensure the parameters still meet necessary conditions.
    From "Differentially private density estimation via Gaussian mixtures model", Wu et al. (2016).
    
    Arguments:
        - X: (n,d)-numpy array, the dataset of n examples in dimension d
        - K: int, the number of Gaussian modes
        - epsilon:  real (>0), privacy budget
        - max_iter: int, the number of EM iterations to perform (default 20)
        - weight_smoothing: real, the weight smoothing term delta from the method (default = 0.05)
        - initialization: string
        - verbose:  int, indicates the amount of information printed during execution (from 0 (no info,default) to 2 (full debug mode))

        
    Returns: a tuple (w,mus,Sigmas) of three numpy arrays
        - w:      (K,)   -numpy array containing the weigths ('mixing coefficients') of the Gaussians
        - mus:    (K,d)  -numpy array containing the means of the Gaussians
        - Sigmas: (K,d,d)-numpy array containing the covariance matrices of the Gaussians
    """
    
    # Parse input    
    (n,d) = X.shape
    lowb = np.amin(X,axis=0)
    uppb = np.amax(X,axis=0)
    R = np.linalg.norm(X,ord=1,axis=1).max() # TODO : beaks DP?
    
    # Allocate privacy budget (magic numbers from the paper)
    eps_w      = 0.04*epsilon/max_iter
    eps_mus    = 0.16*epsilon/(max_iter*K)
    eps_Sigmas = 0.70*epsilon/(max_iter*K)
    # Compensate if we don't use a DP initialization
    if initialization is 'uniform':     
        eps_w      *= 10/9 
        eps_mus    *= 10/9
        eps_Sigmas *= 10/9
    
    
    # Initializations
    w = np.ones(K)/K
    mus = np.empty((K,d))
    Sigmas = np.empty((K,d,d)) # Covariances are initialized as random diagonal covariances, with folded Gaussian values

    # TODO CLEAN THIS MESS
    if initialization in ['PrivGene','privgene','Privgene']:
        centroids_privgene = PrivGene_kmeans(X,K,0.1*epsilon,verbose=verbose)
        r = np.empty((n,K)) # Matrix of posterior probabilities, here memory allocation only
        for k in range(K):
            mus[k] = centroids_privgene[k]
            Sigmas[k] = np.diag(np.abs(0.01*np.random.randn(d))) # TODO CHANGE MAGIC NUMBER
            r[:,k] = w[k]*multivariate_normal.pdf(X, mean=mus[k], cov=Sigmas[k],allow_singular=True)
        r = (r.T/np.sum(r,axis=1)).T # Normalize (the posterior probabilities sum to 1). Dirty :-(
    elif initialization is 'uniform':
        r = np.empty((n,K)) # Matrix of posterior probabilities, here memory allocation only
        for k in range(K):
            mus[k] = np.random.uniform(lowb,uppb)
            Sigmas[k] = np.diag(np.abs(0.01*np.random.randn(d)))
            r[:,k] = w[k]*multivariate_normal.pdf(X, mean=mus[k], cov=Sigmas[k],allow_singular=True)
        r = (r.T/np.sum(r,axis=1)).T # Normalize (the posterior probabilities sum to 1). Dirty :-(
        # Check if condition is met:

        if np.any(np.sum(r,axis=0)/n < 1/(2*K)):
            print("WARNING: condition on weights for DP-GMM algorithm is not met. DP-GMM will fail.")
        else:
            print('condition on weights for DP-GMM algorithm OK.')

    else:
        raise Exception('Unreckognized initialization method ({}). Aborting.'.format(initialization))
        
    # Compute sensitivities
    S_w      = K/n
    S_mus    = 4*R*K/n                       # Valid only if ||x_i||_1 <= R, sum_i(r_ik) >= n/(2*K)
    S_Sigmas = (12*n*K*R*R+8*K*K*R*R)/(n**2) # Valid only if ||x_i||_1 <= R, sum_i(r_ik) >= n/(2*K)

    # Main loop
    for i in range(max_iter):
        if verbose > 0: print('DP-GMM iteration: ',i)
        if verbose > 1: plotGMM(X,(w,mus,Sigmas))
        # E step
        for k in range(K):
            #print(np.linalg.det(Sigmas[k]))
            #print(Sigmas[k])
            r[:,k] = w[k]*multivariate_normal.pdf(X, mean=mus[k], cov=Sigmas[k],allow_singular=True)
        r = (r.T/np.sum(r,axis=1)).T # Normalize (the posterior probabilities sum to 1). Dirty :-(
        
        if np.any(np.sum(r,axis=0)/n < 1/(2*K)):
            print("WARNING: condition on weights for DP-GMM algorithm is not met. Privacy guarantee broke.")
        else:
            print('condition on weights for DP-GMM algorithm OK.')

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
            if np.sum(r[:,k]) > 1/(n*1000):
                Sigmas[k] = num/np.sum(r[:,k])
            else: # class is completely empty, restart
                Sigmas[k] = np.diag(np.abs(0.01*np.random.randn(d)))
            # Dirty trick to avoid singularities
            #Sigmas[k] += (1e-10)*np.eye(d)

        # (end of one usual EM iteration)
        
        # Noise adding step
        w   += np.random.laplace(scale=(S_w/eps_w), size=K)
        mus += np.random.laplace(scale=(S_mus/eps_mus), size=(K,d))
        for k in range(K):
            noiseSigmak = np.random.laplace(scale=(S_Sigmas/eps_Sigmas), size=(d,d))
            iu = np.triu_indices(d,1) # Some black magic to have symmetric noise matrix
            il = (iu[1],iu[0])
            noiseSigmak[il]=noiseSigmak[iu]
            Sigmas[k] += noiseSigmak
        
        
        # Post-processing step: 1) post-process w
        w = (w - w.min())/(w.max()-w.min()) # Map to [0,1]
        w += weight_smoothing # Smoothing
        w /= np.sum(w) # Normalize
        
        # Post-processing step: 2) post-process Sigmas
        for k in range(K):
            (eigs,_) = np.linalg.eig(Sigmas[k]) # Compute eigenvalues
            #deltaSigmak = max(-eigs.min()+,0)     # Min constant to add
            deltaSigmak = max(-eigs.min()+1e-3,0)     # Min constant to add
            Sigmas[k] += deltaSigmak*np.eye(d)
            # Dirty trick to avoid singularities
            #Sigmas[k] += (1e-10)*np.eye(d)
            #print(np.linalg.eig(Sigmas[k]))
            #print(np.linalg.det(Sigmas[k]))
    if verbose > 0: plotGMM(X,(w,mus,Sigmas))
        
    return (w,mus,Sigmas)