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


"""Contains compressive learning algorithms."""

# Main imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls, minimize, LinearConstraint
from copy import copy

# For debug
import time

# We rely on the sketching functions
from .sketching import FeatureMap, SimpleFeatureMap, fourierSketchOfBox, fourierSketchOfGaussian, estimate_Sigma_from_sketch



import numpy as np
import scipy.optimize
from pycle.sketching import FeatureMap


##################################################
# 0: abstract template for CL and CL-OMP solvers #
##################################################

## 0.1 Generic solver (stores a sketch and a solution, can run multiple trials of a learning method to specify)
class Solver:
    """
    Template for a compressive learning solver, used to solve the problem
        min_(theta) || sketch_weight * z - A_Phi(P_theta) ||_2.
    Implements several trials of an abstract method and keeps the best one.
    """
    def __init__(self,Phi,sketch=None,sketch_weight = 1.,verbose=0):
        """
        - Phi: a FeatureMap object
        - sketch_weight: float, a re-scaling factor for the data sketch
        """
        
        # Encode feature map
        assert isinstance(Phi,FeatureMap)
        self.Phi = Phi
        
        # Encode sketch and sketch weight
        self.update_sketch_and_weight(sketch,sketch_weight)
        
        # Encode current theta and cost value
        self.update_current_sol_and_cost(None)
        
        # Verbose
        self.verbose = verbose
        
        
    # Abtract methods
    # ===============
    # Methods that have to be instantiated by child classes
    def sketch_of_solution(self,sol=None):
        """
        Should return the sketch of the given solution, A_Phi(P_theta).
        
        In: a solution P_theta (the exact encoding is determined by child classes), if None use the current sol 
        Out: sketch_of_solution: (m,)-array containing the sketch
        """
        raise NotImplementedError
        
    def fit_once(self,random_restart=False):
        """Optimizes the cost to the given sketch, by starting at the current solution"""
        raise NotImplementedError
    
    # Generic methods
    # ===============
    # They should always work, using the instances of the methdos above
    def fit_several_times(self,n_repetitions=1,forget_current_sol=False):
        """Solves the problem n times. If a sketch is given, updates it."""

        # Initialization
        if forget_current_sol: 
            # start from scratch
            best_sol, best_sol_cost = None, np.inf 
        else:
            # keep current solution as candidate
            best_sol, best_sol_cost = self.current_sol, self.current_sol_cost
        
        # Main loop, perform independent trials
        for i_repetition in range(n_repetitions):
            self.fit_once(random_restart=True)
            self.update_current_sol_and_cost()
                        
            if self.current_sol_cost < best_sol_cost:
                best_sol, best_sol_cost = self.current_sol, self.current_sol_cost
                
        # Set the current sol to the best one we found
        self.current_sol, self.current_sol_cost = best_sol, best_sol_cost
        return self.current_sol
    
    def update_sketch_and_weight(self,sketch=None,sketch_weight=None):
        """Updates the residual and cost to the current solution. If sol given, also updates it."""
        if sketch is not None:
            self.sketch = sketch
        if sketch_weight is not None:
            assert isinstance(sketch_weight,float) or isinstance(sketch_weight,int)
            self.sketch_weight = sketch_weight
        self.sketch_reweighted = self.sketch_weight*self.sketch
        
    def update_current_sol_and_cost(self,sol=None):
        """Updates the residual and cost to the current solution. If sol given, also updates it."""

        # Update current sol if argument given
        if sol is not None:
            self.current_sol = sol
        
        # Update residual and cost
        try:
            self.residual = self.sketch_reweighted - self.sketch_of_solution(self.current_sol)
            self.current_sol_cost = np.linalg.norm(self.residual)
        except AttributeError: # We are here if self.current_sol does not exist yet
            self.current_sol, self.residual = None, self.sketch_reweighted
            self.current_sol_cost = np.inf
        

        
## 0.2 CL-OMP template (stores a *mixture model* and implements a generic OMP for it)
class CLOMP(Solver):
    """
    Template for Compressive Learning with Orthogonal Matching Pursuit (CL-OMP) solver,
    used to the CL problem
        min_(theta) || sketch_weight * z - A_Phi(P_theta) ||_2,
    where P_theta = sum_{k=1}^K is a weighted mixture composed of K components P_theta_k,
    hence the problem to solve becomes
        min_(alpha,theta_k) || sketch_weight * z - sum_k alpha_k*A_Phi(P_theta_k) ||_2.
    The CLOMP algorithm works by adding new elements to the mixture one by one.
    """
    def __init__(self,Phi,K,d_atom,bounds,sketch=None,sketch_weight = 1.,verbose=0):
        """
        - Phi: a FeatureMap object
        - K: int, target number of mixture components
        - d_atom: dimension of an atom, should be determined by a child class
        - sketch: the sketch to be fit (can be None)
        - sketch_weight: float, a re-scaling factor for the data sketch (default 1)
        """
        # Call parent class
        super(CLOMP, self).__init__(Phi,sketch,sketch_weight,verbose)
        
        # Set other values
        self.K = K
        self.n_atoms = 0
        self.d_atom  = d_atom
        
        # Initialize empty solution
        self.initialize_empty_solution()
        
        # Set bounds
        self.set_bounds_atom(bounds) # bounds for an atom
        
        # Other minor params
        self.minimum_atom_norm = 1e-15*np.sqrt(self.d_atom)
        self.weight_lower_bound = 1e-9
        self.weight_upper_bound = 2
        self.step5_ftol = 1e-6
        
        
    # Abtract methods
    # ===============
    # Methods that have to be instantiated by child classes
    
    # Sketch of a single atom
    def sketch_of_atom(self,theta_k,return_jacobian=False):
        """
        Computes and returns A_Phi(P_theta_k) for an atom P_theta_k.
        possibly with the jacobian, of size (d_atom,m)
        """
        assert theta_k.size == self.d_atom
        raise NotImplementedError
        if return_jacobian:
            return sketch_of_atom, jacobian
        else:
            return sketch_of_atom
        
    def set_bounds_atom(self,bounds):
        """
        Should set self.bounds_atom to a list of length d_atom of lower and upper bounds, i.e.,
            self.bounds_atom = [[lowerbound_1,upperbound_1], ..., [lowerbound_d_atom,upperbound_d_atom]]
        """
        self.bounds = bounds # data bounds
        raise NotImplementedError
        self.bounds_atom = None
        return None
        
    def randomly_initialize_new_atom(self):
        raise NotImplementedError
        return new_theta
    
    # Generic methods
    # ===============
    # They should always work, using the instances of the methdos above
    def initialize_empty_solution(self):
        self.n_atoms = 0
        self.alpha = np.empty(0)                   # (n_atoms,)-array, weigths of the mixture elements
        self.Theta = np.empty((0,self.d_atom))     # (n_atoms,d_atom)-array, all the found parameters in matrix form
        self.Atoms = np.empty((self.Phi.m,0))      # (m,n_atoms)-array, the sketch of the found parameters (m is sketch size)
        self.Jacobians = np.empty((0,self.d_atom,self.Phi.m))  # (n_atoms,d_atom,m)-array, the jacobians of the residual wrt each atom
        self.current_sol = (self.alpha,self.Theta) # Overwrite
    
    def compute_Atoms_matrix(self,Theta=None,return_jacobian=False):
        """
        Computes the matrix of atoms from scratch (if no Theta given, uses current Theta)
        """
        if Theta is not None:
            _n_atoms, _Theta = Theta.shape[0], Theta
        else:
            _n_atoms, _Theta = self.n_atoms, self.Theta
        _A = 1j*np.empty((self.Phi.m,_n_atoms))
        
        if return_jacobian:
            _jac = 1j*np.empty((_n_atoms,self.d_atom,self.Phi.m))
            for k,theta_k in enumerate(_Theta):
                _A[:,k], _jac[k,:,:] = self.sketch_of_atom(theta_k,return_jacobian=True)
            return _A, _jac
        else:
            for k,theta_k in enumerate(_Theta):
                _A[:,k] = self.sketch_of_atom(theta_k,return_jacobian=False)
            return _A
        
        
    def update_Atoms(self,Theta=None,update_jacobian=False):
        """
        Update the Atoms matrix (a (n_atoms,m)-array containing the A_Phi(P_theta_k) vectors)
        - with current Theta (self.Theta) if no argument is provided
        - with the provided Theta argument if one is given (self.Theta will also be updated)
        """
        if Theta is not None:
            self.Theta = Theta # If necessary, update Theta
        if update_jacobian:
            self.Atoms, self.Jacobians = self.compute_Atoms_matrix(return_jacobian=True)
        else:
            self.Atoms = self.compute_Atoms_matrix(return_jacobian=False)
   
    # Add/remove atoms
    def add_atom(self,new_theta):
        self.n_atoms += 1
        self.Theta = np.append(self.Theta,[new_theta],axis=0) #np.r_[self.Theta,new_theta]
        self.Atoms = np.c_[self.Atoms,self.sketch_of_atom(new_theta)]
        
    def remove_atom(self,index_to_remove):
        self.n_atoms -= 1
        self.Theta = np.delete(self.Theta,index_to_remove,axis=0)
        self.Atoms = np.delete(self.Atoms,index_to_remove,axis=1)
        
    def replace_atom(self,index_to_replace,new_theta):
        self.Theta[index_to_replace] = new_theta
        self.Atoms[:,index_to_replace] = self.sketch_of_atom(new_theta)
    
    # Stack/de-stack the found atoms
    def _stack_sol(self,alpha=None,Theta=None):
        '''Stacks *all* the atoms and their weights into one vector'''
        if (Theta is not None) and (alpha is not None):
            _Theta, _alpha = Theta, alpha
        else:
            _Theta, _alpha = self.Theta, self.alpha
        return np.r_[_Theta.reshape(-1),_alpha]

    def _destack_sol(self,p):
        assert p.size == self.n_atoms*(self.d_atom + 1)
        Theta = p[:self.d_atom*self.n_atoms].reshape(self.n_atoms,self.d_atom)
        alpha = p[-self.n_atoms:]
        return (alpha,Theta)
        
    # Optimization subroutines
    def _maximize_atom_correlation_fun_grad(self,theta):
        """Computes the fun. value and grad. of step 1 objective: max_theta <A(P_theta),r> / <A(P_theta),A(P_theta)>"""
        # Firstly, compute A(P_theta)...
        sketch_theta, jacobian_theta = self.sketch_of_atom(theta,return_jacobian=True)
        
        # ... and its l2 norm
        norm_sketch_theta = np.linalg.norm(sketch_theta)
        # Trick to avoid division by zero (doesn't change anything because everything will be zero)
        if norm_sketch_theta < self.minimum_atom_norm:
            if self.verbose > 1: print(f'norm_sketch_theta is too small ({norm_sketch_theta}), changed to {self.minimum_atom_norm}.')
            norm_sketch_theta = self.minimum_atom_norm

        # Evaluate the cost function
        fun = -np.real(np.vdot(sketch_theta,self.residual))/norm_sketch_theta # - to have a min problem

        # Secondly, get the Jacobian
        grad = ( -np.real(jacobian_theta@np.conj(self.residual))/(norm_sketch_theta) 
                 +np.real( np.real(jacobian_theta@np.conj(sketch_theta)) * np.vdot(sketch_theta,self.residual))/(norm_sketch_theta**3) )

        return (fun,grad)
    
    def maximize_atom_correlation(self,new_theta):
        sol = scipy.optimize.minimize(self._maximize_atom_correlation_fun_grad,
                                      x0=new_theta,
                                      method='L-BFGS-B', jac=True,
                                      bounds=self.bounds_atom)
        return sol.x
    
    def find_optimal_weights(self,normalize_atoms=False):
        """Using the current atoms matrix, find the optimal weights"""
        # Stack real and imaginary parts if necessary
        if np.any(np.iscomplex(self.Atoms)): # True if complex sketch output
            _A = np.r_[self.Atoms.real, self.Atoms.imag]
            _z = np.r_[self.sketch_reweighted.real, self.sketch_reweighted.imag]
        else:
            _A = self.Atoms
            _z = self.sketch_reweighted

        # Normalize if necessary
        if normalize_atoms:
            norms = np.linalg.norm(self.Atoms,axis=0)
            norm_too_small = np.where(norms < self.minimum_atom_norm)[0]
            if norm_too_small.size > 0: # Avoid division by zero
                if self.verbose > 1: print(f'norm of some atoms is too small (min. {norms.min()}), changed to {self.minimum_atom_norm}.')
                norms[norm_too_small] = self.minimum_atom_norm 
            _A = _A/norms
        
        # Use non-negative least squares to find optimal weights
        (_alpha,_) = scipy.optimize.nnls(_A,_z)
        return _alpha
    
    def _minimize_cost_from_current_sol(self,p):
        """
        Computes the fun. value and grad. of step 5 objective: min_alpha,Theta || z - alpha*A(P_Theta) ||_2,
        at the point given by p (stacked Theta and alpha), and updates the current sol to match.
        """
        # De-stack the parameter vector
        (_alpha, _Theta) = self._destack_sol(p)
        
        # Update the weigths
        self.alpha = _alpha
                
        # Update the atom matrix and compute the Jacobians
        self.update_Atoms(_Theta,update_jacobian=True)

        # Now that the solution is updated, update the residual
        self.residual = self.sketch_reweighted - self.sketch_of_solution()
        
        # Evaluate the cost function
        fun  = np.linalg.norm(self.residual)**2
        
        # Evaluate the gradients
        grad = np.empty((self.d_atom+1)*self.n_atoms)
        for k in range(self.n_atoms): # Gradients of the atoms
            grad[k*self.d_atom:(k+1)*self.d_atom] = -2*self.alpha[k]*np.real(self.Jacobians[k]@self.residual.conj()) 
        grad[-self.n_atoms:] = -2*np.real(self.residual@self.Atoms.conj()) # Gradient of the weights

        return (fun,grad)
        
    
    def minimize_cost_from_current_sol(self,ftol=None):
        if ftol is None: ftol = self.step5_ftol
        bounds_Theta_alpha = self.bounds_atom * self.n_atoms + [[self.weight_lower_bound,self.weight_upper_bound]] * self.n_atoms
        sol = scipy.optimize.minimize(self._minimize_cost_from_current_sol,
                                      x0=self._stack_sol(),  # Start at current solution
                                      method='L-BFGS-B', jac=True,
                                      bounds=bounds_Theta_alpha, options={'ftol': ftol})
        (self.alpha,self.Theta) = self._destack_sol(sol.x)
    
    # Instantiation of methods of parent class
    # ========================================
    def sketch_of_solution(self,sol=None):
        """
        Returns the sketch of the solution, A_Phi(P_theta) = sum_k alpha_k A_Phi(P_theta_k).
        
        In: a solution P_theta, either None for the current sol, either a tuple (alpha,Theta) where
            - alpha is a (n_atoms,)-numpy array containing the weights
            - Theta is a (n_atoms,)
        Out: sketch_of_solution: (m,)-array containing the sketch
        """
        if sol is None:
            # Use the current solution
            (_alpha, _Atoms) = (self.alpha, self.Atoms)
        else:
            (_alpha, _Theta) = sol
            _Atoms = self.compute_Atoms_matrix(_Theta)
        return _Atoms@_alpha
        
    def fit_once(self,random_restart=True,n_iterations=None):
        """
        If random_restart is True, constructs a new solution from scratch with CLOMPR, else fine-tune.
        """
        
        if random_restart:
            ## Main mode of operation
            
            # Initializations
            if n_iterations is None:
                n_iterations = 2*self.K # By default: CLOMP-*R* (repeat twice)
            self.initialize_empty_solution()
            self.residual = self.sketch_reweighted
                
            # Main loop
            for i_iteration in range(n_iterations):
                ## Step 1: find new atom theta most correlated with residual
                new_theta = self.randomly_initialize_new_atom()
                new_theta = self.maximize_atom_correlation(new_theta) 
                
                ## Step 2: add it to the support
                self.add_atom(new_theta)
                
                ## Step 3: if necessary, hard-threshold to enforce sparsity
                if self.n_atoms > self.K:
                    beta = self.find_optimal_weights(normalize_atoms=True)
                    index_to_remove = np.argmin(beta)
                    self.remove_atom(index_to_remove)
                    # Shortcut: if the last added atom is removed, we can skip to next iter
                    if index_to_remove == self.K: continue
                        
                        
                ## Step 4: project to find weights
                self.alpha = self.find_optimal_weights()
                
                ## Step 5: fine-tune
                self.minimize_cost_from_current_sol()

                # Cleanup
                self.update_Atoms() # The atoms have changed: we must re-compute their sketches matrix
                self.residual = self.sketch_reweighted - self.sketch_of_solution()
                
        # Final fine-tuning with increased optimization accuracy
        self.minimize_cost_from_current_sol(ftol=0.02*self.step5_ftol)
        
        # Normalize weights to unit sum
        self.alpha /= np.sum(self.alpha)
        
        # Package into the solution attribute
        self.current_sol = (self.alpha, self.Theta)

##########################
# 1: Compressive K-Means #
##########################          
class CLOMP_CKM(CLOMP):
    """
    CLOMP solver for Compressive K-Means (CKM), where we fit a mixture of K Diracs to the sketch.
    The main algorithm is handled by the parent class.
    """
    
    def __init__(self,Phi,K,bounds,sketch=None,sketch_weight = 1.,verbose=0):
        super(CLOMP_CKM, self).__init__(Phi,K,Phi.d,bounds,sketch,sketch_weight,verbose)
    
    def sketch_of_atom(self,theta_k,return_jacobian=False):
        """
        Computes and returns A_Phi(P_theta_k) for an atom P_theta_k.
        possibly with the jacobian, of size (d_atom,m)
        """
        assert theta_k.size == self.d_atom
        
        sketch_of_atom = self.Phi(theta_k)
        
        if return_jacobian:
            jacobian = self.Phi.grad(theta_k)
            return sketch_of_atom, jacobian
        else:
            return sketch_of_atom
        
    def set_bounds_atom(self,bounds):
        """
        Should set self.bounds_atom to a list of length d_atom of lower and upper bounds, i.e.,
            self.bounds_atom = [[lowerbound_1,upperbound_1], ..., [lowerbound_d_atom,upperbound_d_atom]]
        """
        assert bounds.shape == (2,self.Phi.d)
        self.bounds = bounds # data bounds
        self.bounds_atom = bounds.T.tolist()
        
    def randomly_initialize_new_atom(self):
        new_theta = np.random.uniform(self.bounds[0],self.bounds[1])
        return new_theta
    
    def get_centroids(self):
        return self.current_sol[1]
    
########################
#  2: Compressive GMM  #
########################

## 2.1 (diagonal) GMM with CLOMP
class CLOMP_dGMM(CLOMP):
    """
    CLOMP solver for diagonal Gaussian Mixture Modeling (dGMM), where we fit a mixture of K Gaussians
    with diagonal covariances to the sketch.
    The main algorithm is handled by the parent class.
    Requires the feature map to be Fourier features.
    
    Init_variance_mode is either "bounds" or "sketch" (default).
    """
    
    def __init__(self,Phi,K,bounds,sketch=None,sketch_weight= 1.,init_variance_mode="sketch",verbose=0):
        # Check that the feature map is an instance of RFF, otherwise computations are wrong
        assert isinstance(Phi,SimpleFeatureMap)
        assert Phi.name.lower() == "complexexponential"
        
        self.variance_relative_lowerbound = (1e-4)**2 # Lower bound on the variances, relative to the data domain size
        self.variance_relative_upperbound =  (0.5)**2 # Upper bound on the variances, relative to the data domain size
        
        d_atom = 2*Phi.d # d parameters for the Gaussian centers and d parameters for the diagonal covariance matrix
        super(CLOMP_dGMM, self).__init__(Phi,K,d_atom,bounds,sketch,sketch_weight,verbose)
        
        self.init_variance_mode = init_variance_mode
        
    def sketch_of_atom(self,theta_k,return_jacobian=False):
        """
        Computes and returns A_Phi(P_theta_k) for an atom P_theta_k.
        possibly with the jacobian, of size (d_atom,m)
        """
        assert theta_k.size == self.d_atom
        
        (mu,sig) = (theta_k[:self.Phi.d],theta_k[-self.Phi.d:])
        sketch_of_atom = fourierSketchOfGaussian(mu,np.diag(sig),self.Phi.Omega,self.Phi.xi,self.Phi.c_norm)
        
        if return_jacobian:
            jacobian = 1j*np.zeros((self.d_atom,self.Phi.m))
            jacobian[:self.Phi.d] = 1j*self.Phi.Omega * sketch_of_atom # Jacobian w.r.t. mu
            jacobian[self.Phi.d:] = -0.5*(self.Phi.Omega**2) * sketch_of_atom # Jacobian w.r.t. sigma
            return sketch_of_atom, jacobian
        else:
            return sketch_of_atom
        
    def set_bounds_atom(self,bounds):
        """
        Should set self.bounds_atom to a list of length d_atom of lower and upper bounds, i.e.,
            self.bounds_atom = [[lowerbound_1,upperbound_1], ..., [lowerbound_d_atom,upperbound_d_atom]]
        """
        assert bounds.shape == (2,self.Phi.d)
        self.bounds = bounds # data bounds
        self.bounds_atom = bounds.T.tolist()
        for i in range(self.Phi.d): # bounds for the variance in each dimension
            max_variance_this_dimension = (bounds[1][i]-bounds[0][i])**2
            self.bounds_atom.append([self.variance_relative_lowerbound*max_variance_this_dimension,
                                     self.variance_relative_upperbound*max_variance_this_dimension]) 
        
    def randomly_initialize_new_atom(self):
        mu0 = np.random.uniform(self.bounds[0],self.bounds[1]) # initial mean
        # check we can use sketch heuristic (large enough m)
        MINIMAL_C_VALUE = 6
        MAXIMAL_C_VALUE = 25
        MINIMAL_POINTS_PER_BOX = 5
        if self.init_variance_mode == "sketch":
            c = max(self.Phi.m//MINIMAL_POINTS_PER_BOX,MAXIMAL_C_VALUE)
            if c < MINIMAL_C_VALUE:
                self.init_variance_mode = "bounds"
        
        if self.init_variance_mode == "sketch":
            sigma2_bar = estimate_Sigma_from_sketch(self.sketch,self.Phi,c=c)
            sig0 = sigma2_bar[0]*np.ones(self.Phi.d)
        elif self.init_variance_mode == "bounds":
            sig0 = (10**np.random.uniform(-0.8,-0.1,self.Phi.d) * (self.bounds[1]-self.bounds[0]))**2 # initial covariances
        else:
            raise NotImplementedError

        new_theta = np.append(mu0,sig0)
        return new_theta
    
    def get_GMM(self):
        (weights,_Theta) = self.current_sol
        (K,d) = self.n_atoms, self.Phi.d
        mus = np.zeros((K,d))
        Sigmas = np.zeros((K,d,d))
        for k in range(K):
            mus[k] = _Theta[k][:d]
            Sigmas[k] = np.diag(_Theta[k][d:])

        return (weights,mus,Sigmas)


## 2.2 (diagonal) GMM with Hierarchical Splitting
class CLHS_dGMM(CLOMP_dGMM):
    """
    CL Hierarchical Splitting solver for diagonal Gaussian Mixture Modeling (dGMM), where we fit a mixture of K Gaussians
    with diagonal covariances to the sketch.
    Due to strong overlap, this algorithm is strongly based on CLOMP for GMM algorithm (its the parent class),
    but the core fitting method is overridden.
    Requires the feature map to be Fourier features.
    """
    
    def __init__(self,Phi,K,bounds,sketch=None,sketch_weight = 1.,init_variance_mode="sketch",verbose=0):
        super(CLHS_dGMM, self).__init__(Phi,K,bounds,sketch,sketch_weight,init_variance_mode,verbose)
        
    # New split methods
    def split_one_atom(self,k):
        """Splits the atom at index k in two.
        The first result of the split is replaced at the k-th index,
        the second result is added at the end of the atom list."""
        
        # Pick the dimension with most variance 
        theta_k = self.Theta[k]
        (mu,sig) = (theta_k[:self.Phi.d],theta_k[-self.Phi.d:])
        i_max_var = np.argmax(sig) 
        
        # Direction and stepsize
        direction_max_var = np.zeros(self.Phi.d)
        direction_max_var[i_max_var] = 1. # i_max_var-th canonical basis vector in R^d
        SD_max = np.sqrt(sig[i_max_var]) # max standard deviation 
        
        # Split!
        self.add_atom(       np.append(mu + SD_max*direction_max_var,sig) ) # "Right" split
        self.replace_atom(k, np.append(mu - SD_max*direction_max_var,sig) ) # "Left" split
        
        
    def split_all_atoms(self):
        """Self-explanatory"""
        for k in range(self.n_atoms):
            self.split_one_atom(k)
        
        
    # Override the main fit_once method 
    def fit_once(self,random_restart=True):
        """
        If random_restart is True, constructs a new solution from scratch with CLHS, else fine-tune.
        """
        
        if random_restart:
            ## Main mode of operation
            
            # Initializations
            n_iterations = int(np.ceil(np.log2(self.K))) # log_2(K) iterations
            self.initialize_empty_solution()
            self.residual = self.sketch_reweighted
            
            # Add the starting atom
            new_theta = self.randomly_initialize_new_atom()
            new_theta = self.maximize_atom_correlation(new_theta)
            self.add_atom(new_theta)
                
            # Main loop
            for i_iteration in range(n_iterations):
                ## Step 1-2: split the currently selected atoms
                self.split_all_atoms()
                
                ## Step 3: if necessary, hard-threshold to enforce sparsity
                while self.n_atoms > self.K:
                    beta = self.find_optimal_weights(normalize_atoms=True)
                    index_to_remove = np.argmin(beta)
                    self.remove_atom(index_to_remove)
                        
                        
                ## Step 4: project to find weights
                self.alpha = self.find_optimal_weights()
                
                ## Step 5: fine-tune
                self.minimize_cost_from_current_sol()

                # Cleanup
                self.update_Atoms() # The atoms have changed: we must re-compute their sketches matrix
                self.residual = self.sketch_reweighted - self.sketch_of_solution()
                
        # Final fine-tuning with increased optimization accuracy
        self.minimize_cost_from_current_sol(ftol=0.02*self.step5_ftol)
        
        # Normalize weights to unit sum
        self.alpha /= np.sum(self.alpha)
        
        # Package into the solution attribute
        self.current_sol = (self.alpha, self.Theta)


##############################################
##############################################
##     !!! What follows is deprecated !!!   ##
##  It's here only for compatibility issues ##
##############################################
##############################################


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

def _GMMToThetas(task,GMM):
    """util function, converts a GMM encoding into atoms form"""
    
    (al,mus,Sigmas) = GMM
    (K,d) = mus.shape
    if task == "gmm":
        d2 = d*2
    elif task == "gmm-nondiag":
        d2 = (d+1)*d
    else:
        raise NotImplementedError
        
    Th = np.zeros((K,d2))
    
    for k in range(K):
        Th[k,0:d] = mus[k]
        
        if task == "gmm":
            Th[k,d:2*d] = np.diag(Sigmas[k])
        elif task == "gmm-nondiag":
            Th[k,d:] = Sigmas[k].reshape(d**2)

    return (Th,al)


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


def split(GMM):
    
    (w,mus,Sigmas) = GMM # Unpack
    (K,d) = mus.shape
    
    # Initialize
    splitted_w = np.zeros(2*K)
    splitted_mus = np.zeros((2*K,d))
    splitted_Sigmas = np.zeros((2*K,d,d))

    for k in range(K):
        
        # Compute eigenvalues
        (eigvals,eigvecs) = np.linalg.eig(Sigmas[k])
        
        # Get largest eigenvector (direction vector)
        i_max = np.argmax(eigvals)
        e_max = eigvecs[:,i_max]
        s_max = np.sqrt(eigvals[i_max])
        
        # First half of the split
        splitted_w[2*k]      = w[k]/2
        splitted_mus[2*k]    = mus[k] + s_max*e_max
        splitted_Sigmas[2*k] = Sigmas[k] # v@np.diag(lam)@v.T
        
        # Second half of the split
        splitted_w[2*k+1]      = w[k]/2
        splitted_mus[2*k+1]    = mus[k] - s_max*e_max
        splitted_Sigmas[2*k+1] = Sigmas[k]
    
    return (splitted_w,splitted_mus,splitted_Sigmas)

def CL_GaussianSplitting(task,sketch,featureMap,K,bounds,dimensions_to_consider=None, nb_cat_per_dim=None,nIterations=None, nRepetitions=1, ftol=1e-6, verbose=0 ):
    """
    """
    ## 0) Defining all the tools we need
    ####################################
    ## 0.1) Handle input
    
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
        nIterations = int(np.ceil(np.log2(K)))
    
    ## 0.1.4) Bounds of the optimization problems
    if bounds is None:
        lowb = -np.ones(d) # by default data is assumed normalized
        uppb = +np.ones(d)
        if verbose > 0: print("WARNING: data is assumed to be normalized in [-1,+1]^d")
    else:
        lowb = bounds[0][dimensions_to_consider]
        uppb = bounds[1][dimensions_to_consider] # Bounds for one centroid

    # Format the bounds for the optimization solver
    boundstheta = np.array([lowb,uppb]).T.tolist()  # bounds for the means
    varianceLowerBound = 1e-8
    for i in range(d): boundstheta.append([varianceLowerBound,(uppb[i]-lowb[i])**2]) # bounds for the variance
    
    if task == "gmm-nondiag":
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
    if task == "gmm":
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
        
        # Find the first atom theta, most correlated with residual
            
        mu0 = np.random.uniform(lowb,uppb) # initial mean
        sig0 = (10**np.random.uniform(-0.8,-0.1,d) * (uppb-lowb))**2 # initial covariances
        th_0 = _stackAtom("gmm",mu0,sig0)

        # And solve with LBFGS   
        sol = minimize(lambda th: _CLOMPR_step1_fun_grad("gmm",Phi,th,r,z_to_ignore,verbose),
                                    x0 = th_0, method=opt_method, jac=True,
                                    bounds=boundstheta)

        if task == "gmm-nondiag":
            #(mu0,sig0) = _destackAtom("gmm",sol.x,d)
            th_0 = _stackAtom("gmm-nondiag",mu0,np.diag(sig0))
            
            
            sol = minimize(lambda th: _CLOMPR_step1_fun_grad("gmm-nondiag",Phi,th,r,z_to_ignore,verbose),
                                        x0 = th_0, method=opt_method, jac=True,
                                        bounds=boundstheta_finetuning)

            #(mu0,sig0) = _destackAtom("gmm",sol.x,d)

            #th_0 = _stackAtom("gmm-nondiag",mu0,np.diag(sig0))
            
            th_0 = sol.x
            

        # And solve with LBFGS   
        #sol = minimize(lambda th: _CLOMPR_step1_fun_grad(task,Phi,th,r,z_to_ignore,verbose),
        #                                x0 = th_0, method=opt_method, jac=True,
        #                                bounds=boundstheta)
        new_theta = sol.x

        if task == "gmm-nondiag":
            new_theta = _makeGMMfeasible(new_theta,d)
            
        Theta = np.append(Theta,[new_theta],axis=0) 
        alpha = np.array([1.])
        
            
        
        ## 2) Main optimization
        for i in range(nIterations):
            ## 2.1] Step 1 : split all atoms
            newGMM = split(_ThetasToGMM(task,Theta,alpha))
            (Theta,alpha) = _GMMToThetas(task,newGMM)

            ## 2.2] Step 2 : make them the new support
            A = np.empty([m,0])
            for theta_i in Theta:
                Apthi = _sketchAtom(task,Phi,theta_i,z_to_ignore)
                A = np.c_[A,Apthi]
                
            ## 2.3] Step 3 : if necessary, hard-threshold to nforce sparsity
            while Theta.shape[0] > K:
                norms = np.linalg.norm(A,axis=0)
                norms[np.where(norms < 1e-15)[0]] = 1e-15 # Avoid /0
                A_norm =  A/norms # normalize, unlike step 4
                A_normri = np.r_[A_norm.real, A_norm.imag] 
                (beta,_) = nnls(A_normri,sketch_ri) # non-negative least squares

                index_to_delete = np.argmin(beta)
                Theta = np.delete(Theta, index_to_delete, axis=0)
                A = np.delete(A, index_to_delete, axis=1)


            ## 2.4] Step 4 : project to find weights
            Ari = np.r_[A.real, A.imag]
            (alpha,_) = nnls(Ari,sketch_ri) # non-negative least squares


            ## 2.5] Step 5
            p0 = _stackTheta(task,Theta,alpha) # Initialize at current solution 
            # Compute the bounds for step 5 : boundsOfOneAtom * numberAtoms then boundsOneWeight * numberAtoms
            if task == "gmm-nondiag":
                boundsThetaAlpha = boundstheta_finetuning * Theta.shape[0] + [[1e-9,2]] * Theta.shape[0]
            else:
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
        if task == "gmm-nondiag":
            if verbose > 0: print('finetuning')
            Theta_new = np.zeros((K,d*(d+1))) # Expand to have full cov matrix
            for k in range(K):
                (mu,sig2) = _destackAtom("gmm",Theta[k],d)
                # put current sol on the diagonal of full covariance matrix
                Theta_new[k] = _stackAtom("gmm-nondiag",mu,np.diag(sig2))
            Theta = Theta_new # overwrite
            
            boundsThetaAlpha = boundstheta_finetuning * Theta.shape[0] + [[1e-9,2]] * Theta.shape[0]
            
        if task == "gmm-nondiag" or ftol >= 1e-8:
            p0 = _stackTheta(task,Theta,alpha)
            
            sol = minimize(lambda p: _CLOMPR_step5_fun_grad(task,Phi,p,sketch,z_to_ignore),
                                            x0 = p0, method=opt_method, jac=True,
                                            bounds=boundsThetaAlpha)  # Here ftol is much smaller
            (Theta,alpha) = _destackTheta(task,sol.x,Phi.d)  
            if task == "gmm-nondiag":
                for k in range(Theta.shape[0]):
                    Theta[k] = _makeGMMfeasible(Theta[k],d)
        
        
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
        return _ThetasToGMM(task,bestTheta,bestalpha)
        
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
