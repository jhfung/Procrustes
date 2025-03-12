import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import circmean

def loss1(X, Z):
    """Loss in Euclidean space."""
    return np.sum((X - Z) ** 2) / len(X)

def loss2(X, Z):
    """Loss in terms of angles on the circle R/Z."""
    dist = np.abs(X - Z)
    dist = np.minimum(dist, 1-dist)
    return np.sum(dist ** 2) / len(X)

class GOPP():
    """
    Implements the generalized Procrustes algorithm of ten Berge (1977).  
    Does not handle missing data.
    """
    def __init__(self, max_iter=100, tol=1e-3, verbose=False):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, X):
        self.success = False
        n, p, d = X.shape

        # Initialize
        A  = np.array(X).copy()
        R = np.stack(n * [np.eye(d)])

        losses = np.full(n, np.inf)
        # Compute SVD and update R
        for _ in range(self.max_iter):
            for i in range(n):
                R[i], _ = orthogonal_procrustes(X[i], np.mean(A[np.arange(len(A) != i)], axis=0))
                A[i] = X[i] @ R[i]
                Z = np.mean(A, axis=0)

                new_loss = loss1(A, Z)
                losses[i], diff = new_loss, losses[i] - new_loss

                if self.verbose:
                    print('loss = {:.4f}'.format(losses[i]))
                if diff < self.tol:
                    self.success = True
                    break
            
            if self.success:
                break

        if self.verbose and not self.success:
            print('Max iterations reached before convergence.')

        self.A = A
        self.Z = Z

class ExactCirc():
    """Implements hill-climbing."""
    def __init__(self, num_iter=20, eps_init=0.05, decay_rate=0.5, verbose=False):
        self.num_iter = num_iter
        self.eps_init = eps_init
        self.decay_rate = decay_rate
        self.verbose = verbose

    def fit(self, X):
        """Input should be n x p elements of R/Z."""

        self.success = False
        n, p = X.shape

        A = np.array(X).copy()
        Z = circmean(A, high=1, axis=0)

        curr_loss = loss2(A, Z)
        eps = self.eps_init
        for _ in range(self.num_iter):
            for i in range(n):
                inc = np.zeros_like(A)
                inc[i] = eps
                
                m = np.argmin([loss2(np.mod(A - inc, 1), Z), loss2(A, Z), loss2(np.mod(A + inc, 1), Z)]) - 1
                A = np.mod(A + m * inc, 1)
                Z = circmean(A, high=1, axis=0)

            loss = loss2(A, Z)

            if self.verbose:
                print('loss = {:.4f}'.format(loss))

            eps *= np.exp(-self.decay_rate)

        if self.verbose and not self.success:
            print('Max iterations reached before convergence.')

        self.A = A
        self.Z = Z
        self.loss = loss2(A, Z)

class TwoStageProcrustes():
    def __init__(self, gopp_args={}, ec_args={}, verbose=False):
        self.verbose = verbose
        if verbose:
            gopp_args['verbose'] = True
            ec_args['verbose'] = True
        self.gopp_args = gopp_args
        self.ec_args = ec_args

    def fit(self, X):
        A = np.array(X).copy()
        
        if self.verbose:
            print('Obtaining the orthogonal Procrustes optimization...')
        A2 = np.stack([np.cos(2 * np.pi * A), np.sin(2 * np.pi * A)], axis=-1)

        gopp = GOPP(**self.gopp_args)
        gopp.fit(A2)
        self.gopp_A = np.mod(np.arctan2(gopp.A[..., 1], gopp.A[..., 0]), 2 * np.pi) / (2 * np.pi)
        self.gopp_Z = np.mod(np.arctan2(gopp.Z[:,1], gopp.Z[:,0]), 2 * np.pi) / (2 * np.pi)

        if self.verbose:
            print('Solving the exact problem via hill-climbing...')
        ec = ExactCirc(**self.ec_args)
        ec.fit(self.gopp_A)

        self.A = ec.A
        self.Z = ec.Z
        self.loss = ec.loss