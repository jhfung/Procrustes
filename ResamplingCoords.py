import numpy as np
from ripser import ripser

from scipy.spatial.distance import pdist, squareform
from scipy.stats import bernoulli

import warnings

class PHCoord():
    def __init__(self):
        pass
    
    def fit(self, X, n_perm=None):
        X = np.array(X)
        self.X = X
        
        # Use Ripser's built-in greedy subsampling if enabled, otherwise use all points
        if n_perm is None:
            n_perm = X.shape[0]
            
        # Compute persistent cohomology
        p = 47
        
        self.ph_results = ripser(X, coeff=p, do_cocycles=True, n_perm=n_perm)
        self.dperm2all = self.ph_results['dperm2all']
        self.idx_perm = self.ph_results['idx_perm']
        self.ph1_dgm = self.ph_results['dgms'][1]
        
        cocycle_ix = np.argmax(self.ph1_dgm[:,1] - self.ph1_dgm[:,0])
        self.cocycle = self.ph_results['cocycles'][1][cocycle_ix]
        self.cocycle[:,2][self.cocycle[:,2] > p/2] = self.cocycle[:,2][self.cocycle[:,2] > p/2] - p
        self.birth, self.death = self.ph1_dgm[cocycle_ix, 0], self.ph1_dgm[cocycle_ix, 1]
        self.persistence = self.death - self.birth
        
        # Compute naive harmonic cocycle representative
        cocycle_matrix = np.zeros(shape=(n_perm, n_perm))
        c0 = np.nonzero(self.cocycle[:,0][:,np.newaxis] == self.idx_perm)[1]
        c1 = np.nonzero(self.cocycle[:,1][:,np.newaxis] == self.idx_perm)[1]
        cocycle_matrix[c0, c1] = self.cocycle[:,2]
        cocycle_matrix -= cocycle_matrix.T
        
        self.r = 0.1 * self.birth + 0.9 * self.death
        self.X1 = np.where(np.triu(self.dperm2all[:,self.idx_perm] < self.r, k=1))
        n_edges = len(self.X1[0])
        
        self.d0 = np.zeros(shape=(n_edges, n_perm))
        self.d0[np.arange(n_edges), self.X1[0]] = -1
        self.d0[np.arange(n_edges), self.X1[1]] = 1
        
        self.alpha = cocycle_matrix[self.X1[0], self.X1[1]]
        
        self.w = np.ones(n_edges)
        
        d0Tw = self.d0.T * self.w
        
        self.phase = np.linalg.lstsq(np.dot(d0Tw, self.d0), np.dot(d0Tw, -self.alpha), rcond=None)[0]
        
        self.beta = self.alpha + np.dot(self.d0, self.phase)

class Resampler:
    def __init__(self, X, d=None, metric='euclidean', seed=42):
        self.X = np.asarray(X)
        self.n, self.d = self.X.shape
        self.dists = squareform(pdist(X, metric=metric))
        self.rng = np.random.default_rng(seed)
        
    def generate_sample(self, method, eps=None, k=None, size=50):
        if method == 'count':
            if eps is None:
                raise ValueError("eps cannot be None if method is 'count'.")
            else:
                weights = np.sum(self.dists < eps, axis=0)
                weights = 1 / weights
        elif method == 'nn':
            if k is None:
                raise ValueError("k cannot be None if method is 'nn'.")
            else:
                weights = np.partition(self.dists, k, axis=0)[k] ** self.d
        else:
            raise NotImplementedError("method must be either 'count' or 'nn'.")
        
        scaled_weights = size * weights / np.sum(weights)
        if np.any(scaled_weights >= 1):
            warnings.warn('Some weights exceed 1.  Choose a smaller size.')
            scaled_weights = np.minimum(scaled_weights, 1)
        
        rs = self.rng.integers(2 ** 32)
        a = (bernoulli.rvs(scaled_weights, random_state=rs) == 1)
        return a
