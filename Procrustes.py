import numpy as np
from scipy.spatial.distance import cdist

def weiszfeld(X, tol=1e-10, max_iter=200):
    """The iterative Weiszfeld algorithm for geometric medians.  Possibly runs into division by zero errors."""
    # This is very slow.

    y = np.mean(X, axis=0)
    for _ in range(max_iter):
        y_old = y
        d = np.ravel(cdist(X, [y_old]))
        y = np.average(X, axis=0, weights=1/d)

        if np.linalg.norm(y - y_old) < tol:
            break

    return y

def modified_weiszfeld(X, tol=1e-10, max_iter=200):
    """The modified Weiszfeld algorithm by Vardi and Zhang (PNAS, 2000).  Avoids division by zero errors."""

    y = np.mean(X, axis=0)
    for _ in range(max_iter):
        y_old = y
        d = np.ravel(cdist(X, [y_old]))
        w = 1 / d[d > 0]
        t = np.average(X[d > 0], axis=0, weights=w)

        r = np.sum(w) * (t - y_old)
        if np.any(d == 0):
            r_norm = np.linalg.norm(r)
            if r_norm < 1:
                lam = 0
            else:
                lam = 1 - 1 / r_norm
        else:
            lam = 1
        print('lam = {}'.format(lam))

        y = lam * t + (1 - lam) * y_old

        if np.linalg.norm(y - y_old) < tol:
            break

    return y

class GPA:
    """
    A class implementing alternating minimization to solve the generalized Procrustes problem.  The main method is `.fit()`, which takes in a (number of samples) x (number of points) x (dimension) array.  After fitting, the optimal transformations can be found in `.c`, `.s`, and `.q`, and the centroid is stored in `.Z`.  If the individual aligned samples are desired, call `.fit_transform()` instead.  For the geometric median, initialize with `p=1`.  
    """

    def __init__(self, tol=1e-10, max_iter=200, p=2, centering=True, scaling=False, verbose=False):
        self.tol = tol
        self.max_iter = max_iter
        self.p = p
        self.centering = centering
        self.scaling = scaling
        self.verbose = verbose

    def fit(self, X):
        self.success = False
        k, n, d = X.shape
        mX = np.nan_to_num(X, nan=0)
        m = (~np.any(np.isnan(X), axis=2)).astype(int)
        
        c = np.zeros(shape=(k,d))
        q = np.tile(np.eye(d), (k, 1, 1))
        s = np.ones(shape=k)
        Z = np.nanmean(X, axis=0)

        if self.centering:
            a = (np.eye(n) - m[:,:,np.newaxis] @ m[:,np.newaxis,:] / np.sum(m, axis=1)[:,np.newaxis,np.newaxis]) 
        else:
            a = np.tile(np.eye(n), (k, 1, 1))
        amX = a @ mX
        
        loss = self._compute_loss(X, c, q, s, Z)

        for ii in range(self.max_iter):
            loss_old = loss

            if self.centering:
                c = np.nanmean(Z[np.newaxis, :] - s[:, np.newaxis, np.newaxis] * X @ q,
                               axis=1)

            u, _, vt = np.linalg.svd(amX.transpose(0,2,1) @ (a @ Z))
            q = u @ vt
            
            if self.scaling:
                s = (Z * (amX @ q)).sum(axis=(1,2)) / (amX * amX).sum(axis=(1,2))
                s = np.abs(s)
                s_gm = np.exp(np.mean(np.log(s)))
                s = s / s_gm

            X_hat = c[:, np.newaxis, :] + s[:, np.newaxis, np.newaxis] * X @ q
            if self.p == 2:
                Z = np.nanmean(X_hat, 
                               axis=0)
            elif self.p == 1:
                res = np.empty(shape=(n,d))
                for i in range(n):
                    X_hat_i = X_hat[:,i,:]
                    X_hat_i_dropna = X_hat_i[~np.any(np.isnan(X_hat_i), axis=1)]
                    res[i,:] = weiszfeld(X_hat_i_dropna)
                Z = res
            else:
                raise NotImplementedError

            loss = self._compute_loss(X, c, q, s, Z)

            if self.verbose:
                print('iteration {}: loss = {:.6f}'.format(ii + 1, loss))

            if np.abs(loss - loss_old) < self.tol:
                self.success = True
                break

        if self.verbose & (ii == self.max_iter - 1):
            print('Max iterations reached.')
        
        self.c = c
        self.q = q
        self.s = s
        self.Z = Z

        return self

    def transform(self, X):
        X_hat = self.c[:, np.newaxis, :] + self.s[:, np.newaxis, np.newaxis] * X @ self.q
        return X_hat
    
    def fit_transform(self, X):
        _ = self.fit(X)
        X_hat = self.transform(X)
        return X_hat

    def _compute_loss(self, X, c, q, s, Z):
        k, n, d = X.shape
        X_hat = c[:, np.newaxis, :] + s[:, np.newaxis, np.newaxis] * X @ q
        err = X_hat - Z
        err = np.nan_to_num(err, nan=0)
        loss = np.sum(np.sum(err ** 2, axis=(1,2)) ** (self. p / 2)) ** (1 / self.p)
        return loss / (k ** (1 / self.p) * np.sqrt(n * d))
