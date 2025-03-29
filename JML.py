import numpy as np
import scipy.linalg
from scipy.optimize import minimize
from scipy.special import softplus
from utils import rotate_sparse, rotate_ortho

def JML(Y, X, K, esp = 1e-3, max_iter = 500):
    q = Y.shape[1]
    p = X.shape[1]
    n = X.shape[0]

    # Initialize Z
    Z0 = np.random.randn(n, K)
    Gamma0 = np.random.randn(q, K)
    B0 = np.random.randn(q, p)

    def iter_z_fix(Z, B_old, Gamma_old):
        def obj_fn(BGamma, shape):
            BGamma = BGamma.reshape(shape)
            B = BGamma[:,:p]
            Gamma = BGamma[:,p:]
            P = X @ B.T + Z @ Gamma.T
            return np.sum(softplus(P)-Y * P)

        shape = (q, p + K)
        BGamma0 = np.hstack([B_old, Gamma_old])
        BGamma0 = BGamma0.flatten()
        opt_res = minimize(obj_fn, BGamma0, args=(shape,), method='L-BFGS-B')
        
        BGamma = opt_res.x.reshape(shape)
        B = BGamma[:,:p]
        Gamma = BGamma[:,p:]
        return B, Gamma
    
    def iter_BG_fix(B, Gamma, Z_old):
        def obj_fn(Z, shape):
            Z = Z.reshape(shape)
            P = X @ B.T + Z @ Gamma.T
            return np.sum(softplus(P)-Y * P)
        
        shape = Z_old.shape
        Z_old = Z_old.flatten()
        opt_res = minimize(obj_fn, Z_old, args=(shape,), method='L-BFGS-B')
        return opt_res.x.reshape(shape)
    
    Z = Z0
    B, Gamma = iter_z_fix(Z0, B0, Gamma0)
    Z_new = iter_BG_fix(B, Gamma, Z)
    err = scipy.linalg.norm(Z_new - Z)
    iter = 0
    while (err > esp) & (iter < max_iter):
        Z = Z_new
        B_new, Gamma_new = iter_z_fix(Z, B, Gamma)
        Z_new = iter_BG_fix(B_new, Gamma_new, Z)
        err = np.max([scipy.linalg.norm(Z_new - Z), scipy.linalg.norm(B_new - B), scipy.linalg.norm(Gamma_new - Gamma)])
        B = B_new
        Gamma = Gamma_new
        iter += 1

    loss = np.sum(softplus(X @ B.T + Z_new @ Gamma.T) - Y * (X @ B.T + Z_new @ Gamma.T))

    return B, Z_new, Gamma, loss, iter

def JML_sparse(Y, X, K, nstart = 10, esp = 1e-3, max_iter = 500):
    B, Z, Gamma, loss, iter = JML(Y, X, K, esp, max_iter)

    for i in range(nstart-1):
        B_new, Z_new, Gamma_new, loss_new, iter_new = JML(Y, X, K, esp, max_iter)
        if loss_new < loss:
            B = B_new
            Z = Z_new
            Gamma = Gamma_new
            loss = loss_new
            iter = iter_new
    
    B, Z, Gamma = rotate_sparse(X, B, Z, Gamma)
    return B, Z, Gamma, loss, iter

def JML_ortho(Y, X, K, nstart = 10, esp = 1e-3, max_iter = 500):
    B, Z, Gamma, loss, iter = JML(Y, X, K, esp, max_iter)

    for i in range(nstart-1):
        B_new, Z_new, Gamma_new, loss_new, iter_new = JML(Y, X, K, esp, max_iter)
        if loss_new < loss:
            B = B_new
            Z = Z_new
            Gamma = Gamma_new
            loss = loss_new
            iter = iter_new
    
    B, Z, Gamma = rotate_ortho(X, B, Z, Gamma)
    return B, Z, Gamma, loss, iter
