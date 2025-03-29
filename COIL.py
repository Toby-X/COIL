import numpy as np
import cvxpy as cp
import scipy.sparse.linalg
from utils import rotate_sparse

def denoise(Y, X, lam):
    n = Y.shape[0]
    q = Y.shape[1]
    p = X.shape[1]

    def obj_fn(B, L):
        P = X @ B.T + L
        return cp.sum(cp.logistic(P)-cp.multiply(Y, P)) + lam * cp.norm(L, "nuc")

    def constr(X, L):
        return X.T @ L == 0

    B0 = cp.Variable((q, p))
    L0 = cp.Variable((n, q))
    obj = cp.Minimize(obj_fn(B0, L0))
    constraints = [constr(X, L0)]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    return prob.status, B0.value, L0.value

def SVD_est(L, K):
    n = L.shape[0]
    q = L.shape[1]

    U, S, Vt = scipy.sparse.linalg.svds(L, K)
    Pi = (n/q) ** (1/4) * U @ np.diag(S ** (1/2))
    Gamma = (q/n) ** (1/4) * Vt.T @ np.diag(S ** (1/2))
    
    return Pi, Gamma

def COIL(Y, X, K, lam):
    is_optimal, B, L = denoise(Y, X, lam)
    Pi, Gamma = SVD_est(L, K)

    return B, Pi, Gamma, is_optimal

def COIL_sparse(Y, X, K, lam):
    is_optimal, B, L = denoise(Y, X, lam)
    Pi, Gamma = SVD_est(L, K)

    B_new, Z_new, Gamma_new = rotate_sparse(X, B, Pi, Gamma)

    return B_new, Z_new, Gamma_new, is_optimal