import numpy as np
import cvxpy as cp
import scipy.linalg
from scipy.special import expit

def rotate_sparse(X, B, Z, Gamma):
    n = Z.shape[0]
    q = B.shape[0]
    p = B.shape[1]
    K = Gamma.shape[1]
    X_cov = X[:,1:]

    def obj_fn(Ac):
        return cp.sum(cp.abs(B[:,1:] - Gamma @ Ac))
    
    # Ac0 = cp.Variable((p-1, K))
    Ac0 = cp.Variable((K, p-1))
    prob = cp.Problem(cp.Minimize(obj_fn(Ac0)))
    prob.solve()
    Ac = Ac0.value
    a0 = -np.mean(X_cov @ Ac.T + Z, axis=0)
    A = np.concat([a0.reshape(K, 1), Ac], axis=1)

    B_new = B - Gamma @ A
    Z_mid = Z + X @ A.T

    GG_half = scipy.linalg.fractional_matrix_power(Gamma.T @ Gamma, 0.5)
    G_full = GG_half @ Z_mid.T @ Z_mid @ GG_half /n/q
    d, V = np.linalg.eigh(G_full)
    D = np.diag(d**(-1/4))
    G = GG_half @ V @ D / np.sqrt(q)

    Gamma_new = Gamma @ scipy.linalg.inv(G_full.T)
    Z_new = Z_mid @ G

    return B_new, Z_new, Gamma_new

def rotate_ortho(X, B, Z, Gamma):
    n = Z.shape[0]
    q = B.shape[0]
    p = B.shape[1]
    K = Gamma.shape[1]

    A = - Z.T @ X @ scipy.linalg.inv(X.T @ X)

    B_new = B - Gamma @ A
    Z_mid = Z + X @ A.T

    GG_half = scipy.linalg.fractional_matrix_power(Gamma.T @ Gamma, 0.5)
    G_full = GG_half @ Z_mid.T @ Z_mid @ GG_half /n/q
    d, V = np.linalg.eigh(G_full)
    D = np.diag(d**(-1/4))
    G = GG_half @ V @ D / np.sqrt(q)

    Gamma_new = Gamma @ scipy.linalg.inv(G_full.T)
    Z_new = Z_mid @ G

    return B_new, Z_new, Gamma_new

def gen_data_sparse(n, q, p, K, tau, rho, seed, sparse = False):
    np.random.seed(seed)

    ## Covariance Matrix of X, Z
    i, j = np.ogrid[:(p+K), :(p+K)]
    diff = np.abs(i - j)
    Sigma = tau ** diff

    ## generate X, Pi from N(0, Sigma)
    XPi_com = np.random.multivariate_normal(np.zeros(p+K), Sigma, size=n)
    X = XPi_com[:,:p]
    Pi = XPi_com[:,p:]
    # Add full 1 column to X
    X = np.concatenate((np.ones((n, 1)), X), axis=1)
    
    vk = np.random.uniform(.5, 1.5, size=int(q/K))
    # Create identity matrix of size K
    I = np.eye(K)
    # Vectorized solution
    Gamma = np.kron(I, vk.reshape(-1, 1)).reshape(K * len(vk), K)

    B = np.zeros((q, p))
    B[:p, :] = rho
    if sparse:
        for s in range(p):
            B[int(5*s-4):int(5*s+1),s] = rho
    else:
        for s in range(p):
            Rs = s - np.floor(s/5)
            B[int(Rs*q/5+1):int((Rs+1)*q/5+1),s] = rho
    # add full 0 row to the top of B
    B = np.concatenate((np.zeros((q, 1)), B), axis=1)

    B, Pi, Gamma = rotate_sparse(X, B, Pi, Gamma)
    Y0 = X @ B.T + Pi @ Gamma.T
    Y = np.random.binomial(1, expit(Y0))

    return Y, X, B, Pi, Gamma

