from utils import gen_data_sparse
from JML import JML_sparse
from COIL import COIL
import numpy as np
import pandas as pd
import multiprocess as mp
import scipy.linalg
from tqdm import tqdm
import time
import itertools

def experiment_fn(seed, param_dict, nstart=5, esp=1e-3, max_iter=150):
    # Unpack parameters
    n = param_dict['n']
    q = param_dict['q']
    p = param_dict['p']
    K = param_dict['K']
    tau = param_dict['tau']
    rho = param_dict['rho']

    # Generate data
    Y, X, B, Pi, Gamma = gen_data_sparse(n, q, p, K, tau, rho, seed, sparse=True)
    
    # Run JML
    np.random.seed(seed)
    jml_start = time.perf_counter()
    B_jml, Pi_jml, Gamma_jml, loss, niter = JML_sparse(Y, X, K, nstart, esp, max_iter)
    jml_end = time.perf_counter()
    jml_time = jml_end - jml_start
    
    # Run COIL
    np.random.seed(seed)
    lam = np.sqrt((n+q)*np.log(n+q))
    coil_start = time.perf_counter()
    B_coil, Pi_coil, Gamma_coil, is_optimal = COIL(Y, X, K, lam)
    coil_end = time.perf_counter()
    coil_time = coil_end - coil_start
    
    # Compare F norm
    B_jml_norm = scipy.linalg.norm(B_jml - B, ord='fro')
    B_coil_norm = scipy.linalg.norm(B_coil - B, ord='fro')
    Pi_jml_norm = scipy.linalg.norm(Pi_jml - Pi, ord='fro')
    Pi_coil_norm = scipy.linalg.norm(Pi_coil - Pi, ord='fro')
    Gamma_jml_norm = scipy.linalg.norm(Gamma_jml - Gamma, ord='fro')
    Gamma_coil_norm = scipy.linalg.norm(Gamma_coil - Gamma, ord='fro')

    return {
        'seed': seed,
        'n': n,
        'q': q,
        'p': p,
        'K': K,
        'tau': tau,
        'rho': rho,
        'B_jml_norm': B_jml_norm,
        'B_coil_norm': B_coil_norm,
        'Pi_jml_norm': Pi_jml_norm,
        'Pi_coil_norm': Pi_coil_norm,
        'Gamma_jml_norm': Gamma_jml_norm,
        'Gamma_coil_norm': Gamma_coil_norm,
        'jml_time': jml_time,
        'coil_time': coil_time,
        'cvx_optimal': is_optimal,
        'loss': loss,
        'niter': niter
    }

def run_experiment(param_grid, n_exp=100, n_cores=8):
    # param_grid should be a list of dictionaries
    param_comb = []

    if isinstance(param_grid, list) and all(isinstance(item, dict) for item in param_grid):
        param_comb = param_grid
    else:
        keys = param_grid.keys()
        values = param_grid.values()
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            param_comb.append(param_dict)
    
    all_exp = []
    for param_combo in param_comb:
        for exp_id in range(n_exp):
            all_exp.append((exp_id, param_combo))
    
    with mp.Pool(processes=n_cores) as pool:
        results = list(tqdm(
            pool.starmap(experiment_fn, all_exp),
            total=len(all_exp),
            desc="Running experiments",
        ))

    results_df = pd.DataFrame(results)
    
    results_df.to_csv("COIL_com_sparse.csv", index=False)
    return results_df


# Do Exps with (n,q) = (300, 100), (500, 300), p = 5, K = 2, tau = 0.2, rho = 0.5
if __name__ == "__main__":
    param_grid = {
        # 'n': [300, 500],
        # 'q': [100, 300],
        'n': [300],
        'q': [100],
        'p': [2],
        'K': [2],
        'tau': [0.2],
        'rho': [0.5]
    }

    # Run the experiment
    results_df = run_experiment(param_grid, n_exp=100, n_cores=28)
    print(results_df)
