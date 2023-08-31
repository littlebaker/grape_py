import numpy as np
import matplotlib.pyplot as plt


def grape(H0, Hk, u_0, rho_0, C, T, alpha=1e-3, epsilon=1e-3, max_iter=1000):
    """grape algorithm

    Args:
        H0 (np.ndarray): nxn matrix, basic Hamiltonian
        Hk (np.ndarray): nxn matrix or list of nxn matrices, control Hamiltonian
        u_0 (np.ndarray): u[k, j] is the k-th control function at time j
        rho_0 (np.ndarray): initial state
        C (np.ndarray): final target operator
        T (float): final time
        alpha (float, optional): step size. Defaults to 1e-3.
        epsilon (float, optional): convergence threshold. Defaults to 1e-3.
        max_iter (int, optional): maximum number of iterations. Defaults to 1000.
    """
    
    # basic check
    m, N = u_0.shape
    assert m == len(Hk), "number of control functions must be equal to number of control Hamiltonians"
    delta_t = T / N
    
    # check hamiltonian shape
    n = H0.shape[0]
    assert H0.shape == (n, n), "basic Hamiltonian must be a square matrix"
    for H in Hk:
        assert H.shape == (n, n), "control Hamiltonian must be a square matrix"
    
    
    # copy u_0
    u_kj = np.array(u_0)
    
    # start iteration
    theshold = np.inf
    for i in range(max_iter):
        if theshold < epsilon:
            break 

        # TODO: calculate Uj
        Uj = cal_Uj(H0, Hk, delta_t, u_kj)

        # TODO: calculate rho_j
        rho_j = cal_rhoj(Uj, rho_0)

        # TODO: calculate lambda_j
        lambda_j = cal_lambdaj(Uj, C)
        
        # TODO: calculate update_matrix and update u_kj
        update_matrix = None
        u_kj = u_kj + alpha * update_matrix
        
        # TODO: update threshold
        theshold = None
        
    return theshold, u_kj


def cal_Uj(H0, Hk, delta_t, u_kj):
    # TODO
    pass

def cal_rhoj(Uj, rho_0):
    # TODO
    pass

def cal_lambdaj(Uj, C):
    # TODO
    pass

