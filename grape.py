import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
from qutip import Qobj
from scipy.optimize import BFGS, line_search, minimize


def cal_Uj(H0, Hk, delta_t, u_kj):  # u_kj mxN, Hk mxnxn, Uj Nxnxn matrix
    m, N = np.shape(u_kj)
    n = H0.shape[0]

    sigma = np.tensordot(u_kj, Hk, axes=([0], [0]))  # Nxnxn
    Uj = expm(-1j * delta_t * (H0 + sigma))
    return Uj


def cal_rhoj(Uj, rho_0):
    N = np.shape(Uj)[0]
    n = np.shape(Uj)[1]
    rho_0 = np.array(rho_0)

    rhoj = np.ndarray((N, n, n), np.complex128)
    a = list(range(N))
    rhoj[0] = Uj[0] @ rho_0 @ (Uj[0].conj().T)
    for j in range(1, N):
        rhoj[j] = Uj[j] @ rhoj[j - 1] @ (Uj[j].conj().T)

    return rhoj


def cal_lambdaj(Uj, C):
    N = np.shape(Uj)[0]
    n = np.shape(Uj)[1]
    C = np.array(C)

    lambdaj = np.ndarray((N, n, n), np.complex128)
    lambdaj[-1] = C
    for j in range(N - 2, -1, -1):
        lambdaj[j] = Uj[j + 1].conj().T @ lambdaj[j + 1] @ Uj[j + 1]

    return lambdaj


def gradient(lambdaj, rhoj, delta_t, Hk):  # lambdaj: n*n, rhoj: N*n*n delta_t: constant number Hk : m*n*n
    m = len(Hk)
    N = np.shape(rhoj)[0]
    lambdaj = np.array(lambdaj)
    rhoj = np.array(rhoj)
    Hk = np.array(Hk)

    commutation = 1j * delta_t * (np.matmul(Hk[:, None], rhoj) - np.matmul(rhoj, Hk[:, None]))
    lambdaj = lambdaj.conj().swapaxes(1, 2)
    ipmat = -np.matmul(lambdaj, commutation)
    um = np.trace(ipmat, axis1=2, axis2=3)

    return um





def grape(
    H0, 
    Hk, 
    u_0, 
    rho_0, 
    C, 
    T, 
    alpha, 
    target="trace_real", 
    max_iter=1000, 
    fidility=0.9999, 
    epsilon=None,
):
    """grape algorithm

    Args:
        H0 (np.ndarray): nxn matrix, basic Hamiltonian
        Hk (np.ndarray): nxn matrix or list of nxn matrices, control Hamiltonian
        u_0 (np.ndarray): mxN matrix u[k, j] is the k-th control function at time j
        rho_0 (np.ndarray): nxn matrix initial state
        C (np.ndarray): final target operator
        T (float): final time
        alpha (float, optional): step size. Defaults to 1e-3.
        target (str, optional): target function. Defaults to "trace_real", options: ["trace_real", "trace_both", "abs"].
        epsilon (float, optional): convergence threshold. Defaults to 1e-3.
        max_iter (int, optional): maximum number of iterations. Defaults to 1000.
        fidility (float, optional): fidility threshold. Defaults to 0.9999.
        method (str, optional): optimization method. Defaults to "default", options: ["default", "BFGS"].
    """

    # basic check
    m, N = u_0.shape
    assert m == len(Hk), "number of control functions must be equal to number of control Hamiltonians"
    delta_t = T / N
    if isinstance(rho_0, Qobj):
        rho_0 = rho_0.full()
    if isinstance(C, Qobj):
        C = C.full()
        
    assert target in ["trace_real", "trace_both", "abs"], "target function not supported"

    # check hamiltonian shape
    if isinstance(H0, Qobj):
        H0 = H0.full()
    n = H0.shape[0]
    assert H0.shape == (n, n), "basic Hamiltonian must be a square matrix"
    for i, H in enumerate(Hk):
        if isinstance(H, Qobj):
            Hk[i] = H.full()
        assert H.shape == (n, n), "control Hamiltonian must be a square matrix"

    # copy u_0
    u_kj = np.array(u_0)

    # start iteration
    threshold = np.inf
    Uj = cal_Uj(H0, Hk, delta_t, u_kj)
    rhoj = cal_rhoj(Uj, rho_0)
    lambdaj = cal_lambdaj(Uj, C)

    reach_threshold = False
    
    for i in range(max_iter):
        if epsilon is not None and threshold < epsilon:
            reach_threshold = True
            print("threshold reached, iteration number: ", i)
            break

        # last phi
        phi = np.trace(C.T.conj() @ rhoj[-1])
        if target == "trace_real":
            pass
        elif target == "trace_both":
            phi = np.real(phi)
        elif target == "abs":
            phi = phi * (phi.conj())
        else:
            raise ValueError("target function not supported")

        # calculate update_matrix and update u_kj, step to optimization
        update_matrix = None
        if target == "trace_real":
            update_matrix = gradient(lambdaj, rhoj, delta_t, Hk).real.astype(np.float64)
        elif target == "trace_both":
            lx = (lambdaj + lambdaj.T.conj()) / 2
            ly = (lambdaj - lambdaj.T.conj()) / 2j
            rx = (rhoj + rhoj.T.conj()) / 2
            ry = (rhoj - rhoj.T.conj()) / 2j
            umx = gradient(lx, rx, delta_t, Hk).real.astype(np.float64)
            umy = gradient(ly, ry, delta_t, Hk).real.astype(np.float64)
            update_matrix = - umx - umy
        elif target == "abs":
            um1 = gradient(lambdaj, rhoj, delta_t, Hk)
            um2 = np.trace(rhoj[-1].conj().T @ C)
            update_matrix = -2 * np.real(um1 * um2).real.astype(np.float64)
        else:
            raise ValueError("target function not supported")

        u_kj = u_kj + alpha * update_matrix

        # update threshold
        # calculate new Uj
        Uj_new = cal_Uj(H0, Hk, delta_t, u_kj)
        # calculate rhoj
        rhoj_new = cal_rhoj(Uj_new, rho_0)
        # calculate lambdaj
        lambdaj_new = cal_lambdaj(Uj_new, C)
        # calculate phi_new
        phi_new = np.trace(np.dot(C.T.conjugate(), rhoj_new[N - 1]))
        threshold = phi_new - phi

        # results to next iteration
        Uj = Uj_new
        rhoj = rhoj_new
        lambdaj = lambdaj_new
        
        if fidility is not None and phi_new > fidility:
            reach_threshold = True
            print("fidility reached, iteration number: ", i)
            break

    if not reach_threshold:
        print("max iterations reached")

    return threshold, u_kj, rhoj



def grape_bfgs(
    H0, 
    Hk, 
    u_0, 
    rho_0, 
    C, 
    T, 
    target="trace_real", 
    max_iter=1000,
    gtol=1e-6
):
    # basic check
    m, N = u_0.shape
    assert m == len(Hk), "number of control functions must be equal to number of control Hamiltonians"
    delta_t = T / N
    if isinstance(rho_0, Qobj):
        rho_0 = rho_0.full()
    if isinstance(C, Qobj):
        C = C.full()
        
    assert target in ["trace_real", "trace_both", "abs"], "target function not supported"

    # check hamiltonian shape
    if isinstance(H0, Qobj):
        H0 = H0.full()
    n = H0.shape[0]
    assert H0.shape == (n, n), "basic Hamiltonian must be a square matrix"
    for i, H in enumerate(Hk):
        if isinstance(H, Qobj):
            Hk[i] = H.full()
        assert H.shape == (n, n), "control Hamiltonian must be a square matrix"

    # copy u_0
    u_kj = np.array(u_0)
    
    def _f(x):
        return -np.trace(np.dot(C.T.conjugate(), cal_rhoj(cal_Uj(H0, Hk, delta_t, x.reshape(m, N)), rho_0)[-1])).real.astype(np.float64)
    
    def _grad_f(x):
        _Uj = cal_Uj(H0, Hk, delta_t, x.reshape(m, N))
        _lambda_j = cal_lambdaj(_Uj, C)
        _rho_j = cal_rhoj(_Uj, rho_0)
        return -gradient(_lambda_j, _rho_j, delta_t, Hk).flatten().real.astype(np.float64)
    
    res = minimize(
        _f,
        u_kj.flatten(),
        method="BFGS",
        jac=_grad_f,
        options={
            "gtol": gtol,
            "disp": True,
            "maxiter": max_iter,
        }
    )
    # print(res)
    return res