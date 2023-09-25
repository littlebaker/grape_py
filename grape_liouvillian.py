from typing import List, Union
import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
from qutip import Qobj, liouvillian
from scipy.optimize import BFGS, line_search, minimize, OptimizeResult



def _liouvillian_propagator(
    H0: Qobj,
    Hk: List[Qobj],
    c_ops: List[Qobj],
    dissipators: Union[List[Qobj], None],
    delta_t: float,
    u_kj: np.ndarray,
) -> np.ndarray:
    """calculate propagator in liouvillian form

    Args:
        H0 (np.ndarray): 
        Hk (List[np.ndarray]): 
        c_ops (List[np.ndarray]): 
        dissipators (Union[List[np.ndarray], None]): 
        delta_t (float): 
        u_kj (np.ndarray): 
    """
    m, N = np.shape(u_kj)
    n = H0.shape[0] 
    _Hk_sum = np.tensordot(u_kj, Hk, axes=([0], [0]))  # Nxnxn
    
    L = (liouvillian(H0 + _Hk_sum, c_ops) + sum(dissipators)).full()
    Lj = expm(delta_t * L)
    return Lj



def _liouvillian_density_matrix(Lj: np.ndarray, rho_0: Union[Qobj, np.ndarray]) -> np.ndarray:
    """_summary_

    Args:
        Lj (): _description_
        rho_0 (_type_): _description_
    """
    N = np.shape(Lj)[0]
    n = np.shape(Lj)[1]
    rho_0 = np.array(rho_0).flatten().reshape((n * n, 1))

    rhoj = np.ndarray((N, n * n, 1), np.complex128)
    rhoj[0] = Lj[0] @ rho_0 
    for j in range(1, N):
        rhoj[j] = Lj[j] @ rhoj[j - 1]

    return rhoj


def _liouvillian_lambda(Lj: np.ndarray, C: Union[Qobj, np.ndarray]) -> np.ndarray:
    """_summary_

    Args:
        Lj (_type_): _description_
        C (_type_): _description_
    """
    N = np.shape(Lj)[0]
    n = np.shape(Lj)[1]
    C = np.array(C).reshape((n * n, 1))

    lambdaj = np.ndarray((N, n * n, 1), np.complex128)
    lambdaj[-1] = C.flatten().reshape((n * n, 1))
    for j in range(N - 2, -1, -1):
        lambdaj[j] = Lj[j + 1].conj().T @ lambdaj[j + 1]

    return lambdaj


def _liouvillian_gradient(
    lambdaj: np.ndarray, 
    rhoj: np.ndarray, 
    delta_t: float, 
    Hk: List[Qobj]
) -> np.ndarray:
    """_summary_

    Args:
        lambdaj (_type_): _description_
        rhoj (_type_): _description_
        delta_t (_type_): _description_
        Hk (_type_): _description_
    """
    m = len(Hk)
    N, n2, _ = np.shape(rhoj)
    n = int(np.sqrt(n2))
    lambdaj = np.array(lambdaj)
    rhoj = np.array(rhoj)
    Hk = np.array(Hk)

    commutation = 1j * delta_t \
        * (np.matmul(Hk[:, None], rhoj.reshape((N, n, n))) \
        - np.matmul(rhoj.reshape((N, n, n)), Hk[:, None])).reshape((m, n2, 1))
    lambdaj = lambdaj.conj().swapaxes(1, 2)
    ipmat = -np.matmul(lambdaj, commutation)
    um = np.trace(ipmat, axis1=2, axis2=3)

    return um


def grape_liouvillian_bfgs(
    u_0: np.ndarray, 
    rho_0: Qobj, 
    C: Qobj, 
    T: int, 
    H0: Qobj,
    Hk: List[Qobj], 
    c_ops: List[Qobj],
    dissipators: Union[List[Qobj], None],
    target: str = "trace_real", 
    max_iter:int = 1000,
    gtol: float = 1e-6,
    disp: bool = True,
) -> OptimizeResult:
    """grape algorithm, using BFGS method from scipy

    Args:
        H0 (Union[np.ndarray, Qobj]): nxn matrix or a Qobj with same shape, basic Hamiltonian
        Hk (List[Union[np.ndarray, Qobj]]): list of nxn matrices or list of Qobj with same shape, control Hamiltonian
        u_0 (np.ndarray): mxN matrix u[k, j] is the k-th control function at time j
        rho_0 (Union[np.ndarray, Qobj]): nxn matrix or a Qobj with same shape, initial state
        C (Union[np.ndarray, Qobj]): final target operator
        T (int): final time
        c_ops (List[Union[np.ndarray, Qobj]]): list of nxn matrices or list of Qobj with same shape, collapse operators
        dissipators (Union[List[Union[np.ndarray, Qobj]], None]): list of nxn matrices or list of Qobj with same shape, other dissipators with liouvillian form
        target (str, optional): different evaluation function. Defaults to "trace_real". Options: ["trace_real", "trace_both", "abs"].
        max_iter (int, optional): maxium iteration number. Defaults to 1000.
        gtol (float, optional): BFGS options,  gradient tolerence. Defaults to 1e-6.
        disp (bool, optional): BFGS options, whether to print state to console. Defaults to True.

    Returns:
        OptimizeResult: result of optimization, note that x is a 1d array, need to reshape to (m, N)
    """
    # basic check
    m, N = u_0.shape
    assert m == len(Hk), "number of control functions must be equal to number of control Hamiltonians"
    delta_t = T / N
        
    assert target in ["trace_real", "trace_both", "abs"], "target function not supported"

    # copy u_0
    u_kj = np.array(u_0)
    
    def _f(x):
        return -np.trace(
            np.dot(
                C.T.conjugate(), 
                _liouvillian_density_matrix(
                    _liouvillian_propagator(H0, Hk, delta_t, x.reshape(m, N)), 
                    rho_0
                )[-1]
            )
        ).real.astype(np.float64)
    
    def _grad_f(x):
        _Uj = _liouvillian_propagator(H0, Hk, delta_t, x.reshape(m, N))
        _lambda_j = _liouvillian_lambda(_Uj, C)
        _rho_j = _liouvillian_density_matrix(_Uj, rho_0)
        return - _liouvillian_gradient(_lambda_j, _rho_j, delta_t, Hk).flatten().real.astype(np.float64)
    
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

    return res
