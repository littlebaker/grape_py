from typing import List, Union
import numpy as np
from scipy.linalg import expm
from qutip import Qobj
from scipy.optimize import minimize, OptimizeResult
import torch

torch.set_default_device('cuda:0')
device = torch.device('cuda:0')

"""
The algorithm is based on the paper: 
Optimal control of coupled spin dynamics: design of NMR pulse sequences by gradient ascent algorithms,
which is currently as same as the algorithm in `grape_hamiltonian.py`
"""



def _vec(v: torch.Tensor) -> torch.Tensor:
    # column vectorization
    return v.T.reshape((-1, 1))


def _unvec(v: torch.Tensor, shape: tuple) -> torch.Tensor:
    # reshape column vector to matrix
    Shape = (shape[1],shape[0])
    return v.reshape(Shape).T


def torch_trace(grad_original: torch.Tensor) -> torch.Tensor:
    grad_diag = torch.diagonal(grad_original, dim1 = -1, dim2 = -2)
    grad = torch.sum(grad_diag, dim = -1)
    return grad


def _liouvillian_operator_batch(H: torch.Tensor, c_ops: List[torch.Tensor]):
    _, n, _ = H.size()
    # l1 is the vectorized form of the Hamiltonian operator
    _l1 = -1j * (
        torch.kron(torch.eye(n), H) - torch.kron(torch.transpose(H, 1, 2).contiguous(), torch.eye(n))
    )
    # l2 is the vectorized form of the Lindblad collapse operators
    _l2 = 0
    for c_op in c_ops:
        _l2 += torch.kron(c_op.conj(), c_op) \
               - 0.5 * torch.kron(torch.eye(n), c_op.conj().T @ c_op) \
               - 0.5 * torch.kron((c_op.conj().T @ c_op).T, torch.eye(n))

    return _l1 + _l2


def _liouvillian_propagator(
    H0: torch.Tensor,
    Hk: torch.Tensor,
    c_ops: List[torch.Tensor],
    dissipators: Union[List[torch.Tensor], None],
    delta_t: float,
    u_kj: torch.Tensor,
) -> torch.Tensor:

    _Hk_sum = torch.tensordot(u_kj, Hk, dims=([0], [0]))  # Nxnxn
    L = _liouvillian_operator_batch(H0 + _Hk_sum, c_ops) + 0 if dissipators is None else torch.sum(dissipators)
    Lj = torch.matrix_exp(delta_t * L)
    return Lj


def _liouvillian_density_matrix(Lj: torch.Tensor, rho_0: Union[torch.Tensor]) -> torch.Tensor:
    N, n2, _ = Lj.size()
    rho_0 = _vec(rho_0)

    rhoj = torch.empty((N, n2, 1), dtype=torch.complex128)
    rhoj[0] = rho_0
    for j in range(1, N):
        rhoj[j] = Lj[j] @ rhoj[j - 1]

    return rhoj


def _liouvillian_lambda(Lj: torch.Tensor, C: Union[torch.Tensor]) -> torch.Tensor:
    N, n2, _ = Lj.size()
    c_vec = _vec(C)

    lambdaj = torch.empty((N, n2, 1), dtype=torch.complex128)
    lambdaj[-1] = Lj[-1].conj().T @ c_vec
    for j in range(N - 2, -1, -1):
        lambdaj[j] = Lj[j + 1].conj().T @ lambdaj[j + 1]

    return lambdaj


def _liouvillian_gradient(
    lambdaj: torch.Tensor,
    rhoj: torch.Tensor,
    delta_t: float,
    Hk: torch.Tensor
) -> torch.Tensor:

    Lk = _liouvillian_operator_batch(Hk, [])
    Lk = torch.unsqueeze(Lk, 1)
    lambdaj_dagger = lambdaj.conj().swapaxes(1, 2)
    commutator = Lk * delta_t

    grad_original = lambdaj_dagger @ commutator @ rhoj
    grad = torch_trace(grad_original)

    return grad


def grape_liouvillian_bfgs(
        u_0: np.ndarray,
        rho_0: Qobj,
        C: Qobj,
        T: int,
        H0: Qobj,
        Hk: List[Qobj],
        c_ops: List[Qobj] = [],
        dissipators: Union[List[Qobj], None] = None,
        target: str = "trace_real",
        max_iter: int = 1000,
        gtol: float = 1e-6,
        atol: float = 1e-6,
        disp: bool = True,
        method="direct"
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

    for i, c_op in enumerate(c_ops):
        if isinstance(c_op, Qobj):
            c_ops[i] = c_op.full()
        assert c_op.shape == (n, n), "collapse operator must be a square matrix"

    # copy u_0
    u_kj = np.array(u_0)

    def _f(x):
        x_t = torch.from_numpy(x).type(torch.complex128).to(device)
        rho_0_t = torch.from_numpy(rho_0).type(torch.complex128).to(device)
        H0_t = torch.from_numpy(H0).type(torch.complex128).to(device)
        Hk_t = np.array(Hk)
        Hk_t = torch.from_numpy(Hk_t).type(torch.complex128).to(device)
        C_t = torch.from_numpy(C).type(torch.complex128).to(device)


        fx = torch.real(
            torch_trace(
                rho_0_t.T.conj()
                @
                _unvec(
                    _liouvillian_lambda(
                        _liouvillian_propagator(H0_t, Hk_t, c_ops, dissipators, delta_t, x_t.reshape(m, N)),
                        C_t
                    )[0],
                    (n, n)
                )
            )
        )
        return -1 * torch.Tensor.cpu(fx).numpy()

    def _grad_f(x):
        x = torch.from_numpy(x).type(torch.complex128).to(device)
        C_t = torch.from_numpy(C).type(torch.complex128).to(device)
        H0_t = torch.from_numpy(H0).type(torch.complex128).to(device)
        Hk_t = np.array(Hk)
        Hk_t = torch.from_numpy(Hk_t).type(torch.complex128).to(device)
        rho_0_t = torch.from_numpy(rho_0).type(torch.complex128).to(device)

        _Lj = _liouvillian_propagator(H0_t, Hk_t, c_ops, dissipators, delta_t, x.reshape(m, N))
        _lambda_j = _liouvillian_lambda(_Lj, C_t)
        _rho_j = _liouvillian_density_matrix(_Lj, rho_0_t)
        grad = torch.real(_liouvillian_gradient(_lambda_j, _rho_j, delta_t, Hk_t).flatten())
        return -1 * torch.Tensor.cpu(grad).numpy()


    res = None
    if method == "direct" or method == "cascaded":

        for i in range(max_iter):
            phi = _f(u_kj.flatten())
            grad = _grad_f(u_kj.flatten()).reshape((m, N)).real.astype(np.float64)

            u_kj = u_kj - 10 * grad

            # if np.abs(phi - phi_old) < atol:
            #     break
            if phi < -0.9999:
                break

            print(phi)
        res = u_kj
    if method.lower() == "bfgs" or method == "cascaded":
        res = minimize(
            _f,
            u_kj.flatten(),
            method="BFGS",
            jac=_grad_f,
            options={
                "gtol": gtol,
                "disp": disp,
                "maxiter": max_iter,
            }
        )

    return res
