from typing import List, Union
from qutip import Qobj
import numpy as np
import torch

def _propagator(H0, Hk, delta_t, u_kj):
    sigma = torch.tensordot(u_kj, Hk, dims=([0], [0]))
    Uj = torch.matrix_exp(-1j * delta_t * (H0 + sigma))
    return Uj

def _density_matrix(Uj, rho_0):
    N = Uj.shape[0]
    n = Uj.shape[1]

    rhoj = torch.empty((N, n, n), dtype=torch.complex128)
    rhoj[0] = Uj[0] @ rho_0 @ (Uj[0].conj().T)
    for j in range(1, N):
        rhoj[j] = Uj[j] @ rhoj[j - 1] @ (Uj[j].conj().T)

    return rhoj

def _lambda(Uj, C):
    N = Uj.shape[0]
    n = Uj.shape[1]

    lambdaj = torch.empty((N, n, n), dtype=torch.complex128)
    lambdaj[-1] = C
    for j in range(N - 2, -1, -1):
        lambdaj[j] = Uj[j + 1].conj().T @ lambdaj[j + 1] @ Uj[j + 1]

    return lambdaj

def gradient(lambdaj, rhoj, delta_t, Hk):
    # lambdaj = torch.tensor(lambdaj, dtype = torch.complex128)
    # rhoj = torch.tensor(rhoj, dtype = torch.complex128)

    commutation = 1j * delta_t * (torch.matmul(Hk.unsqueeze(1), rhoj) - torch.matmul(rhoj, Hk.unsqueeze(1)))
    lambdaj = lambdaj.conj().swapaxes(1, 2)
    ipmat = -torch.matmul(lambdaj, commutation)
    ipmat_diag = torch.diagonal(ipmat, dim1 = -1, dim2 = -2)
    um = torch.sum(ipmat_diag, dim = -1)

    return um


def grape_torch(
        H0: Union[torch.Tensor, Qobj],
        Hk: List[Union[torch.Tensor, Qobj]],
        u_0: torch.Tensor,
        rho_0: Union[torch.Tensor, Qobj],
        C: Union[torch.Tensor, Qobj],
        T: int,
        alpha: float = 1,
        target: str = "trace_real",
        max_iter: int = 500,
        fidility: float = 0.9999,
        epsilon: Union[float, None] = None,
):
    """grape algorithm

    Args:
        H0 (torch.Tensor): nxn matrix, basic Hamiltonian
        Hk (List[torch.Tensor]): nxn matrix or list of nxn matrices, control Hamiltonian
        u_0 (torch.Tensor): mxN matrix u[k, j] is the k-th control function at time j
        rho_0 (torch.Tensor): nxn matrix initial state
        C (torch.Tensor): final target operator
        T (float): final time
        alpha (float, optional): step size. Defaults to 1e-3.
        target (str, optional): target function. Defaults to "trace_real", options: ["trace_real", "trace_both", "abs"].
        epsilon (float, optional): convergence threshold. Defaults to 1e-3.
        max_iter (int, optional): maximum number of iterations. Defaults to 1000.
        fidility (float, optional): fidility threshold. Defaults to 0.9999.
    """

    # basic check
    m, N = u_0.shape
    assert m == len(Hk), "number of control functions must be equal to number of control Hamiltonians"
    delta_t = T / N
    if isinstance(rho_0, Qobj):
        rho_0 = rho_0.full()
        rho_0 = torch.tensor(rho_0)
    if isinstance(C, Qobj):
        C = C.full()
        C = torch.tensor(C)

    assert target in ["trace_real", "trace_both", "abs"], "target function not supported"

    # check hamiltonian shape
    if isinstance(H0, Qobj):
        H0 = H0.full()
        H0 = torch.tensor(H0)
    n = H0.shape[0]
    assert H0.shape == (n, n), "basic Hamiltonian must be a square matrix"
    for i, H in enumerate(Hk):
        if isinstance(H, Qobj):
            Hk[i] = H.full()
        assert H.shape == (n, n), "control Hamiltonian must be a square matrix"
    Hk = np.array(Hk)
    Hk = torch.tensor(Hk, dtype = torch.complex128)


    # copy u_0
    u_kj = u_0.clone().detach()
    u_kj = torch.tensor(u_kj, dtype = torch.complex128)


    # start iteration
    threshold = torch.tensor(float('inf'))
    Uj = _propagator(H0, Hk, delta_t, u_kj)
    rhoj = _density_matrix(Uj, rho_0)
    lambdaj = _lambda(Uj, C)

    reach_threshold = False

    for i in range(max_iter):
        if epsilon is not None and threshold < epsilon:
            reach_threshold = True
            print("threshold reached, iteration number: ", i)
            break

        if i%50 == 1:
            print("iteration number:", i)

        # last phi
        phi = torch.trace(C.T.conj() @ rhoj[-1])
        if target == "trace_real":
            pass
        elif target == "trace_both":
            phi = torch.real(phi)
        elif target == "abs":
            phi = phi * (phi.conj())
        else:
            raise ValueError("target function not supported")

        if i % 50 == 1:
            print("phi:", phi)

        # calculate update_matrix and update u_kj, step to optimization
        update_matrix = None
        if target == "trace_real":
            update_matrix = torch.real(gradient(lambdaj, rhoj, delta_t, Hk))
        elif target == "trace_both":
            lx = (lambdaj + lambdaj.T.conj()) / 2
            ly = (lambdaj - lambdaj.T.conj()) / (2j)
            rx = (rhoj + rhoj.T.conj()) / 2
            ry = (rhoj - rhoj.T.conj()) / (2j)
            umx = torch.real(gradient(lx, rx, delta_t, Hk))
            umy = torch.real(gradient(ly, ry, delta_t, Hk))
            update_matrix = - umx - umy
        elif target == "abs":
            um1 = gradient(lambdaj, rhoj, delta_t, Hk)
            um2 = torch.trace(rhoj[-1].conj().T @ C)
            update_matrix = -2 * torch.real(um1 * um2)
        else:
            raise ValueError("target function not supported")

        u_kj = u_kj + alpha * update_matrix

        # update threshold
        # calculate new Uj
        Uj_new = _propagator(H0, Hk, delta_t, u_kj)
        # calculate rhoj
        rhoj_new = _density_matrix(Uj_new, rho_0)
        # calculate lambdaj
        lambdaj_new = _lambda(Uj_new, C)
        # calculate phi_new
        phi_new = torch.trace(C.T.conj() @ rhoj_new[- 1])
        threshold = abs(phi_new - phi)

        # results to next iteration
        Uj = Uj_new
        rhoj = rhoj_new
        lambdaj = lambdaj_new

        if fidility is not None and abs(phi_new) > fidility:
            reach_threshold = True
            print("fidility reached, iteration number: ", i)
            break

    if not reach_threshold:
        print("max iterations reached")

    return threshold, rhoj

