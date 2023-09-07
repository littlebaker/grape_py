import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


def grape(H0, Hk, u_0, rho_0, C, T, alpha=0.2, epsilon=1e-5, max_iter=1000):
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
    threshold = np.inf
    Uj = cal_Uj(H0, Hk, delta_t, u_kj)
    rhoj = cal_rhoj(Uj, rho_0)
    lambdaj = cal_lambdaj(Uj, C)

    for i in range(max_iter):
        print("i = ", i)
        print("threshold = ", threshold)
        # if threshold < epsilon:
        #     print("threshold reached")
        #     break

        # last phi
        phi = np.trace(np.dot( C.T.conjugate(), rhoj[N-1] ))
        print("last phi = ", phi)

        # calculate update_matrix and update u_kj, step to optimization
        update_matrix = gradient(lambdaj, rhoj, delta_t, Hk)
        u_kj = u_kj + alpha * update_matrix

        # update threshold
        # calculate new Uj
        Uj_new = cal_Uj(H0, Hk, delta_t, u_kj)
        # calculate rhoj
        rhoj_new = cal_rhoj(Uj_new, rho_0)
        # calculate lambdaj
        lambdaj_new = cal_lambdaj(Uj_new, C)
        # calculate phi_new
        phi_new = np.trace(np.dot( C.T.conjugate(), rhoj_new[N-1] ))
        threshold = phi_new - phi
        print("phi_new = ", phi_new)

        # results to next iteration
        Uj = Uj_new
        rhoj = rhoj_new
        lambdaj = lambdaj_new

    return threshold, u_kj, rhoj[N - 1]


def cal_Uj(H0, Hk, delta_t, u_kj):
    m, N = np.shape(u_kj)
    n = H0.shape[0]

    Uj = np.ndarray((N, n, n), "complex128")
    for j in range(N):
        sigma = np.zeros((n, n), "complex128")
        for k in range(m):
            sigma += u_kj[k, j] * Hk[k]

        Uj[j] = expm(-1j * delta_t * (H0 + sigma))

    return Uj


def cal_rhoj(Uj, rho_0):
    N = np.shape(Uj)[0]
    n = np.shape(Uj)[1]

    rhoj = np.ndarray((N, n, n), "complex128")
    for j in range(N):
        rho = rho_0
        for i in range(j + 1):
            Ut = Uj[i].T.conjugate()
            A = np.dot(Uj[i], rho)
            rho = np.dot(A, Ut)
        rhoj[j] = rho

    return rhoj


def cal_lambdaj(Uj, C):
    N = np.shape(Uj)[0]
    n = np.shape(Uj)[1]

    lambdaj = np.ndarray((N, n, n), "complex128")
    for j in range(N):
        lmda = C
        for i in range(j + 1, N):
            Ut = Uj[i].T.conjugate()
            A = np.dot(Uj[i], lmda)
            lmda = np.dot(A, Ut)
        lambdaj[j] = lmda

    return lambdaj


def gradient(lambdaj, rhoj, delta_t, Hk):
    m = len(Hk)
    N = np.shape(rhoj)[0]

    um = np.ndarray((m, N), "complex128")
    for k in range(m):
        for j in range(N):
            duiyi = 1j * delta_t * (np.dot(Hk[k], rhoj[j]) - np.dot(rhoj[j], Hk[k]))
            ipmat = np.dot(lambdaj[j].T.conjugate(), duiyi)
            um[k, j] = -np.trace(ipmat)

    return um
