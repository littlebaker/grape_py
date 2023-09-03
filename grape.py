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
    threshold = np.inf
    Uj = cal_Uj(H0, Hk, delta_t, u_kj)
    rhoj = cal_rhoj(Uj, rho_0)
    lambdaj = cal_lambdaj(Uj, C)

    for i in range(max_iter):
        if threshold < epsilon:
            break

        phi = np.trace(np.dot(lambdaj[0].T.conjugate(), rhoj[0]))
        # calculate update_matrix and update u_kj
        update_matrix = gradient(lambdaj, delta_t, Hk, rhoj)
        u_kj = u_kj + alpha * update_matrix

        # update threshold
        # calculate new Uj
        Uj_new = cal_Uj(H0, Hk, delta_t, u_kj)
        # calculate rhoj
        rhoj_new = cal_rhoj(Uj_new, rho_0)
        # calculate lambdaj
        lambdaj_new = cal_lambdaj(Uj_new, C)
        # calculate phi_new
        phi_new = np.trace(np.dot(lambdaj_new[0].T.conjugate(), rhoj_new[0]))
        threshold = phi_new - phi

        # results to next iteration
        Uj = Uj_new
        rhoj = rhoj_new
        lambdaj = lambdaj_new

    return threshold, u_kj


def cal_Uj(H0, Hk, delta_t, u_kj):
    m, N = np.shape(u_kj)
    n = H0.shape[0]

    Uj = np.ndarray((N, n, n))
    for j in range(N):
        sigma = np.zeros(n, n)
        for k in range(m):
            sigma += u_kj[k, j] * Hk[k]

        Uj[j] = np.exp(-1j * delta_t * (H0 + sigma))

    return Uj


def cal_rhoj(Uj, rho_0):
    N = np.shape(Uj)[0]
    n = np.shape(Uj)[1]

    rhoj = np.ndarray((N, n, n))
    for j in range(N):
        rho = rho_0
        for i in range(j + 1):
            rho = np.dot(Uj[i], rho, Uj.T.conjugate())
        rhoj[j] = rho

    return rhoj


def cal_lambdaj(Uj, C):
    N = np.shape(Uj)[0]
    n = np.shape(Uj)[1]

    lambdaj = np.ndarray((N, n, n))
    for j in range(N):
        lmda = C
        for i in range(j + 1, N):
            lmda = np.dot(Uj[i], lmda, Uj.T.conjugate())
        lambdaj[j] = lmda

    return lambdaj


def gradient(lambdaj, delta_t, Hk, rhoj):
    m = len(Hk)
    N = np.shape(rhoj)[0]

    um = np.ndarray((m, N))
    for k in range(m):
        for j in range(N):
            duiyi = 1j * delta_t * (np.dot(Hk[k], rhoj[j]) - np.dot(rhoj[j], Hk[k]))
            ipmat = np.dot(lambdaj[j].T.conjugate(), duiyi)
            um[k, j] = -np.trace(ipmat)

    return um
