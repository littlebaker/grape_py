# TODO: add test cases
from grape_h_torch import grape_torch
from qutip import destroy, identity, tensor, expect, mesolve, basis, ket2dm
import numpy as np
import torch


torch.set_default_device('cuda:0')

def test_grape_jcmodel():
    steps = 100
    # jaynes-cummings model
    N1 = 4
    N2 = 6
    g = 0.01
    chi = 0.1
    sm = tensor(destroy(N1), identity(N2))
    a = tensor(identity(N1), destroy(N2))

    rho_0 = ket2dm(tensor(basis(N1, 0), basis(N2, 0)))

    # Hamiltonian of the system, supposed w_c = w_a = w_d
    H0 = + g * (sm * a.dag() + sm.dag() * a) + chi * (sm.dag() * sm.dag() * sm * sm)

    # H1 is sigma_x
    H1 = (sm + sm.dag())
    # H2 is sigma_y
    H2 = -1j * (sm - sm.dag())
    # H3 is coherent driving
    H3 = (a + a.dag())
    H4 = (a * a + a.dag() * a.dag())

    # C = (|0>+|4>)/sqrt(2)
    C = ket2dm(tensor(basis(N1, 0), (basis(N2, 4) + basis(N2, 0)) / np.sqrt(2)))

    Hk = [
        H1,
        H2,
        H3,
        # H4
    ]
    T = 1

    threshold, rhoj = grape_torch(H0, Hk, torch.normal(0, 1, (len(Hk), steps)), rho_0, C, T)


    # print("u_kj: ", res.x.reshape(len(Hk), steps))
    print("C: ", C)
    print("rhoj:" , rhoj[-1])
    print("threshold: ", threshold)


    assert torch.allclose(rhoj[-1], torch.tensor(C.full()), atol=0.1) == True, "final state is not close to target state"


if __name__ == "__main__":
    # test_grape_bfgs()
    test_grape_jcmodel()
