# TODO: add test cases
from grape import grape
from qutip import destroy, identity, tensor, expect, mesolve, basis, ket2dm
import numpy as np


def test_grape():
    sm = destroy(2)
    rho_0 = basis(2, 0)
    
    # desired state is (|0> + exp(-i*pi/4)|1>)/sqrt(2)
    state_T = (basis(2, 0) + np.exp(-1j * np.pi / 4) * basis(2, 1)) / np.sqrt(2)
    C = ket2dm(state_T)
    
    H0 = sm.dag() * sm
    # H1 is sigma_x
    H1 = sm + sm.dag()
    # H2 is sigma_y
    H2 = -1j * (sm - sm.dag())
    
    Hk = [H1, H2]
    T = 1
    threshold, u_kj, rho_T = grape(H0, Hk, np.zeros((2, 50)), rho_0, C, T)
    
    print(u_kj)
    assert np.allclose(rho_T, C, atol=1e-2) == True, "final state is not close to target state"
    


if __name__ == "__main__":
    test_grape()