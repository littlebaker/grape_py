# TODO: add test cases
from grape import grape
from qutip import destroy, identity, tensor, expect, mesolve, basis, ket2dm
import numpy as np


def test_grape():
    N = 100
    sm = destroy(2)
    rho_0 = basis(2, 0)
    rho_0 = ket2dm(rho_0)
    
    # # desired state is (|0> + exp(-i*pi/4)|1>)/sqrt(2)
    # state_T = (basis(2, 0) + np.exp(-1j * np.pi / 4) * basis(2, 1)) / np.sqrt(2)
    # C = ket2dm(state_T).full()
    
    # desired state is |1>
    C = ket2dm(basis(2, 1)).full()
    
    # H0 is sigma_z
    H0 = np.array([[1, 0], [0, -1]])
    # H1 is sigma_x
    H1 = (sm + sm.dag()).full()
    # H2 is sigma_y
    H2 = (-1j * (sm - sm.dag())).full()
    
    Hk = [H1]
    T = 1
    threshold, u_kj, rho_T = grape(H0, Hk, np.zeros((1, N)), rho_0.full(), C, T, alpha=0.1)
    
    # print(u_kj)
    print(rho_T[-1])
    print(C)
    print("threshold: ", threshold)
    assert np.allclose(rho_T[-1], C, atol=1e-2) == True, "final state is not close to target state"
    
    print("rho_T:" , rho_T[-1])
    print("threshold: ", threshold)
    


if __name__ == "__main__":
    test_grape()
