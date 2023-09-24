# TODO: add test cases
from grape import grape
from qutip import destroy, identity, tensor, expect, mesolve, basis, ket2dm
import numpy as np


def test_grape():
    N = 200
    sm = destroy(2)
    rho_0 = basis(2, 0)
    rho_0 = ket2dm(rho_0).full().astype(np.complex128)
    
    # # desired state is (|0> + exp(-i*pi/4)|1>)/sqrt(2)
    # state_T = (basis(2, 0) + np.exp(-1j * np.pi / 4) * basis(2, 1)) / np.sqrt(2)
    # C = ket2dm(state_T).full()
    
    # desired state is |1>
    C = ket2dm(basis(2, 1)).full()
    
    # desired state is (|0> + np.exp(1j*np.pi/4)*|1>)/sqrt(2)
    C = ket2dm((basis(2, 0) + np.exp(1j*np.pi/4)*basis(2, 1))/ np.sqrt(2)).full()
    
    # H0 is 0, it seems that using sigma_z will lead to a slower precision
    H0 = (sm.dag() * sm - 0.5 * identity(2)).full()
    # H1 is sigma_x
    H1 = (sm + sm.dag()).full()
    # H2 is sigma_y
    H2 = (-1j * (sm - sm.dag())).full()
    
    Hk = [H1, H2]
    T = 1
    # alpha needs to be rather large, otherwise the precision is not good
    # alpha needs to be set depending on the delta_t, because the gradient's calculation 
    # is based on delta_t * alpha
    threshold, u_kj, rho_T = grape(H0, Hk, np.full((2, N), 3), rho_0, C, T, alpha=10, max_iter=10000, fidility=0.9999)
    
    # print(u_kj)

    print("C: ", C)
    assert np.allclose(rho_T[-1], C, atol=0.1) == True, "final state is not close to target state"
    
    print("rho_T:" , rho_T[-1])
    print("threshold: ", threshold)
    
    
def test_grape_jc():
    steps = 100
    # jaynes-cummings model
    N = 3
    g = 0.01
    sm = tensor(destroy(2), identity(N))
    a = tensor(identity(2), destroy(N))
    
    rho_0 = ket2dm(tensor(basis(2, 0), basis(N, 0)))
    
    # Hamiltonian of the system, supposed w_c = w_a = w_d
    H0 = + g * (sm * a.dag() + sm.dag() * a)
    
    # H1 is sigma_x
    H1 = (sm + sm.dag())
    # H2 is sigma_y
    H2 = -1j * (sm - sm.dag())
    # H3 is coherent driving
    H3 = (a + a.dag())
    
    C = ket2dm(tensor(basis(2, 0), basis(N, 1)))
    
    Hk = [H1, H2, H3]
    T = 1
    
    threshold, u_kj, rho_T = grape(H0, Hk, np.full((3, steps), 1), rho_0, C, T, alpha=100, max_iter=10000, fidility=0.999)
    
    
    print("C: ", C)
    print("rho_T:" , rho_T[-1])
    print("threshold: ", threshold)
    
    assert np.allclose(rho_T[-1], C, atol=0.1) == True, "final state is not close to target state"


if __name__ == "__main__":
    # test_grape()
    test_grape_jc()
