from grape_L_torch import grape_liouvillian_bfgs
from qutip import destroy, identity, tensor, expect, mesolve, basis, ket2dm
import numpy as np
import time
import torch
torch.set_default_device('cuda:0')


def test_grape():
    sm = destroy(2)
    rho_0 = basis(2, 0)
    rho_0 = ket2dm(rho_0).full().astype(np.complex128)
    
    # desired state is (|0> + np.exp(1j*np.pi/4)*|1>)/sqrt(2)
    C = ket2dm((basis(2, 0) + np.exp(1j*np.pi/4)*basis(2, 1))/ np.sqrt(2)).full()
    print(C.conj().T - C)
    # C = ket2dm(basis(2, 1)).full()
    
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
    starttime = time.time()
    res = grape_liouvillian_bfgs(
        np.full((2, 100), 3), 
        rho_0, 
        C, 
        T, 
        H0, 
        Hk, 
        c_ops=[],
        gtol=1e-9,
        atol=1e-6,
        method="bfgs"
    )
    endtime = time.time()
    print(endtime - starttime)
    
    # print(u_kj)

    print(res)
    
    
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
    H4 = (a*a + a.dag()*a.dag())
    
    # C = (|0>+|4>)/sqrt(2)
    C = ket2dm(tensor(basis(N1, 0), (basis(N2, 4)+basis(N2, 0))/np.sqrt(2)))
    
    Hk = [
        H1,
        H2,
        H3,
        # H4
    ]
    T = 1
    
    res = grape_bfgs(H0, Hk, np.random.normal(0, 1, (len(Hk), steps)), rho_0, C, T)
    
    print(res)
    # print("u_kj: ", res.x.reshape(len(Hk), steps))
    # print("C: ", C)
    # print("rho_T:" , rho_T[-1])
    # print("threshold: ", threshold)
    
    # assert np.allclose(rho_T[-1], C, atol=0.1) == True, "final state is not close to target state"



if __name__ == "__main__":
    test_grape()
