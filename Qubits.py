#Imports
import numpy as np
from itertools import product
import math
from scipy.sparse import kron, csr_matrix
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import qutip.piqs as piqs
from qutip import mesolve, expect

N=10  # Number of qubits

#Creating the number basis class
class SymmetricLiouvilleQubitNumberBasis:
    def __init__(self, N):
        self.N = N
        self.q = 2
        self.d = self.q ** 2
        self.basis = self._generate_basis()
        self.index_map = {tuple(state): i for i, state in enumerate(self.basis)}

    def _generate_basis(self):
        basis= [n for n in product(range(self.N + 1), repeat=self.d) if sum(n) == self.N]
        basis.reverse()
        return basis

    def dim(self):
        return len(self.basis)

    def get_index(self, n_tuple):
        return self.index_map.get(tuple(n_tuple), -1)

    def show_basis(self, compact=True):
        for i, state in enumerate(self.basis):
            if compact:
                print(f"{i}: {state}")
            else:
                mat = np.array(state).reshape(self.q, self.q)
                print(f"{i}:\n{mat}\n")

Numberbasis=SymmetricLiouvilleQubitNumberBasis(N)


#Defining the P_operator

def sigma(x, y, d):
    result = np.zeros((d, d))
    result[x, y] = 1
    return result

def sigma_list(d):
    operators = [[sigma(i, j, d) for j in range(d)] for i in range(d)]  
    return operators

# Constructing P_operators
P_operator = []
for idx, n in enumerate(Numberbasis.basis):
    C_den = np.prod([math.factorial(i) for i in n])
    C = math.factorial(N) / C_den

    # Constructing P operator
    nij = np.array(n).reshape(2, 2)
    sigmaij = sigma_list(2)

    # Tensor product terms of individual sigmas
    sigma_operator = []
    for i in range(2):
        for j in range(2):
            ncount = nij[i, j]
            sigma_tensorprod = csr_matrix(np.array([[1]]) ) 
            for k in range(ncount):
                sigma_tensorprod = kron(sigma_tensorprod, csr_matrix(sigmaij[i][j])) 
            sigma_operator.append(sigma_tensorprod)

    rho_operator = csr_matrix(sigma_operator[0])
    for op in sigma_operator[1:]:
        rho_operator = kron(rho_operator, op)

    # Final P operator
    P_operator.append(C * rho_operator)

#Constructing the density matrix and corresponding P_values

ket0=csr_matrix([0,1])
ket1=csr_matrix([1,0])
bra0=csr_matrix([0,1]).T
bra1=csr_matrix([1,0]).T


rho00= csr_matrix(kron(ket0, bra0))
rho11= csr_matrix(kron(ket1, bra1))
rho10= csr_matrix(kron(ket1, bra0))     
rho01= csr_matrix(kron(ket0, bra1))
for i in range (1,N):
    rho11= csr_matrix(kron(rho11,kron(ket1, bra1)))     
    rho00= csr_matrix(kron(rho00,kron(ket0, bra0)))     
    rho10= csr_matrix(kron(rho10,kron(ket1, bra0)))     
    rho01= csr_matrix(kron(rho01,kron(ket0, bra1)))

rho=1/2*(rho00 + rho11)  # GHZ state
#rho=rho00                             #Ground state               
rho=rho11                              #Excited state


P_values=[]
for i in P_operator:
    P_values.append((i @ rho).diagonal().sum())

P_values.reverse()

#Constructing the Entire Liouvillian

gamma=0.5        #Decay rate
k=0.5            #Pumping rate
gamma_phi=0.5    #Dephasing rate

H=0              #Hamiltonian


#Defining the rate of change of P values
def dP_dt(t,P):
    dP=np.zeros_like(P)

    for idx, n in enumerate(Numberbasis.basis):
        nij=np.array(n).reshape(2,2)


        #For decay
        nij=np.array(n).reshape(2,2)
        nij_copy=nij.copy()
        nij_copy[1,1]+=1
        nij_copy[0,0]-=1

        i=Numberbasis.get_index(nij_copy.flatten())

        #For pumping
        nij=np.array(n).reshape(2,2)
        nij_copy=nij.copy()
        nij_copy[1,1]-=1
        nij_copy[0,0]+=1

        j=Numberbasis.get_index(nij_copy.flatten())


        dP[idx]=-gamma/2*((2*nij[1,1]+nij[1,0]+nij[0,1])*P[idx])-k/2*((2*nij[0,0]+nij[1,0]+nij[0,1])*P[idx])-gamma_phi*(nij[1,0]+nij[0,1])*P[idx]

        if i!=-1:
            dP[idx]+=gamma/2*(2*(nij[1,1]+1)*P[i])
        if j!=-1:
            dP[idx]+=k/2*(2*(nij[0,0]+1)*P[j])

    return dP  

#ODE solving

#Time parameters
t_span = (0, 15)
t_eval = np.linspace(*t_span, 100)

#Solving the full ODE system using RK45 method
sol = solve_ivp(dP_dt, t_span, P_values, t_eval=t_eval,method='RK45')

# Extracting P(t)
Pt = sol.y 


#Defining the observables
#Extracting <J11>
index_list=[]
for idx,n in enumerate(Numberbasis.basis):
    nij=np.array(n).reshape(2,2)
    

    if nij[0,0]+nij[1,1]==N:
        index_list.append(idx)


J11=0*Pt[0]

for i in range(1,len(index_list)):
    J11+=i*Pt[index_list[i]]    #Population of the excited state

Jz=J11-N/2 # Expectation of Jz values


#PIQS Simulation
[jx , jy , jz] = piqs.jspin (N)
piqs_sys = piqs.Dicke(N=N, pumping=k,emission=gamma, dephasing=gamma_phi)
D=piqs_sys.liouvillian()
rho0_piqs = piqs.dicke (N,N/2,N/2)
#rho0_piqs=piqs.ghz(N)

result = mesolve (D , rho0_piqs , t_eval,[])
rhot = result . states
jzt = expect (rhot , jz)
piqs_pop_n11 = [i+N/2 for i in jzt]


#Comparison Plotting
plt.plot(t_eval, piqs_pop_n11, '*', label='QuTiP PIQS')
plt.plot(t_eval, J11, label="From the Code")

plt.xlabel("Time")
plt.ylabel("Population")
plt.title(r"Time evolution of population of the excited state under al three processes")
plt.grid(True)
plt.legend()
plt.show()