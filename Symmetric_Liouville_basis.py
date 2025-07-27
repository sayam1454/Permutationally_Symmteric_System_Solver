from itertools import product
import numpy as np

#Defining the symmetric Liouville qutrit basis 
class SymmetricLiouvilleQutritBasis:
    def __init__(self, N):
        self.N = N
        self.q = 3  # qutrit
        self.d = self.q ** 2  # 9 operator basis elements: |i⟩⟨j|
        self.basis = self._generate_basis()
        self.index_map = {tuple(state): i for i, state in enumerate(self.basis)}
    
    def _generate_basis(self):
        valid_basis = []
        for n in product(range(self.N + 1), repeat=self.d):
            if sum(n) == self.N:
                valid_basis.append(n)
        return valid_basis

    def dim(self):
        return len(self.basis)
    
    def get_index(self, n_tuple):
        return self.index_map.get(tuple(n_tuple),-1)
    
    def show_basis(self, compact=True):
        for i, state in enumerate(self.basis):
            if compact:
                print(f"{i}: {state}")
            else:
                mat = np.array(state).reshape(self.q, self.q)
                print(f"{i}:\n{mat}\n")
