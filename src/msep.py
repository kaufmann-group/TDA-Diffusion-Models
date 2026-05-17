"""
class creates multi species exclusion process.
"""

import numpy as np
import numba as nb
from itertools import combinations
import matplotlib.pyplot as plt

"""
assign seed inside no just in time compiled function. 
"""
@nb.njit
def _jit_random_seed(seed):
    np.random.seed(seed)

@nb.njit
def _jit_shuffle(chain):
    n = len(chain)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(i + 1)
        chain[i], chain[j] = chain[j], chain[i]

@nb.njit
def jit_update(state, rates_matrix, max_rate):
    L = len(state)

    j = np.random.randint(0, L)
    k = j + 1

    if k == L:
        k = 0

    alpha = state[j]
    beta = state[k]

    rate = rates_matrix[alpha, beta]

    if np.random.random() < rate / max_rate:
        state[j] = beta
        state[k] = alpha

"""
monte carlo simulation with no just in time compiling.  
"""
@nb.njit
def _jit_simulate_final(chain, rates_matrix, max_rate, steps):
    length = len(chain)
    current_chain = chain.copy()

    for step in range(steps):
        i = np.random.randint(length)
        j = (i + 1) % length

        species_i = current_chain[i]
        species_j = current_chain[j]

        rate = rates_matrix[species_i, species_j]

        if np.random.random() < rate / max_rate:
            current_chain[i] = species_j
            current_chain[j] = species_i

    return current_chain

@nb.njit
def _jit_build_chain(density, length):
    dimension = len(density)
    chain = np.empty(length, dtype=np.int64)

    idx = 0
    for species in range(dimension):
        count = int(density[species] * length)
        for _ in range(count):
            chain[idx] = species
            idx += 1

    return chain

@nb.njit
def _jit_simulate_history(chain, rates_matrix, max_rate, steps):
    L = len(chain)
    state = chain.copy()

    history = np.empty((steps + 1, L), dtype=chain.dtype)

    for j in range(L):
        history[0, j] = state[j]

    for step in range(1, steps + 1):
        jit_update(state, rates_matrix, max_rate)

        for j in range(L):
            history[step, j] = state[j]

    return history

@nb.njit
def _jit_get_path(chain, proj_vectors):
    length = len(chain)
    path_dim = proj_vectors.shape[1]

    path = np.zeros((length + 1, path_dim), dtype=np.float64)

    for i in range(length):
        species = chain[i]
        for k in range(path_dim):
            path[i + 1, k] = path[i, k] + proj_vectors[species, k]

    return path

class MultiSpeciesExclusionProcess:
    def __init__(self, dimension, density, rates, length, seed=sum("Topological Data Analysis!".encode()), shuffle=True, checkPairwiseBalence=True):
        self.rng = np.random.default_rng(seed=seed)
        
        assert len(density) == dimension, "dimension-density mismatch"
        assert len(rates) == dimension * (dimension - 1), "rate-density mismatch"
        assert length%dimension == 0, "length-dimension mismatch"
        assert sum(density) == 1.0, "density not normalized"
        assert all(rate >= 0 for rate in rates.values()), "rates must be nonnegative"
        
        assert not checkPairwiseBalence or MultiSpeciesExclusionProcess.check_pairwise_balance(rates, dimension), "pairwise balance not imposed"

        self.dimension = dimension
        self.density = np.asarray(density, dtype=np.float64)
        self.rates = rates
        self.length = length
        self.seed = seed

        self.rates_matrix = np.zeros((dimension, dimension), dtype=np.float64)
        for (i, j), rate in rates.items():
            self.rates_matrix[i, j] = rate

        self.max_rate = np.max(self.rates_matrix)
        if self.max_rate == 0.0:
            self.max_rate = 1.0

        self.proj_vectors = self.get_projected_vectors()

        _jit_random_seed(seed)
        self.chain = _jit_build_chain(self.density, self.length)

        if shuffle:
            _jit_shuffle(self.chain)

    """
    note that this ensures the uniform distribution over all particle configurations 
    are an equilibrium/stationary distribution. 
    """
    @staticmethod
    def check_pairwise_balance(rates, dimension):
        species = range(dimension)

        for alpha, beta, gamma in combinations(species, 3):
            lhs = (rates[(alpha, beta)] + rates[(beta, gamma)] + rates[(gamma, alpha)])
            rhs = (rates[(beta, alpha)] + rates[(gamma, beta)] + rates[(alpha, gamma)])

            if not np.isclose(lhs, rhs):
                print("Balance failed for triple:", (alpha, beta, gamma))
                print("lhs =", lhs)
                print("rhs =", rhs)
                return False

        return True

    """
    get basis vectors of d-1 dimensional hyper-surface parallel to (1 ... 1)
    """
    def get_projected_vectors(self):
        norm_vector = np.ones(self.dimension)
        n_hat = norm_vector / np.linalg.norm(norm_vector)
        I = np.eye(self.dimension)

        # Proj(v) = v - (v . n_hat) * n_hat
        projected_vectors = np.zeros((self.dimension, self.dimension))
        for i in range(self.dimension):
            e_i = I[i]
            projected_vectors[i] = e_i - np.dot(e_i, n_hat) * n_hat

        # Plane basis
        A = np.zeros((self.dimension, self.dimension))
        A[:, 0] = n_hat
        A[:, 1:] = self.rng.standard_normal((self.dimension, self.dimension - 1))

        q, r = np.linalg.qr(A)
        plane_basis = q[:, 1:]

        coords_2d = np.dot(projected_vectors, plane_basis)
        norms = np.linalg.norm(coords_2d, axis=1, keepdims=True)
        normalized_coords = np.divide(coords_2d, norms, out=np.zeros_like(coords_2d), where=norms!=0)

        return normalized_coords

    def simulate(self, steps=100000, store_history=False):
        if store_history:
            history = _jit_simulate_history(self.chain, self.rates_matrix, self.max_rate, steps)
            self.chain = history[-1].copy()
            return history

        self.chain = _jit_simulate_final(self.chain, self.rates_matrix, self.max_rate, steps)
        return self.chain

    def get_path(self):
        return _jit_get_path(self.chain, self.proj_vectors)
    
    def get_chain(self):
        return self.chain
    
    @staticmethod
    def plot_path_2d(path):
        assert len(path[0]) == 2, "can only plot with dimension = 4"

        plt.figure(figsize=(6, 6))
        plt.plot(path[:, 0], path[:, 1], "-o", markersize=2)
        plt.axis("equal")
        plt.xlabel(r"$h_1$")
        plt.ylabel(r"$h_2$")
        plt.title("projected directed polymer path, d = 3")

        plt.savefig("figures/projected_directed_polymer_3d.png", dpi=300)
        plt.show()

    @staticmethod
    def plot_path_3d(path):
        assert len(path[0]) == 3, "can only plot with dimension = 4"

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(path[:, 0], path[:, 1], path[:, 2], "-o", markersize=2)
        ax.set_xlabel(r"$h_1")
        ax.set_ylabel(r"h_2")
        ax.set_zlabel(r"h_3")
        ax.set_title("projected directed polymer path, d = 4")
        ax.set_box_aspect([1, 1, 1])

        plt.savefig("figures/projected_directed_polymer_4d.png", dpi=300)
        plt.show()
