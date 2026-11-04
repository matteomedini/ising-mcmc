import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

class Ising:
    def __init__(self, N, seed = None):
        self.N = N

        ## Optionally set up the random number generator state
        if seed is not None:
            np.random.seed(seed)
    
        # s is the initial configuration vector
        # The spins are arranged over a NxN lattice
        s = np.ones((N,N), dtype=int)
        self.s = s
        self.init_config()

        pass

    ## Initialize (or reset) the current configuration
    # ordered=True starts from an all +1 lattice (cold start). This is useful
    # in the ferromagnetic phase, where a random start tends to get stuck in
    # long lived striped states that single spin flips cannot easily undo.
    def init_config(self, ordered=False):
        N = self.N
        if ordered:
            self.s = np.ones((N, N), dtype=int)
        else:
            self.s = np.random.choice([-1,1], size=(N,N))

    ## Definition of the cost function
    # Here you need to complete the function computing the cost as described in the pdf file
    # The cost function depends on the value of the spins and the values of their nearest neighbors
    def cost(self):
        s = self.s
        # With periodic boundary conditions every spin has a right and a down neighbor,
        # so summing only those two directions already covers all bonds without
        # counting any bond twice. np.roll wraps around, which gives us the PBC for free.
        right = np.sum(s * np.roll(s, -1, axis=1))
        down = np.sum(s * np.roll(s, -1, axis=0))
        # The 1/2 in the Hamiltonian cancels the double counting of the i,j sum,
        # which here is already avoided, so the energy is just minus the bond sum.
        return -float(right + down)

    ## Propose a valid random move. 
    def propose_move(self):
        N = self.N
        move = (np.random.choice(N), np.random.choice(N))
        return move
    
    ## Modify the current configuration, accepting the proposed move
    def accept_move(self, move):
        self.s[move] *= -1

    ## Compute the extra cost of the move (new-old, negative means convenient)
    # Here you need complete the compute_delta_cost function as explained in the pdf file
    # Check the constant in front of the delta cost
    def compute_delta_cost(self, move):
        i, j = move
        N = self.N
        s = self.s
        # Only the four nearest neighbors of the flipped spin matter, the rest of
        # the lattice keeps the same energy. The modulo keeps us inside the lattice
        # and takes care of the periodic boundary conditions.
        nb = s[(i + 1) % N, j] + s[(i - 1) % N, j] + s[i, (j + 1) % N] + s[i, (j - 1) % N]
        return 2.0 * s[i, j] * nb

    ## Average spin of the current configuration (magnetization per spin)
    def magnetization(self):
        return float(self.s.mean())
    
    ## Make an entirely independent duplicate of the current object.
    def copy(self):
        return deepcopy(self)
    
    ## The display function is used for having a graphical representation of the configurations in white and black
    def display(self):
        plt.clf()
        plt.imshow(self.s, cmap='gray', vmin=-1, vmax=1)
        plt.show()
        

