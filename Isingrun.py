# This is an example on how to generate samples from the Ising model

import mcmc
import Ising
import Clusters_Ising

## Generate a problem to solve.
# This generate a Ising instance with a size of the lattice N
N = 10

ising = Ising.Ising(N, seed=42)

## Display initial configuration
# print("Initial configuration:")
# print(ising.s)
# ising.display()

## Sample from the Boltzmann distribution.
# debug_delta_cost can be set to True to double check that the incremental
# energy update stays consistent with a full recomputation of the cost.
s = mcmc.mcmc(ising,
              samples=1, wait=20, burn_in=10000, beta=0.4,
              seed=42, debug_delta_cost=False)

## Display the final configuration
# print("Final configuration:")
# print(s[0])
# ising.display()

print("cost of the sampled configuration:", ising.cost())
print("magnetization:", s[0].mean())

## Analysis of the clusters
# Find the clusters of same sign spins and compute their sizes
res = Clusters_Ising.find_clusters(s[0])
print("number of clusters:", len(res), " largest cluster:", max(res))
