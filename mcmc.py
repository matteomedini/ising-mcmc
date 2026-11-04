import numpy as np

## Stochastically determine whether to acccept a move according to the
## Metropolis rule (valid for symmetric proposals)
def accept(delta_c, beta):
    ## If the cost doesn't increase, we always accept
    if delta_c <= 0:
        return True
    ## If the cost increases and beta is infinite, we always reject
    if beta == np.inf:
        return False
    ## Otherwise the probability is going to be somwhere between 0 and 1
    p = np.exp(-beta * delta_c)
    ## Returns True with probability p
    return np.random.rand() < p

def mcmc(probl,
    burn_in = 2000, samples = 2000, wait = 10, beta=0.5,         
    seed = None, debug_delta_cost = False, return_stats = False,
    start = "random"):
    
    ## Optionally set up the random number generator state
    if seed is not None:
        np.random.seed(seed)

    # init. start="up" gives a cold (ordered) start, anything else keeps the
    # original random start.
    probl.init_config(ordered=(start == "up"))
    E = probl.cost()

    # There are two acceptances that can be considered for traking the sampling efficiency:
    # 1) acc_burn: acceptance during the burn-in phase
    # 2) acc_meas: acceptance during the measurement phase
    # Both are reported at the end of the function.

    # burn-in phase of the Markov chain
    acc_burn = 0
    for _ in range(burn_in):
        move = probl.propose_move()
        dE = probl.compute_delta_cost(move)
        if debug_delta_cost:
            chk = probl.copy(); 
            chk.accept_move(move)
            assert abs(E + dE - chk.cost()) < 1e-10
        if accept(dE, beta):
            probl.accept_move(move); 
            E += dE; 
            acc_burn += 1

    acc_burn /= burn_in

    # Measurement phase with correct waiting time between samples
    snaps = []
    energies = []
    acc_meas = 0
    for t in range(samples):
        for _ in range(wait):
            move = probl.propose_move()
            dE = probl.compute_delta_cost(move)
            if accept(dE, beta):
                probl.accept_move(move); 
                E += dE; 
                acc_meas += 1
        s = np.asarray(probl.s)
        snaps.append(s.copy())
        energies.append(E)
    acc_meas /= samples * wait

    # print(f"[MCMC] beta={beta:.6f}  acc_burn={acc_burn:.3f}  acc_meas={acc_meas:.3f}")
    # By default we keep the original behaviour and just return the snapshots.
    # When studying the burn-in and the acceptance (task 2) it is handy to also
    # get back the acceptance rates and the energy of each saved configuration.
    if return_stats:
        return snaps, {"acc_burn": acc_burn,
                       "acc_meas": acc_meas,
                       "energies": np.asarray(energies, dtype=float)}
    return snaps
