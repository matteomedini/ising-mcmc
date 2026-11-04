"""
Task 2: exploring the MCMC parameters of the 2D Ising model.

What this script produces (all saved in the figures/ folder):
  - snapshots of typical configurations at different T and N
  - the energy during the burn-in, to see how many steps we need to equilibrate
  - acceptance rate and cost per spin as a function of T, for the three N

Run it with:  python3 explore_mcmc.py
"""

import os
import time
import functools

import numpy as np
from matplotlib import pyplot as plt

import Ising
import mcmc

# print immediately so we can follow the progress of the long runs
print = functools.partial(print, flush=True)


FIG_DIR = "figures"

# values asked in the assignment
TS = [0.5, 2.5, 5.0]
NS = [10, 25, 100]


def ensure_dir():
    if not os.path.isdir(FIG_DIR):
        os.makedirs(FIG_DIR)


def run_chain_record(problem, beta, sweeps, seed):
    """Run a plain single spin flip chain from a random start and record the
    energy per spin after every sweep (one sweep = N*N proposed flips).

    We keep this loop here instead of inside mcmc.py because for the burn-in
    study we want the full energy trajectory, not only the saved samples.
    """
    np.random.seed(seed)
    problem.init_config()
    N = problem.N
    n_spins = N * N
    E = problem.cost()
    trace = np.empty(sweeps + 1, dtype=float)
    trace[0] = E / n_spins
    for sweep in range(sweeps):
        for _ in range(n_spins):
            move = problem.propose_move()
            dE = problem.compute_delta_cost(move)
            if mcmc.accept(dE, beta):
                problem.accept_move(move)
                E += dE
        trace[sweep + 1] = E / n_spins
    return trace


def make_snapshots():
    """Grid of configuration snapshots, rows are N and columns are T."""
    fig, axes = plt.subplots(len(NS), len(TS), figsize=(8, 8))
    for r, N in enumerate(NS):
        for c, T in enumerate(TS):
            beta = 1.0 / T
            problem = Ising.Ising(N, seed=100 + r * 10 + c)
            # burn-in measured in lattice sweeps (one sweep = N*N flips) so the
            # picture has time to settle. 100 sweeps is enough for a snapshot.
            burn = 100 * N * N
            mcmc.mcmc(problem, samples=1, wait=N * N, burn_in=burn,
                      beta=beta, seed=100 + r * 10 + c)
            ax = axes[r, c]
            # gray_r so that positive spins look black, as stated in the pdf
            ax.imshow(problem.s, cmap="gray_r", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title("T = %.1f" % T)
            if c == 0:
                ax.set_ylabel("N = %d" % N, fontsize=11)
    fig.suptitle("Ising configurations at different T and N")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "snapshots.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


def make_burnin_study():
    """Energy per spin during the burn-in for several T, at two lattice sizes."""
    burn_temps = [0.5, 2.27, 5.0]
    sizes = [25, 50]
    fig, axes = plt.subplots(1, len(sizes), figsize=(10, 4), sharey=True)
    for ax, N in zip(axes, sizes):
        sweeps = 300
        for k, T in enumerate(burn_temps):
            problem = Ising.Ising(N, seed=7)
            trace = run_chain_record(problem, 1.0 / T, sweeps, seed=7 + k)
            ax.plot(np.arange(sweeps + 1), trace, label="T = %.2f" % T)
        ax.set_title("N = %d" % N)
        ax.set_xlabel("burn-in sweeps")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("energy per spin")
    axes[0].legend()
    fig.suptitle("Equilibration of the energy during burn-in")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "burnin.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


def make_acceptance_and_cost():
    """Acceptance rate and cost per spin versus T, one curve per N.

    For each (N, T) we use a burn-in proportional to the number of spins so
    that the chain has equilibrated, then average over a handful of samples.
    We start from an ordered configuration so that the low T runs settle on the
    ground state instead of getting stuck in a striped metastable state, which
    would otherwise leave the energy above its equilibrium value.
    """
    temps = np.linspace(0.5, 5.0, 10)
    fig, (ax_acc, ax_cost) = plt.subplots(1, 2, figsize=(11, 4))
    for N in NS:
        acc = np.empty_like(temps)
        cost_per_spin = np.empty_like(temps)
        for idx, T in enumerate(temps):
            problem = Ising.Ising(N, seed=2024)
            burn = 80 * N * N
            snaps, stats = mcmc.mcmc(problem, samples=12, wait=N * N,
                                     burn_in=burn, beta=1.0 / T, seed=2024,
                                     return_stats=True, start="up")
            acc[idx] = stats["acc_meas"]
            cost_per_spin[idx] = stats["energies"].mean() / (N * N)
        ax_acc.plot(temps, acc, marker="o", label="N = %d" % N)
        ax_cost.plot(temps, cost_per_spin, marker="o", label="N = %d" % N)
    for ax in (ax_acc, ax_cost):
        ax.axvline(2.27, color="gray", ls="--", alpha=0.6)
        ax.set_xlabel("T")
        ax.grid(True, alpha=0.3)
        ax.legend()
    ax_acc.set_ylabel("measurement acceptance rate")
    ax_acc.set_title("Acceptance rate vs T")
    ax_cost.set_ylabel("energy per spin")
    ax_cost.set_title("Cost per spin vs T")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "acceptance_cost.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


def main():
    ensure_dir()
    t0 = time.time()
    make_snapshots()
    make_burnin_study()
    make_acceptance_and_cost()
    print("done in %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
