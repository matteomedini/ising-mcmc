"""
Task 3: statistical properties of the 2D Ising model and the critical
temperature.

This script does three things:
  1. estimates the critical temperature from the average absolute
     magnetization <|m|> and fits the exponent beta_tilde below Tc
  2. estimates Tc from the susceptibility peak and fits gamma above Tc
  3. studies how the cluster size distribution changes with temperature

The heavy MCMC runs are cached in the data/ folder, so re-running the script
only re-makes the plots unless you pass --recompute on the command line.

Run it with:  python3 Ising_properties.py
"""

import os
import sys
import time
import functools

import numpy as np
from matplotlib import pyplot as plt

import Ising
import mcmc
import Clusters_Ising

print = functools.partial(print, flush=True)

FIG_DIR = "figures"
DATA_DIR = "data"

# exact value for the infinite 2D Ising model, used only for comparison
TC_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))


def ensure_dirs():
    for d in (FIG_DIR, DATA_DIR):
        if not os.path.isdir(d):
            os.makedirs(d)


# ---------------------------------------------------------------------------
# 1 and 2: magnetization and susceptibility
# ---------------------------------------------------------------------------

def beta_grid():
    """Inverse temperatures in the range asked by the pdf (0.1 to 1.0), with a
    finer spacing and a few extra points around beta_c so that the power law
    fits have enough data on each side of the transition."""
    grid = list(np.arange(0.10, 1.0001, 0.05))
    grid += [0.42, 0.44, 0.46, 0.48]
    grid = sorted(set(np.round(grid, 4)))
    return np.array(grid)


def compute_mag_susc(N=32, recompute=False):
    path = os.path.join(DATA_DIR, "mag_susc.npz")
    if os.path.isfile(path) and not recompute:
        print("loading cached", path)
        d = np.load(path)
        return d["betas"], d["mabs"], d["mabs_err"], d["chi"], d["chi_err"], int(d["N"])

    betas = beta_grid()
    n_spins = N * N
    n_seeds = 3
    samples_per_seed = 70

    mabs = np.empty_like(betas)
    mabs_err = np.empty_like(betas)
    chi = np.empty_like(betas)
    chi_err = np.empty_like(betas)

    print("running magnetization / susceptibility sweep, N =", N)
    t0 = time.time()
    for k, beta in enumerate(betas):
        # close to beta_c the samples decorrelate slowly, so we wait longer
        # and burn longer there. wait and burn-in are measured in lattice sweeps.
        near = abs(beta - 0.44) < 0.12
        wait = (4 if near else 2) * n_spins
        burn = (250 if near else 120) * n_spins

        # We use a cold (ordered) start. In the ferromagnetic phase a random
        # start gets stuck in striped metastable states, which badly biases
        # <|m|>. Starting ordered measures the spontaneous magnetization, and
        # above Tc the chain quickly forgets the ordered start anyway.
        # Several independent chains are pooled together for better statistics.
        m_all = []
        for sd in range(n_seeds):
            problem = Ising.Ising(N, seed=1000 + sd)
            snaps = mcmc.mcmc(problem, samples=samples_per_seed, wait=wait,
                              burn_in=burn, beta=beta, seed=1000 + sd,
                              start="up")
            m_all.extend(s.mean() for s in snaps)

        m = np.array(m_all)
        absm = np.abs(m)

        mabs[k] = absm.mean()
        mabs_err[k] = absm.std(ddof=1) / np.sqrt(len(absm))

        # finite size susceptibility with the <|m|> convention (see pdf footnote)
        var_m = (m ** 2).mean() - absm.mean() ** 2
        chi[k] = beta * n_spins * var_m
        # rough error on chi from a small bootstrap over the samples
        chi_err[k] = bootstrap_chi_err(m, beta, n_spins)

        print("  beta=%.3f  T=%.3f  <|m|>=%.3f  chi=%.2f" %
              (beta, 1.0 / beta, mabs[k], chi[k]))

    print("sweep done in %.1f s" % (time.time() - t0))
    np.savez(path, betas=betas, mabs=mabs, mabs_err=mabs_err,
             chi=chi, chi_err=chi_err, N=N)
    return betas, mabs, mabs_err, chi, chi_err, N


def bootstrap_chi_err(m, beta, n_spins, n_boot=200):
    rng = np.random.RandomState(0)
    n = len(m)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        mb = m[idx]
        var_m = (mb ** 2).mean() - np.abs(mb).mean() ** 2
        vals[b] = beta * n_spins * var_m
    return vals.std()


def linear_fit(x, y):
    """Least squares straight line, returns slope, intercept and the fitted y."""
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept, slope * x + intercept


def analyse_mag_susc(betas, mabs, mabs_err, chi, chi_err, N):
    T = 1.0 / betas
    order = np.argsort(T)
    T = T[order]
    mabs = mabs[order]; mabs_err = mabs_err[order]
    chi = chi[order]; chi_err = chi_err[order]

    # Tc from the susceptibility peak (clearest signal on a finite lattice)
    tc_chi = T[np.argmax(chi)]
    # Tc from magnetization: highest T where <|m|> is still clearly above its
    # finite size floor. We use a small threshold based on the high-T values.
    floor = mabs[T > 4.0].mean() + mabs_err[T > 4.0].mean()
    above = T[mabs > 3 * floor]
    tc_mag = above.max() if len(above) else tc_chi

    print("Tc from susceptibility peak: %.3f" % tc_chi)
    print("Tc from magnetization:       %.3f" % tc_mag)
    print("Tc exact (infinite lattice): %.3f" % TC_EXACT)

    tc_used = tc_chi

    # --- magnetization curve ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(T, mabs, yerr=mabs_err, marker="o", capsize=3)
    ax.axvline(TC_EXACT, color="gray", ls="--", label="Tc exact = %.2f" % TC_EXACT)
    ax.axvline(tc_chi, color="red", ls=":", label="Tc (chi peak) = %.2f" % tc_chi)
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\langle |m| \rangle$")
    ax.set_title("Average absolute magnetization, N = %d" % N)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "magnetization.png"), dpi=140)
    plt.close(fig)

    # --- susceptibility curve ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(T, chi, yerr=chi_err, marker="o", capsize=3, color="C1")
    ax.axvline(TC_EXACT, color="gray", ls="--", label="Tc exact = %.2f" % TC_EXACT)
    ax.axvline(tc_chi, color="red", ls=":", label="Tc (chi peak) = %.2f" % tc_chi)
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\chi(T)$")
    ax.set_title("Susceptibility, N = %d" % N)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "susceptibility.png"), dpi=140)
    plt.close(fig)

    # --- power law fit for the magnetization below Tc ---
    # We keep only an intermediate window in the reduced temperature: points
    # too close to Tc are rounded by the finite lattice, and points deep in the
    # ferromagnetic phase are saturated at |m| ~ 1, so both spoil the slope.
    red_t_all = (tc_used - T) / tc_used
    mask = (T < tc_used) & (mabs > 0) & (red_t_all > 0.02) & (red_t_all < 0.30)
    x = np.log(red_t_all[mask])
    y = np.log(mabs[mask])
    beta_slope, b0, yfit = linear_fit(x, y)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, "o", label="data (T < Tc)")
    ax.plot(x, yfit, "-", label="fit slope = %.3f" % beta_slope)
    ax.set_xlabel(r"$\log[(T_c - T)/T_c]$")
    ax.set_ylabel(r"$\log \langle |m| \rangle$")
    ax.set_title(r"Magnetization exponent, expected $\tilde\beta = 1/8 = 0.125$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fit_beta.png"), dpi=140)
    plt.close(fig)
    print("beta_tilde estimate (slope): %.3f   (expected 0.125)" % beta_slope)

    # --- power law fit for the susceptibility above Tc ---
    # Same idea: drop the rounded points right at Tc and the far points that
    # are no longer in the asymptotic critical regime.
    red_t_all = (T - tc_used) / tc_used
    mask = (T > tc_used) & (chi > 0) & (red_t_all > 0.08) & (red_t_all < 1.0)
    x = np.log(red_t_all[mask])
    y = np.log(chi[mask])
    g_slope, g0, yfit = linear_fit(x, y)
    gamma = -g_slope

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, "o", color="C1", label="data (T > Tc)")
    ax.plot(x, yfit, "-", color="C3", label="fit slope = %.3f" % g_slope)
    ax.set_xlabel(r"$\log[(T - T_c)/T_c]$")
    ax.set_ylabel(r"$\log \chi(T)$")
    ax.set_title(r"Susceptibility exponent, expected $\gamma = 7/4 = 1.75$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fit_gamma.png"), dpi=140)
    plt.close(fig)
    print("gamma estimate (-slope): %.3f   (expected 1.75)" % gamma)

    return tc_chi, tc_mag, beta_slope, gamma


# ---------------------------------------------------------------------------
# 3: cluster size distribution
# ---------------------------------------------------------------------------

def compute_clusters(recompute=False):
    path = os.path.join(DATA_DIR, "clusters.npz")
    temps = [0.5, 2.5, 5.0]
    sizes = [10, 50, 100, 200]

    if os.path.isfile(path) and not recompute:
        print("loading cached", path)
        d = np.load(path, allow_pickle=True)
        return d["temps"], d["sizes"], d["results"].item()

    results = {}
    print("running cluster sampling")
    t0 = time.time()
    for N in sizes:
        n_spins = N * N
        for T in temps:
            problem = Ising.Ising(N, seed=999)
            burn = 150 * n_spins
            # cold start below Tc to avoid the striped metastable states, hot
            # start at and above Tc where the chain equilibrates quickly
            start = "up" if T < 2.0 else "random"
            snaps = mcmc.mcmc(problem, samples=1, wait=n_spins, burn_in=burn,
                              beta=1.0 / T, seed=999, start=start)
            cs = Clusters_Ising.find_clusters(snaps[0])
            results[(N, T)] = np.array(cs)
            print("  N=%3d  T=%.1f  clusters=%5d  largest=%5d" %
                  (N, T, len(cs), max(cs)))
    print("clusters done in %.1f s" % (time.time() - t0))

    np.savez(path, temps=np.array(temps), sizes=np.array(sizes),
             results=results)
    return np.array(temps), np.array(sizes), results


def plot_clusters(temps, sizes, results):
    fig, axes = plt.subplots(1, len(temps), figsize=(12, 4))
    for ax, T in zip(axes, temps):
        for N in sizes:
            cs = results[(int(N), float(T))]
            # log spaced bins so we can see the heavy tail near Tc
            bins = np.logspace(0, np.log10(max(cs.max(), 2)), 25)
            hist, edges = np.histogram(cs, bins=bins)
            centers = np.sqrt(edges[:-1] * edges[1:])
            nz = hist > 0
            ax.plot(centers[nz], hist[nz], marker="o", ms=4, label="N = %d" % N)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("T = %.1f" % T)
        ax.set_xlabel("cluster size")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("count")
    fig.suptitle("Cluster size distribution at different temperatures")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "clusters.png"), dpi=140)
    plt.close(fig)
    print("saved", os.path.join(FIG_DIR, "clusters.png"))


def main():
    recompute = "--recompute" in sys.argv
    ensure_dirs()

    betas, mabs, mabs_err, chi, chi_err, N = compute_mag_susc(recompute=recompute)
    analyse_mag_susc(betas, mabs, mabs_err, chi, chi_err, N)

    temps, sizes, results = compute_clusters(recompute=recompute)
    plot_clusters(temps, sizes, results)


if __name__ == "__main__":
    main()
