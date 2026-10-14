"""Decay of an l = 2 temperature mode in an insulated sphere (linear conduction).

T = T0 + eps j_2(k r) P_2(cos theta), j_2'(k R) = 0 (no flux through the wall),
decays as exp(-chi k^2 t), chi = kappa / (rho c_v).  Exercises both the radial
and the angular implicit sweeps of the 2D diffusion solver."""
from common import *
import numpy as np
from scipy.special import spherical_jn
from sife import ideal, Layer, constants as C
from sife.simulation2d import Simulation2D, Options2D, legendre

X2 = 3.342093657


def run(N=60, Nth=24, show=True):
    mat = ideal("gas", 1.0, 2.5, 1.0)
    R = 1.0; T0 = 100.0; eps = 1.0; kappa0 = 1e18
    o = Options2D(degeneracy=False, hydro=False, cond_e=True, cond_i=False, exchange=False, radiation=False,
                  burn=False, verbose=0, history_every=10 ** 9, snap_dt=1.0, kappa_mode=1, kappa0=kappa0,
                  kappa_n=0.0, n_picard=1, dT_rel=1.0, dV_rel=1.0)
    sim = Simulation2D([Layer(mat, R, N, T0=T0)], n_theta=Nth, half=True, opts=o)
    g = sim.geometry()
    k = X2 / R
    shape = spherical_jn(2, k * g["rc"]) * legendre(2, np.cos(sim.theta_c))[None, :]
    sim.Te = T0 + eps * shape / np.abs(shape).max()
    sim.ee = sim.Cve * sim.Te  # ideal (non-degenerate) electrons: e = Cv T
    sim._eos(); sim.E0 = sim.total_energy()
    Cv = float(sim.Cve.mean()) * 1.0   # erg/g/eV
    chi = kappa0 / (1.0 * Cv)
    tau = 1.0 / (chi * k * k)
    sim.o.dt_max = tau / 200; sim.dt = tau / 200; sim.o.dt_init = tau / 200
    V = g["V"]
    proj = lambda: float(np.sum((sim.Te - T0) * shape * V) / np.sum(shape ** 2 * V))
    a0 = proj()
    ts, amps = [0.0], [1.0]
    for n in range(1, 11):
        sim.run(n * 0.2 * tau)
        ts.append(sim.t); amps.append(proj() / a0)
    ts = np.array(ts); amps = np.array(amps)
    rate = -np.polyfit(ts[1:], np.log(amps[1:]), 1)[0]
    if show:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.semilogy(ts / tau, amps, "C0o", label="SIFE-2D")
        ax.semilogy(ts / tau, np.exp(-ts / tau), "k-", lw=0.8, label="exact exp(-chi k^2 t)")
        ax.set_xlabel("t / tau"); ax.set_ylabel("l = 2 amplitude"); ax.legend()
        ax.set_title("l = 2 thermal mode decay in an insulated sphere")
        fig.tight_layout(); fig.savefig(f"{OUT}/conduction2d.png")
    return dict(rate_exact=1.0 / tau, rate_sim=float(rate), rel_err=float(abs(rate * tau - 1.0)),
                energy_err=float(sim.energy_error()))


if __name__ == "__main__":
    print(run())
