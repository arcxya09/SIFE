"""Nonlinear heat conduction from a point source (Zel'dovich-Raizer / Barenblatt
self-similar solution) in a static sphere:  rho cv dT/dt = div(kappa0 T^n grad T).
  T = Tc (1 - r^2/rf^2)^(1/n),   rf = xi1 (a Q^n t)^(1/(3n+2)),  a = kappa0/(rho cv), Q = E/(rho cv)."""
from common import *
import numpy as np
from math import gamma, pi
from scipy.integrate import quad
from sife import Simulation, ideal, constants as C

def exact(r, t, Q, a, n):
    xi1 = ((3 * n + 2) / (2 ** (n - 1) * n * pi ** n)) ** (1 / (3 * n + 2)) * \
          (gamma(2.5 + 1 / n) / (gamma(1 + 1 / n) * gamma(1.5))) ** (n / (3 * n + 2))
    rf = xi1 * (a * Q ** n * t) ** (1 / (3 * n + 2))
    I = quad(lambda x: x * x * (1 - x * x) ** (1 / n), 0, 1)[0]
    Tc = Q / (4 * pi * rf ** 3 * I)
    T = Tc * np.clip(1 - (r / rf) ** 2, 0, None) ** (1 / n)
    return T, rf, Tc

def run(N=200, n=2.5, show=True):
    mat = ideal("gas", 1.0, 2.5, 1.0)
    R = 1.0; E = 1e15
    r = np.linspace(0, R, N + 1)
    Te = np.full(N, 1e-3)
    o = hydro_only(hydro=False, cond_e=True, kappa_mode=1, kappa0=6.5e9, kappa_n=n, dt_init=1e-16,
                   dt_max=1e-11, dT_rel=0.1, n_picard=4, snap_dt=1.0)
    sim = Simulation(r=r, mat_idx=np.zeros(N, int), materials=[mat], rho=np.ones(N), Te=Te, Ti=Te.copy(), opts=o)
    sim.ee[0] += E / sim.m[0]; sim._eos(); sim.E0 = sim.total_energy()
    cv = sim.Cve[1]
    a = o.kappa0 / cv; Q = E / cv
    out = []
    if show:
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for k, t in enumerate([0.1e-9, 0.4e-9, 1.0e-9]):
        sim.run(t)
        Tex, rf, Tc = exact(sim.rc, t, Q, a, n)
        # numerical front: outermost zone with T > 1% Tc
        i = np.where(sim.Te > 0.01 * Tc)[0][-1]
        out.append(dict(t_ns=t * 1e9, rf_exact=rf, rf_sim=sim.r[i + 1], Tc_exact=Tc,
                        Tc_sim=float(sim.Te[0]),
                        L1_rel=float(np.sum(np.abs(sim.Te - Tex) * sim.m) / np.sum(Tex * sim.m))))
        if show:
            rr = np.linspace(0, R, 1000)
            ax.plot(rr, exact(rr, t, Q, a, n)[0], "k-", lw=1)
            ax.plot(sim.rc, sim.Te, f"C{k}.", ms=3, label=f"t={t*1e9:.1f} ns")
    if show:
        ax.set_xlabel("r [cm]"); ax.set_ylabel("T [eV]"); ax.legend(); ax.set_title(f"Nonlinear conduction, n={n} (lines: exact)")
        fig.tight_layout(); fig.savefig(f"{OUT}/conduction.png")
    return dict(results=out, energy_err=sim.energy_error())

if __name__ == "__main__":
    import pprint; pprint.pprint(run())
