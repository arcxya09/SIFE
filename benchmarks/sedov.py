"""Spherical Sedov-Taylor point blast, gamma = 5/3.
Exact shock radius R = xi0 (E t^2 / rho)^(1/5), xi0 = 1.1517 (alpha = 0.4936)."""
from common import *
import numpy as np
from sife import Simulation, ideal

XI0 = 1.1517

def run(N=200, show=True):
    mat = ideal("gas", 1.0, 2.5, 1.0)
    R = 1.0; E = 1.0e16; rho0 = 1.0
    r = np.linspace(0, R, N + 1)
    Te = np.full(N, 1e-3); Ti = Te.copy()
    sim = Simulation(r=r, mat_idx=np.zeros(N, int), materials=[mat], rho=np.full(N, rho0), Te=Te, Ti=Ti,
                     opts=hydro_only(C1=0.1, C2=1.0, cfl=0.25, dt_init=1e-15, snap_dt=1.0))
    nhot = 2
    sim.ei[:nhot] += E / sim.m[:nhot].sum()
    sim._eos(); sim.E0 = sim.total_energy()
    times = [0.6e-9, 1.2e-9, 1.96e-9]
    Rs, Rex = [], []
    profiles = []
    for t in times:
        sim.run(t)
        i = np.argmax(sim.rho)
        Rs.append(sim.rc[i]); Rex.append(XI0 * (E * t * t / rho0) ** 0.2)
        profiles.append((t, sim.rc.copy(), sim.rho.copy(), sim.P.copy()))
    if show:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        for t, rc, rho, P in profiles:
            Rt = XI0 * (E * t * t / rho0) ** 0.2
            ax[0].plot(rc / Rt, rho / rho0, ".-", ms=2, lw=0.6, label=f"t={t*1e9:.2f} ns")
            P2 = 2.0 / (5 / 3 + 1) * rho0 * (0.4 * Rt / t) ** 2
            ax[1].plot(rc / Rt, P / P2, ".-", ms=2, lw=0.6)
        ax[0].axvline(1, color="k", lw=0.8); ax[0].axhline(4, color="k", ls=":", lw=0.8)
        ax[0].set_xlabel(r"$r/R_{exact}(t)$"); ax[0].set_ylabel(r"$\rho/\rho_0$"); ax[0].legend()
        ax[1].axvline(1, color="k", lw=0.8); ax[1].axhline(1, color="k", ls=":", lw=0.8)
        ax[1].set_xlabel(r"$r/R_{exact}(t)$"); ax[1].set_ylabel(r"$P/P_{shock,exact}$")
        fig.suptitle("Sedov-Taylor blast (self-similar scaling; exact shock at 1, jump 4)")
        fig.tight_layout(); fig.savefig(f"{OUT}/sedov.png")
    return dict(times_ns=[t * 1e9 for t in times], R_sim=Rs, R_exact=Rex,
                rel_err=[abs(a - b) / b for a, b in zip(Rs, Rex)], rho_peak=float(profiles[-1][2].max()),
                energy_err=sim.energy_error())

if __name__ == "__main__":
    print(run())
