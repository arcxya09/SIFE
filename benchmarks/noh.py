"""Spherical Noh implosion (gamma=5/3): exact shock at r = |u0| t / 3,
post-shock density 64 rho0, pre-shock rho0 (1 + |u0| t / r)^2."""
from common import *
import numpy as np
from sife import Simulation, ideal

def run(N=200, show=True):
    mat = ideal("gas", 1.0, 2.5, 1.0)
    R = 1.0; u0 = 1.0e7; t_end = 0.6 / u0  # shock at 0.2 cm
    r = np.linspace(0, R, N + 1)
    u = -u0 * np.ones(N + 1); u[0] = 0.0
    sim = Simulation(r=r, mat_idx=np.zeros(N, int), materials=[mat], rho=np.ones(N),
                     Te=np.full(N, 1e-3), Ti=np.full(N, 1e-3), u=u,
                     opts=hydro_only(C1=0.1, C2=1.0, cfl=0.25, snap_dt=1.0))
    sim.run(t_end)
    rc = sim.rc
    rs = u0 * t_end / 3
    rho_ex = np.where(rc < rs, 64.0, (1 + u0 * t_end / rc) ** 2)
    # L1 error away from the origin (wall heating region excluded)
    mask = rc > 0.05
    L1 = np.mean(np.abs(sim.rho[mask] - rho_ex[mask]) / rho_ex[mask])
    post = sim.rho[(rc > 0.08) & (rc < 0.17)].mean()
    # shock position: where rho crosses (64+16)/2
    i = np.where(sim.rho > 40)[0][-1]
    if show:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        rr = np.linspace(1e-3, R, 2000)
        ax[0].plot(rr, np.where(rr < rs, 64.0, (1 + u0 * t_end / rr) ** 2), "k-", lw=1, label="exact")
        ax[0].plot(rc, sim.rho, "C3.", ms=3, label=f"SIFE N={N}")
        ax[0].set_xlabel("r [cm]"); ax[0].set_ylabel(r"$\rho$ [g/cc]"); ax[0].legend(); ax[0].set_xlim(0, 0.5)
        ax[1].plot(rr, np.where(rr < rs, 0.0, -u0), "k-", lw=1)
        ax[1].plot(sim.r, sim.u, "C0.", ms=3); ax[1].set_xlim(0, 0.5)
        ax[1].set_xlabel("r [cm]"); ax[1].set_ylabel("u [cm/s]")
        fig.suptitle(f"Noh spherical implosion  t={t_end*1e9:.0f} ns")
        fig.tight_layout(); fig.savefig(f"{OUT}/noh.png")
    res = dict(N=N, plateau_rho=post, exact_plateau=64.0, shock_r=sim.r[i + 1], exact_shock_r=rs,
               L1_rel=L1, energy_err=sim.energy_error(), steps=sim.nstep)
    return res

if __name__ == "__main__":
    print(run())
