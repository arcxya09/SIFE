"""2D (z, R) spherical Noh implosion on a polar Lagrangian mesh (gamma = 5/3).
Exact: shock at r = |u0| t / 3, post-shock density 64.  Tests convergent
strong shocks and symmetry preservation of the 2D scheme (axis included)."""
from common import *
import numpy as np
from sife import ideal, Layer
from sife.simulation2d import Simulation2D, Options2D
from sife.simulation_ale import SimulationALE


def run(N=100, Nth=16, show=True, cls=Simulation2D):
    mat = ideal("gas", 1.0, 2.5, 1.0)
    u0 = 1.0e7; t_end = 0.6 / u0
    o = Options2D(degeneracy=False, cond_e=False, cond_i=False, exchange=False, radiation=False,
                  burn=False, verbose=0, dt_max=1e-6, history_every=50, C1=0.1, C2=1.0, cfl=0.25,
                  dt_init=1e-13, snap_dt=1.0)
    sim = cls([Layer(mat, 1.0, N, T0=1e-3)], n_theta=Nth, half=True, opts=o)
    if hasattr(sim, "vr"):
        sim.vr[:] = -u0
    else:
        sim.ur[1:] = -u0
    sim.E0 = sim.total_energy()
    sim.run(t_end)
    g = sim.geometry()
    rc = g["rc"]
    rs = u0 * t_end / 3
    post = np.array([sim.rho[(rc[:, j] > 0.08) & (rc[:, j] < 0.17), j].mean() for j in range(sim.Nj)])
    shock = np.array([rc[np.where(sim.rho[:, j] > 40)[0][-1], j] for j in range(sim.Nj)])
    if show:
        fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
        from matplotlib.collections import PolyCollection
        from sife.animate import _quads
        q = _quads(sim.z, sim.R)[:, :, ::-1]
        pc = PolyCollection(q, array=sim.rho.ravel(), cmap="viridis", edgecolors="k", linewidths=0.1)
        ax[0].add_collection(pc); ax[0].set_aspect("equal"); ax[0].set_xlim(0, 0.4); ax[0].set_ylim(0, 0.4)
        ax[0].add_patch(plt.Circle((0, 0), rs, fill=False, color="r", lw=1))
        fig.colorbar(pc, ax=ax[0], label="density"); ax[0].set_xlabel("R [cm]"); ax[0].set_ylabel("z [cm]")
        ax[0].set_title("density; exact shock in red")
        rr = np.linspace(1e-3, 0.5, 500)
        ax[1].plot(rr, np.where(rr < rs, 64.0, (1 + u0 * t_end / rr) ** 2), "k-", lw=1, label="exact")
        for j in range(sim.Nj):
            ax[1].plot(rc[:, j], sim.rho[:, j], ".", ms=2, color=plt.cm.viridis(j / sim.Nj))
        ax[1].set_xlim(0, 0.5); ax[1].set_xlabel("r [cm]"); ax[1].set_ylabel("density")
        ax[1].set_title("all angular columns"); ax[1].legend()
        fig.suptitle("2D Noh spherical implosion"); fig.tight_layout(); fig.savefig(f"{OUT}/noh2d.png")
    return dict(exact_shock_r=rs, shock_r_mean=float(shock.mean()), shock_r_spread=float(np.ptp(shock)),
                plateau_mean=float(post.mean()), plateau_spread=float(np.ptp(post)), exact_plateau=64.0,
                energy_err=float(sim.energy_error()), steps=sim.nstep)


if __name__ == "__main__":
    print(run())
