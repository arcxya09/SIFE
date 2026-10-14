"""2D (z, R) Sedov-Taylor blast on a polar Lagrangian mesh (gamma = 5/3).
Checks the shock radius against the exact self-similar value along every
angular column, i.e. accuracy AND preservation of spherical symmetry by the
area-weighted 2D scheme (including the axis)."""
from common import *
import numpy as np
from sife import ideal, Layer
from sife.simulation2d import Simulation2D, Options2D

XI0 = 1.1517


def run(N=100, Nth=16, half=True, show=True):
    mat = ideal("gas", 1.0, 2.5, 1.0)
    E = 1.0e16; rho0 = 1.0
    o = Options2D(degeneracy=False, cond_e=False, cond_i=False, exchange=False, radiation=False,
                  burn=False, verbose=0, dt_max=1e-6, history_every=50, C1=0.1, C2=1.0, cfl=0.25,
                  dt_init=1e-15, snap_dt=1.0)
    sim = Simulation2D([Layer(mat, 1.0, N, T0=1e-3)], n_theta=Nth, half=half, opts=o)
    nhot = 2
    sim.ei[:nhot] += (E / sim.sym) / sim.m[:nhot].sum()
    sim._eos(); sim.E0 = sim.total_energy()
    t = 1.96e-9
    sim.run(t)
    g = sim.geometry()
    rc = g["rc"]
    i = np.argmax(sim.rho, axis=0)
    Rs = rc[i, np.arange(sim.Nj)]
    Rex = XI0 * (E * t * t / rho0) ** 0.2
    if show:
        fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
        from matplotlib.collections import PolyCollection
        from sife.animate import _quads
        q = _quads(sim.z, sim.R)[:, :, ::-1]
        pc = PolyCollection(q, array=sim.rho.ravel(), cmap="viridis", edgecolors="k", linewidths=0.1)
        ax[0].add_collection(pc); ax[0].set_aspect("equal"); ax[0].set_xlim(0, 1); ax[0].set_ylim(0 if half else -1, 1)
        ax[0].add_patch(plt.Circle((0, 0), Rex, fill=False, color="r", lw=1))
        fig.colorbar(pc, ax=ax[0], label="density"); ax[0].set_xlabel("R"); ax[0].set_ylabel("z")
        ax[0].set_title("density, exact shock radius in red")
        for j in range(sim.Nj):
            ax[1].plot(rc[:, j] / Rex, sim.rho[:, j], lw=0.6, color=plt.cm.viridis(j / sim.Nj))
        ax[1].axvline(1, color="k", lw=0.8); ax[1].axhline(4, color="k", ls=":", lw=0.8)
        ax[1].set_xlabel("r / R_exact"); ax[1].set_ylabel("density"); ax[1].set_title("all angular columns")
        fig.tight_layout(); fig.savefig(f"{OUT}/sedov2d.png")
    return dict(R_exact=Rex, R_mean=float(Rs.mean()), rel_err=float(abs(Rs.mean() - Rex) / Rex),
                angular_spread=float((Rs.max() - Rs.min()) / Rex), rho_peak=float(sim.rho.max()),
                rho_peak_spread=float(np.ptp(sim.rho.max(axis=0)) / sim.rho.max()),
                energy_err=float(sim.energy_error()), steps=sim.nstep)


if __name__ == "__main__":
    print(run())
