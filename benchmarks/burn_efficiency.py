"""Burn efficiency of uniformly pre-heated, freely expanding DT spheres (full physics:
hydro, conduction, radiation, alpha transport).  Reference: the classic burn-up
estimate  Phi = rhoR / (rhoR + H_B), H_B ~ 6-7 g/cm^2 (Fraley et al. 1974;
Atzeni & Meyer-ter-Vehn 2004, Sec. 3.3)."""
from common import *
import numpy as np
from sife import Simulation, Options, Layer, DT_gas, constants as C

def burn_sphere(rhoR, rho=100.0, T0=20e3, N=120, verbose=0):
    R = rhoR / rho
    mat = DT_gas(rho)
    o = Options(alpha_n_mu=12, verbose=verbose, dt_init=1e-15, dt_max=1e-11, snap_dt=1.0, history_every=20)
    sim = Simulation([Layer(mat, R, N, 1.0, T0=T0, rho=rho)], opts=o)
    tdis = 8 * R / np.sqrt(5 / 3 * 2 * T0 * C.eV / (2.5 * C.mu))
    stop = lambda s: s.t > 0.2 * tdis and s.reac_rate < 1e-4 * max(s.hist["reac_rate"] or [1e99])
    sim.run(tdis, stop=stop)
    fb = 1 - sim.yDT.sum() / sim.yDT0.sum() if False else 1 - np.sum(sim.m * sim.yDT) / np.sum(sim.m * sim.yDT0)
    return fb, sim

def run(show=True):
    rhoRs = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    out = []
    for x in rhoRs:
        fb, sim = burn_sphere(x)
        out.append(dict(rhoR=x, burnup=float(fb), phi_HB6=x / (x + 6), phi_HB7=x / (x + 7),
                        alpha_escape=float(sim.E_alpha_esc / max(sim.E_fus / 5, 1e-30)),
                        E_err=float(sim.energy_error()), steps=sim.nstep))
        print(out[-1], flush=True)
    if show:
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        xx = np.linspace(0.1, 12, 200)
        ax.fill_between(xx, xx / (xx + 7), xx / (xx + 6), color="0.8", label=r"$\rho R/(\rho R+H_B)$, $H_B$=6-7")
        ax.plot(rhoRs, [o["burnup"] for o in out], "C3o-", label="SIFE (T$_0$=20 keV, ρ=100 g/cc)")
        ax.set_xlabel(r"$\rho R$ [g/cm$^2$]"); ax.set_ylabel("burn-up fraction"); ax.legend()
        ax.set_title("Burn efficiency of pre-assembled DT spheres")
        fig.tight_layout(); fig.savefig(f"{OUT}/burn_efficiency.png")
    return out

if __name__ == "__main__":
    run()
