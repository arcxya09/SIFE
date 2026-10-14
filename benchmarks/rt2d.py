"""Rayleigh-Taylor growth at a spherical interface (2D, linear phase).

A heavy shell (rho2 = 3) sits on a light core (rho1 = 1) at R = 1 cm, both in
hydrostatic equilibrium under a uniform inward acceleration g.  The interface is
seeded with eps * P_l(cos theta).  Incompressible linear theory for a spherical
interface (heavy fluid outside) gives

    gamma^2 = (g / R) l (l + 1) (rho2 - rho1) / (l rho2 + (l + 1) rho1),

which tends to A k g for l >> 1.  The run measures the exponential growth rate of
the P_l amplitude of the interface and reports gamma_sim / gamma_theory for the
default anti-hourglass stiffness and for a list of others, so the numerical damping
of low-l modes can be quantified."""
from common import *
import numpy as np
import sife.constants as C
from sife import ideal, Layer
from sife.simulation2d import Simulation2D, Options2D, legendre
from sife.eos import energies_from_T

RHO1, RHO2, R0, D2 = 1.0, 3.0, 1.0, 0.3
T_TOP = 100.0                      # eV at the outer wall -> c_s ~ 1.8e7 cm/s


def gamma_theory(l, g, R=R0):
    return np.sqrt(g / R * l * (l + 1) * (RHO2 - RHO1) / (l * RHO2 + (l + 1) * RHO1))


def setup(l, Nth, hg, eps, gamma_target=2e7, N1=100, N2=30, cls=Simulation2D):
    g = gamma_target ** 2 / gamma_theory(l, 1.0) ** 2
    o = Options2D(degeneracy=False, cond_e=False, cond_i=False, exchange=False, radiation=False, burn=False,
                  verbose=0, dt_max=1e-9, history_every=10 ** 9, cfl=0.3, dt_init=1e-12, snap_dt=1.0,
                  fix_outer=True, core_rings=N1 // 2, gravity=g, hourglass=hg)
    layers = [Layer(ideal("light", 1.0, 1.0, RHO1), R0, N1, T0=T_TOP), Layer(ideal("heavy", 1.0, 1.0, RHO2), D2, N2, T0=T_TOP)]
    sim = cls(layers, n_theta=Nth, half=True, opts=o, perturbations=[(0, l, eps)])
    # discrete hydrostatic pressure (exact for the symmetric 1D state), integrated inward from the wall
    r = sim.r.mean(axis=1)
    rho = sim.rho.mean(axis=1)
    mz = rho * 4 * np.pi / 3 * np.diff(r ** 3)
    P = np.empty(sim.Ni)
    n_top = rho[-1] / (1.0 * C.mu)
    P[-1] = 2 * n_top * T_TOP * C.eV
    for i in range(sim.Ni - 1, 0, -1):          # node i between zones i-1 and i
        P[i - 1] = P[i] + g * 0.5 * (mz[i - 1] + mz[i]) / (4 * np.pi * r[i] ** 2)
    T = P[:, None] / (2 * sim.rho / (sim.Ab_mu) * C.eV)
    sim.Te[:] = T; sim.Ti[:] = T
    n = sim.Ni * sim.Nj
    F = lambda a: a.reshape(n)
    energies_from_T(F(sim.rho), F(sim.Te), F(sim.Ti), F(sim.Zb), F(sim.Ab_mu), sim.deg, F(sim.ee), F(sim.ei))
    sim._eos()
    return sim, g


def interface_amp(sim, l):
    if hasattr(sim, "X"):           # Godunov ALE: interface from the mass fractions
        return sim.legendre_amp(sim.interface_radius(0), l)[0]
    k = int(np.argmax(sim.mat[:, 0] == 1))
    rr = sim.r[k]; th = sim.theta; w = np.sin(th)
    P = legendre(l, np.cos(th))
    r0 = np.trapezoid(rr * w, th) / np.trapezoid(w, th)
    return np.trapezoid((rr - r0) * P * w, th) / np.trapezoid(P * P * w, th)


def growth(l=20, Nth=64, hg=0.05, eps=2e-4, n_efold=3.0, cls=Simulation2D):
    sim, g = setup(l, Nth, hg, eps, cls=cls)
    gth = gamma_theory(l, g)
    t_end = n_efold / gth
    ts, a = [0.0], [interface_amp(sim, l)]
    while sim.t < t_end:
        sim.run(min(sim.t + t_end / 60, t_end))
        ts.append(sim.t); a.append(interface_amp(sim, l))
    ts = np.array(ts); a = np.array(a)
    # fit a(t) = c cosh(gamma t): from the late-time logarithmic slope
    m = ts > 0.4 * t_end
    gsim = np.polyfit(ts[m], np.log(np.abs(a[m])), 1)[0]
    return dict(l=l, Nth=Nth, hg=hg, gamma_theory=gth, gamma_sim=gsim, ratio=gsim / gth,
                amp_growth=float(a[-1] / a[0])), ts * gth, a / a[0]


def run(show=True, cases=((20, 64, "ale", 0.0), (20, 64, "lag", 0.05), (20, 64, "lag", 0.0),
                          (8, 32, "ale", 0.0), (8, 32, "lag", 0.05), (8, 32, "lag", 0.0),
                          (8, 16, "ale", 0.0), (8, 16, "lag", 0.0))):
    from sife.simulation_ale import SimulationALE
    res = []
    if show:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    for (l, Nth, solver, hg) in cases:
        cls = SimulationALE if solver == "ale" else Simulation2D
        d, tt, aa = growth(l, Nth, hg, cls=cls)
        d["solver"] = solver
        res.append(d)
        print(d, flush=True)
        if show:
            lab = "ALE" if solver == "ale" else f"Lagrangian hg={hg}"
            ax.semilogy(tt, aa, "-" if solver == "ale" else ":",
                        label=f"l={l}, {Nth} zones/90deg, {lab}: gamma/theory={d['ratio']:.2f}")
    if show:
        tt = np.linspace(0, 3, 50)
        ax.semilogy(tt, np.cosh(tt), "k--", label="linear theory cosh(gamma t)")
        ax.set_xlabel("gamma_theory * t"); ax.set_ylabel("interface P_l amplitude / initial")
        ax.set_title("Rayleigh-Taylor growth, spherical interface"); ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(f"{OUT}/rt2d.png")
    return res


if __name__ == "__main__":
    run()
