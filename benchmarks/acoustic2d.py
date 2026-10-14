"""Non-radial (l = 2) acoustic eigenmode of a gas sphere with a rigid wall.

Linear theory: p' ~ j_l(k r) P_l(cos theta) cos(omega t), with j_l'(k R) = 0, so
for l = 2 the fundamental is k R = 3.3421 and omega = c_s k.  The mode needs
angular pressure forces, angular momentum and the angular remap working together,
so its period checks the non-radial part of the 2D scheme."""
from common import *
import numpy as np
from scipy.special import spherical_jn
from sife import ideal, Layer
from sife.simulation2d import Simulation2D, Options2D, legendre

X2 = 3.342093657


def run(N=60, Nth=24, eps=2e-3, show=True):
    mat = ideal("gas", 1.0, 2.5, 1.0)
    R = 1.0
    o = Options2D(degeneracy=False, cond_e=False, cond_i=False, exchange=False, radiation=False,
                  burn=False, verbose=0, dt_max=1e-6, history_every=10 ** 9, C1=0.0, C2=0.0, cfl=0.3,
                  dt_init=1e-12, snap_dt=1.0, fix_outer=True, core_rings=2)
    sim = Simulation2D([Layer(mat, R, N, T0=100.0)], n_theta=Nth, half=True, opts=o)
    g = sim.geometry()
    k = X2 / R
    shape = spherical_jn(2, k * g["rc"]) * legendre(2, np.cos(sim.theta_c))[None, :]
    sim.ei *= 1.0 + eps * shape
    sim._eos(); sim.E0 = sim.total_energy()
    c = float(np.sqrt(sim.cs2.mean()))
    T_ex = 2 * np.pi / (c * k)
    # probe: amplitude of the l=2 projection of the pressure

    ts, amp = [], []
    P0 = sim.P.mean()
    t_end = 2.5 * T_ex
    while sim.t < t_end:
        sim.run(min(sim.t + T_ex / 80, t_end))
        ts.append(sim.t)
        Vz = sim.m / sim.rho
        amp.append(float(np.sum((sim.P - P0) * shape * Vz) / np.sum(shape ** 2 * Vz) / P0))
    ts = np.array(ts); amp = np.array(amp)
    # period from zero crossings
    ii = np.where(np.diff(np.sign(amp)) != 0)[0]
    zc = ts[ii] - amp[ii] * (ts[ii + 1] - ts[ii]) / (amp[ii + 1] - amp[ii])
    T_sim = 2 * float(np.mean(np.diff(zc))) if len(zc) > 2 else float("nan")
    if show:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(ts / T_ex, amp / max(abs(amp[0]), 1e-30), "C0.-", ms=3, label="SIFE-2D")
        ax.plot(ts / T_ex, np.cos(2 * np.pi * ts / T_ex), "k-", lw=0.8, label="linear theory")
        ax.set_xlabel("t / T_exact"); ax.set_ylabel("l = 2 pressure amplitude (normalized)")
        ax.set_title("l = 2 acoustic mode of a rigid sphere"); ax.legend()
        fig.tight_layout(); fig.savefig(f"{OUT}/acoustic2d.png")
    return dict(T_exact=T_ex, T_sim=T_sim, rel_err=abs(T_sim - T_ex) / T_ex,
                amp_ratio_after_2p5_periods=float(abs(amp[-20:]).max() / abs(amp[:5]).max()),
                energy_err=float(sim.energy_error()))


if __name__ == "__main__":
    print(run())
