"""Inverse-bremsstrahlung absorption in a linear density ramp ne = nc x / L.
Analytic (Kruer; constant Coulomb log):  A = 1 - exp(-32 nu_c L cos^5(theta) / (15 c)).
Reference with the code's own lnL(ne): A = 1 - exp(-2 Int nu(x) X / (c sqrt(cos^2 th - X)) dx).
The 1D spherical ray tracer is tested on a large-radius sphere (R = 1 cm >> L = 100 um)."""
from common import *
import numpy as np
from scipy.integrate import quad
from sife import constants as C
from sife.laser import critical_density, trace_rays
from sife.transport import nu_ei, coulomb_log

lam = 0.248; nc = critical_density(lam)
Te = 2000.0; Zeff = 18.5 / 3.5   # CH plasma at 2 keV
L = 30e-4; R0 = 10.0

def reference(theta):
    c2 = np.cos(theta) ** 2
    f = lambda X: nu_ei(X * nc, Te, Zeff, coulomb_log(X * nc, Te, Zeff)) * X / (C.c * np.sqrt(c2 - X)) * L
    tau = 2 * quad(f, 0, c2, limit=200)[0]
    return 1 - np.exp(-tau)

def kruer(theta):
    nu_c = nu_ei(nc, Te, Zeff, coulomb_log(nc, Te, Zeff))
    return 1 - np.exp(-32 * nu_c * L * np.cos(theta) ** 5 / (15 * C.c))

def code(theta, N):
    rin = R0 - 1.5 * L
    r = np.concatenate([[0.0], np.linspace(rin, R0, N + 1)])
    rc = 0.5 * (r[1:] + r[:-1])
    ne = nc * (R0 - rc) / L
    ne[0] = 10 * nc
    Tz = np.full(len(rc), Te); Zz = np.full(len(rc), Zeff)
    dep = np.zeros(len(rc))
    b = np.array([R0 * np.sin(theta) * 0.999999]); w = np.array([1.0])
    ab, mi, es = trace_rays(r, ne, Tz, Zz, nc, b, w, 1.0, 0.0, dep)
    return ab

def run(show=True):
    th = np.radians([0, 15, 30, 45, 60])
    res = []
    for t in th:
        res.append(dict(theta=float(np.degrees(t)), kruer=float(kruer(t)), reference=float(reference(t)),
                        sife_N20=float(code(t, 30)), sife_N50=float(code(t, 75)), sife_N300=float(code(t, 450))))
    if show:
        tt = np.radians(np.linspace(0, 75, 60))
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.plot(np.degrees(tt), [kruer(t) for t in tt], "k--", lw=1, label="Kruer analytic (lnΛ at nc)")
        ax.plot(np.degrees(tt), [reference(t) for t in tt], "k-", lw=1, label="planar integral (code lnΛ)")
        ax.plot(np.degrees(tt), [code(t, 75) for t in tt], "C1o", ms=3, label="SIFE ray trace, 50 zones/L")
        ax.plot(np.degrees(tt), [code(t, 450) for t in tt], "C3.", ms=4, label="SIFE ray trace, 300 zones/L")
        ax.set_xlabel("incidence angle [deg]"); ax.set_ylabel("absorption fraction")
        ax.set_title("KrF IB absorption, linear ramp L=30 µm, CH, Te=2 keV"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{OUT}/laser_absorption.png")
    return res

if __name__ == "__main__":
    for x in run(): print(x)
