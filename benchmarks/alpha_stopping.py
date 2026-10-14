"""Alpha-particle (3.5 MeV) slowing down in DT plasma.
Compared with the fits quoted by Atzeni & Meyer-ter-Vehn (2004, Sec. 4.3):
  range   rho*l_a = 1.5e-2 T^(5/4) / (1 + 8.2e-3 T^(5/4))  g/cm^2   (T in keV, Te = Ti)
  fraction of alpha energy given to ions  f_i ~ T / (32 + T)."""
from common import *
import numpy as np
from sife import constants as C
from sife.fusion import alpha_loss_rates
from sife.transport import coulomb_log

def slow_down(T_keV, rho=100.0):
    ni = rho / (2.5 * C.mu); ne = ni
    T = T_keV * 1e3
    lnLe = coulomb_log(ne, T, 1.0); lnLi = lnLe + 1.0
    E = C.E_alpha; s = 0.0; Ee = Ei = 0.0
    while E > 2 * T * C.eV:
        se, si = alpha_loss_rates(E, ne, T, ni, T, 1.0, 2.5 * C.mu, lnLe, lnLi)
        ds = 0.01 * E / (se + si)
        Ee += se * ds; Ei += si * ds; E -= (se + si) * ds; s += ds
    Ei += E * si / (se + si); Ee += E * se / (se + si)
    return rho * s, Ei / (Ee + Ei)

def run(show=True, rho=10.0):
    Ts = np.array([2, 5, 10, 20, 30, 50, 80])
    out = []
    for T in Ts:
        rl, fi = slow_down(T, rho)
        fit = 1.5e-2 * T ** 1.25 / (1 + 8.2e-3 * T ** 1.25)
        out.append(dict(T_keV=float(T), rhoR_sife=rl, rhoR_fit=fit, fi_sife=fi, fi_fit=T / (32 + T)))
    if show:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        TT = np.linspace(1, 100, 200)
        ax[0].plot(TT, 1.5e-2 * TT ** 1.25 / (1 + 8.2e-3 * TT ** 1.25), "k-", lw=1, label="fit (Atzeni & MtV)")
        for i, rho in enumerate([1.0, 10.0, 100.0]):
            ax[0].plot(Ts, [slow_down(T, rho)[0] for T in Ts], f"C{i+1}o-", ms=4, lw=0.6, label=rf"SIFE, $\rho$={rho:g} g/cc")
        ax[0].set_xlabel("T [keV]"); ax[0].set_ylabel(r"$\rho\ell_\alpha$ [g/cm$^2$]"); ax[0].legend()
        ax[1].plot(TT, TT / (32 + TT), "k-", lw=1, label="T/(32+T)  (Fraley et al. 1974)")
        ax[1].plot(Ts, [o["fi_sife"] for o in out], "C0o", ms=4, label="SIFE")
        ax[1].set_xlabel("T [keV]"); ax[1].set_ylabel("fraction of alpha energy to ions"); ax[1].legend()
        fig.suptitle("3.5 MeV alpha slowing down in DT")
        fig.tight_layout(); fig.savefig(f"{OUT}/alpha_stopping.png")
    return out

if __name__ == "__main__":
    for o in run(): print({k: round(v, 4) for k, v in o.items()})
