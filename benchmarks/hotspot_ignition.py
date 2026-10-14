"""Isobaric hot-spot ignition threshold.
Static assembly: central DT hot spot (T_h, rho_h, radius R_h) inside cold dense DT fuel
(rho_c, rho_c*dR = 2 g/cm^2) at the same pressure P = 200 Gbar.  Full physics (hydro,
conduction, gray radiation, alpha transport).  The ignition threshold (rho R)_h at a given
T_h is compared with the classic isobaric ignition condition (Atzeni & Meyer-ter-Vehn 2004,
Ch. 4: (rho R)_h ~ 0.2-0.3 g/cm^2 at T_h ~ 8-12 keV)."""
from common import *
import numpy as np
from scipy.optimize import brentq
from sife import Simulation, Options, DT_gas, constants as C
from sife.eos import fermi_T

P = 2.0e17   # erg/cc = 200 Gbar

def cold_T(rho):
    ni = rho / (2.5 * C.mu); TF = fermi_T(ni, 1)
    f = lambda T: ni * np.sqrt(T * T + TF * TF) * C.eV + ni * T * C.eV - P
    return brentq(f, 1e-3, 1e5)

def assembly(rhoR_h, Th, rho_c=600.0, rhoR_c=2.0, Nh=60, Nc=100):
    rho_h = P / (2 * Th * C.eV) * 2.5 * C.mu
    Rh = rhoR_h / rho_h; dRc = rhoR_c / rho_c
    Tc = cold_T(rho_c)
    r = np.concatenate([np.linspace(0, Rh, Nh + 1), Rh + dRc * (1 - np.linspace(1, 0, Nc + 1)[1:] ** 1.0)])
    N = len(r) - 1
    mat = np.zeros(N, int)
    rho = np.where(np.arange(N) < Nh, rho_h, rho_c)
    T = np.where(np.arange(N) < Nh, Th, Tc)
    o = Options(verbose=0, dt_init=1e-15, dt_max=1e-12, snap_dt=1.0, history_every=20, alpha_n_mu=12)
    sim = Simulation(r=r, mat_idx=mat, materials=[DT_gas(rho_c)], rho=rho, Te=T, Ti=T.copy(), opts=o)
    return sim, rho_h, Rh, Tc

def run_one(rhoR_h, Th):
    sim, rho_h, Rh, Tc = assembly(rhoR_h, Th)
    cs = np.sqrt(5 / 3 * P / (P / (2 * Th * C.eV) * 2.5 * C.mu))
    sim.run(30 * Rh / cs + 200e-12)
    fb = 1 - np.sum(sim.m * sim.yDT) / np.sum(sim.m * sim.yDT0)
    return fb, sim

def run(show=True):
    Ths = [5e3, 8e3, 12e3]
    rhoRs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    res = {}
    for Th in Ths:
        res[Th] = []
        for x in rhoRs:
            fb, sim = run_one(x, Th)
            res[Th].append(fb)
            print(f"T_h={Th/1e3:4.1f} keV  rhoR_h={x:4.2f}  burn-up={fb:.4f}  Ti_max={max(sim.hist['Ti_max'])/1e3:6.1f} keV", flush=True)
    thr = {}
    for Th in Ths:
        fb = np.array(res[Th])
        ig = np.where(fb > 0.05)[0]
        thr[Th / 1e3] = float(rhoRs[ig[0]]) if len(ig) else None
    if show:
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        for k, Th in enumerate(Ths):
            ax.semilogy(rhoRs, np.maximum(res[Th], 1e-6), f"C{k}o-", label=f"T_h = {Th/1e3:.0f} keV")
        ax.axvspan(0.2, 0.3, color="0.85", label="classic threshold 0.2-0.3")
        ax.set_xlabel(r"hot-spot $\rho R_h$ [g/cm$^2$]"); ax.set_ylabel("fuel burn-up fraction")
        ax.set_title("Isobaric hot-spot ignition (P = 200 Gbar, cold fuel ρR = 2)"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{OUT}/hotspot_ignition.png")
    return dict(rhoR_h=rhoRs, burnup={str(k / 1e3): v for k, v in res.items()}, threshold_rhoR_h=thr)

if __name__ == "__main__":
    print(run())
