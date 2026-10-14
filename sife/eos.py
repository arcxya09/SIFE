"""Two-temperature equation of state.

Electrons: fully ionized ideal Fermi gas approximated by an effective
temperature  T_eff = sqrt(Te^2 + T_F^2),  T_F = (2/5) E_F, so that
    P_e = n_e T_eff,   e_e = (3/2) n_e T_eff / rho
which is exact in both the classical (Te >> E_F) and fully degenerate (Te = 0)
limits.  This captures the Fermi pressure that sets the fuel adiabat
alpha = P / P_deg.

Ions: ideal gas, P_i = n_i T_i, e_i = (3/2) n_i T_i / rho.

Cold curve (solids only): a density-only pressure P_c(rho) chosen so that the
T=0 isotherm has zero pressure and bulk modulus B0 at rho0, and reduces to the
Fermi pressure at high compression; for rho < rho0 it cancels the degenerate
pressure and adds a bounded tension (max 0.2 B0, vanishing below 0.6 rho0) so a
cryogenic surface does not spuriously expand into the low-pressure vapour.  Its energy e_c(rho) is a potential, so it needs no
separate evolution equation.

All routines are numba kernels acting on zone arrays.  Temperatures in eV.
"""
import numpy as np
from numba import njit

from .constants import CF, eV

T_FLOOR = 1.0e-4  # eV


@njit(cache=True)
def cold_curve(rho, rho0, Pd0, B0, solid):
    """Return (P_c, e_c, dP_c/drho) for one zone."""
    if solid == 0:
        return 0.0, 0.0, 0.0
    eta = rho / rho0
    if eta >= 1.0:
        A = Pd0 - 3.0 * B0
        B = -2.0 * Pd0 + 3.0 * B0
        e13 = eta ** (1.0 / 3.0)
        Pc = A * eta + B * eta * e13
        ec = (A * np.log(eta) + 3.0 * B * (e13 - 1.0)) / rho0
        dP = (A + 4.0 / 3.0 * B * e13) / rho0
    else:
        # cancel the degenerate pressure, plus a bounded tension B0*g(eta) so a
        # cold solid surface does not 'sublime' into a low-pressure gas:
        #   g = eta-1 (0.8<eta<1),  -(eta-0.6) (0.6<eta<0.8),  0 (eta<0.6)
        e23 = eta ** (2.0 / 3.0)
        Pc = -Pd0 * eta * e23
        ec = -1.5 * Pd0 * (e23 - 1.0) / rho0
        dP = -5.0 / 3.0 * Pd0 * e23 / rho0
        # energy integral  (B0/rho0) * Int_1^eta g(x)/x^2 dx
        if eta >= 0.8:
            Pc += B0 * (eta - 1.0)
            ec += B0 / rho0 * (np.log(eta) + 1.0 / eta - 1.0)
            dP += B0 / rho0
        else:
            I08 = np.log(0.8) + 1.0 / 0.8 - 1.0
            if eta >= 0.6:
                Pc += -B0 * (eta - 0.6)
                ec += B0 / rho0 * (I08 - (np.log(eta) + 0.6 / eta - np.log(0.8) - 0.6 / 0.8))
                dP += -B0 / rho0
            else:
                ec += B0 / rho0 * (I08 - (np.log(0.6) + 1.0 - np.log(0.8) - 0.6 / 0.8))
    return Pc, ec, dP


@njit(cache=True)
def fermi_T(ne, deg):
    """Degeneracy temperature T_F = 0.4 E_F [eV]."""
    if deg == 0:
        return 0.0
    return 0.4 * CF * ne ** (2.0 / 3.0)


@njit(cache=True)
def eos_update(rho, ee, ei, Zb, Ab_mu, rho0, Pd0, B0, solid, deg,
               Te, Ti, Pe, Pi, Pc, cs2, Cve, Cvi):
    """From (rho, specific electron/ion thermal energies) compute temperatures,
    pressures, heat capacities and sound speed squared (in place)."""
    n = rho.shape[0]
    for k in range(n):
        ni = rho[k] / Ab_mu[k]
        ne = Zb[k] * ni
        TF = fermi_T(ne, deg)
        # electrons
        Teff = ee[k] * rho[k] / (1.5 * ne * eV)
        if Teff <= TF * (1.0 + 1e-12) or Teff <= T_FLOOR:
            T = T_FLOOR
            Teff = np.sqrt(TF * TF + T * T)
            ee[k] = 1.5 * ne * Teff * eV / rho[k]
        else:
            T = np.sqrt(Teff * Teff - TF * TF)
            if T < T_FLOOR:
                T = T_FLOOR
                Teff = np.sqrt(TF * TF + T * T)
                ee[k] = 1.5 * ne * Teff * eV / rho[k]
        Te[k] = T
        Pe[k] = ne * Teff * eV
        Cve[k] = 1.5 * ne * eV / rho[k] * T / Teff
        # ions
        Tion = ei[k] * rho[k] / (1.5 * ni * eV)
        if Tion < T_FLOOR:
            Tion = T_FLOOR
            ei[k] = 1.5 * ni * Tion * eV / rho[k]
        Ti[k] = Tion
        Pi[k] = ni * Tion * eV
        Cvi[k] = 1.5 * ni * eV / rho[k]
        # cold
        pc, ec, dpc = cold_curve(rho[k], rho0[k], Pd0[k], B0[k], solid[k])
        Pc[k] = pc
        # sound speed: adiabatic thermal part + cold part.  For the degenerate
        # electron gas d(n T_F)/drho|_s = 5/3 * n T_F/rho as well.
        c2 = 5.0 / 3.0 * (Pe[k] + Pi[k]) / rho[k] + dpc
        cmin = 5.0 / 3.0 * Pi[k] / rho[k]
        if c2 < cmin:
            c2 = cmin
        cs2[k] = c2


@njit(cache=True)
def energies_from_T(rho, Te, Ti, Zb, Ab_mu, deg, ee, ei):
    n = rho.shape[0]
    for k in range(n):
        ni = rho[k] / Ab_mu[k]
        ne = Zb[k] * ni
        TF = fermi_T(ne, deg)
        ee[k] = 1.5 * ne * np.sqrt(Te[k] ** 2 + TF ** 2) * eV / rho[k]
        ei[k] = 1.5 * ni * Ti[k] * eV / rho[k]


@njit(cache=True)
def cold_energy(rho, rho0, Pd0, B0, solid):
    n = rho.shape[0]
    out = np.empty(n)
    for k in range(n):
        out[k] = cold_curve(rho[k], rho0[k], Pd0[k], B0[k], solid[k])[1]
    return out


def fermi_pressure_DT(rho):
    """Fully degenerate DT electron pressure [erg/cc] (for adiabat diagnostics)."""
    from .constants import mu
    ne = rho / (2.5 * mu)
    return 0.4 * CF * ne ** (2.0 / 3.0) * ne * eV
