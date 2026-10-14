"""DT thermonuclear burn and alpha-particle transport.

* Reactivity: Bosch & Hale, Nucl. Fusion 32, 611 (1992), valid 0.2-100 keV.
* Burn: per-zone fuel depletion  dy/dt = -(rho <sv>/2) y^2  (y = DT ions per gram,
  50/50 D:T), integrated exactly over a step.
* Each reaction releases a 3.5 MeV alpha (transported) and a 14.1 MeV neutron
  (assumed to escape; tallied).
* Alpha transport: straight-line, multi-angle (Gauss-Legendre in mu) ray
  slowing-down through the spherical mesh.  Energy loss rate to each field
  species f follows the Trubnikov / NRL formulary expression
      d eps/dt = -nu_eps eps,   nu_eps = 2 [ (m_a/m_f) psi(x) - psi'(x) ] nu_0,
      x = m_f v^2 / (2 T_f),  psi(x) = erf(sqrt x) - 2 sqrt(x/pi) e^-x,
      nu_0 = 4 pi e^4 Z_a^2 Z_f^2 n_f lnL / (m_a^2 v^3),
  which covers both the v_a << v_te (electron) and v_a >> v_ti (ion) limits
  and gives the electron/ion heating split self-consistently.
"""
import math

import numpy as np
from numba import njit

from . import constants as C
from .transport import coulomb_log

_BG = 34.3827
_MRC2 = 1124656.0
_C = (1.17302e-9, 1.51361e-2, 7.51886e-2, 4.60643e-3, 1.35000e-2, -1.06750e-4, 1.36600e-5)


@njit(cache=True)
def sigv_DT(T_keV):
    """Bosch-Hale DT reactivity <sigma v> [cm^3/s], T in keV."""
    T = T_keV
    if T < 0.2:
        if T < 0.05:
            return 0.0
        T = T  # extrapolate the fit below its range (negligible rates anyway)
    if T > 100.0:
        T = 100.0
    C1, C2, C3, C4, C5, C6, C7 = _C
    theta = T / (1.0 - T * (C2 + T * (C4 + T * C6)) / (1.0 + T * (C3 + T * (C5 + T * C7))))
    xi = (_BG * _BG / (4.0 * theta)) ** (1.0 / 3.0)
    return C1 * theta * math.sqrt(xi / (_MRC2 * T ** 3)) * math.exp(-3.0 * xi)


@njit(cache=True)
def burn_step(rho, m, Ti, yDT, dt, Sa):
    """Deplete fuel over dt.  yDT: DT ions per gram (updated in place).
    Sa: alpha energy released per zone [erg] (output).  Returns number of reactions."""
    N = rho.shape[0]
    ntot = 0.0
    for k in range(N):
        Sa[k] = 0.0
        if yDT[k] <= 0.0:
            continue
        sv = sigv_DT(Ti[k] * 1.0e-3)
        if sv <= 0.0:
            continue
        y0 = yDT[k]
        y1 = y0 / (1.0 + 0.5 * y0 * rho[k] * sv * dt)
        yDT[k] = y1
        nreac = 0.5 * (y0 - y1) * m[k]
        ntot += nreac
        Sa[k] = nreac * C.E_alpha
    return ntot


@njit(cache=True)
def _psi(x):
    if x < 1e-6:
        return 4.0 / (3.0 * math.sqrt(math.pi)) * x ** 1.5
    sx = math.sqrt(x)
    return math.erf(sx) - 2.0 * sx / math.sqrt(math.pi) * math.exp(-x)


@njit(cache=True)
def _dpsi(x):
    return 2.0 / math.sqrt(math.pi) * math.sqrt(x) * math.exp(-x)


@njit(cache=True)
def alpha_loss_rates(E, ne, Te, ni, Ti, Z2i, Ai_mu, lnLe, lnLi):
    """Energy loss per unit path length [erg/cm] to electrons and ions."""
    ma = C.m_alpha
    v = math.sqrt(2.0 * E / ma)
    k0 = 4.0 * math.pi * C.e ** 4 * C.Z_alpha ** 2 / (ma * ma * v ** 3)
    # electrons
    xe = C.me * v * v / (2.0 * Te * C.eV)
    nue = 2.0 * ((ma / C.me) * _psi(xe) - _dpsi(xe)) * k0 * ne * lnLe
    # ions
    xi = Ai_mu * v * v / (2.0 * Ti * C.eV)
    nui = 2.0 * ((ma / Ai_mu) * _psi(xi) - _dpsi(xi)) * k0 * ni * Z2i * lnLi
    if nue < 0.0:
        nue = 0.0
    if nui < 0.0:
        nui = 0.0
    return nue * E / v, nui * E / v


@njit(cache=True)
def alpha_transport_core(r, Sa, ne, Te, ni, Ti, Z2i, Ai_mu, mus, wts, dep_e, dep_i):
    N = Sa.shape[0]
    for k in range(N):
        dep_e[k] = 0.0
        dep_i[k] = 0.0
    lnLe = np.empty(N)
    lnLi = np.empty(N)
    for k in range(N):
        lnLe[k] = coulomb_log(ne[k], Te[k], 1.0)
        lnLi[k] = lnLe[k] + 1.0
    escaped = 0.0
    Smax = 0.0
    for k in range(N):
        if Sa[k] > Smax:
            Smax = Sa[k]
    for k0 in range(N):
        S = Sa[k0]
        if S <= 1.0e-8 * Smax or S <= 0.0:
            continue
        r0 = 0.5 * (r[k0] + r[k0 + 1])
        for jm in range(mus.shape[0]):
            mu_ = mus[jm]
            Wt = S * wts[jm]
            E = C.E_alpha
            p = r0 * math.sqrt(max(0.0, 1.0 - mu_ * mu_))
            t = r0 * mu_
            z = k0
            alive = True
            while alive:
                # exit point of current zone
                if t < 0.0 and p < r[z]:
                    t_exit = -math.sqrt(r[z] * r[z] - p * p)
                    znext = z - 1
                else:
                    t_exit = math.sqrt(r[z + 1] * r[z + 1] - p * p)
                    znext = z + 1
                L = t_exit - t
                if L < 0.0:
                    L = 0.0
                Tth = 2.0 * max(Te[z], Ti[z]) * C.eV
                # slow down along L in zone z
                while L > 0.0:
                    se, si = alpha_loss_rates(E, ne[z], Te[z], ni[z], Ti[z], Z2i[z], Ai_mu[z], lnLe[z], lnLi[z])
                    st = se + si
                    if st <= 0.0:
                        break
                    ds = 0.1 * E / st
                    if ds > L:
                        ds = L
                    dE = st * ds
                    if dE > E:
                        dE = E
                    fe = se / st
                    dep_e[z] += Wt * dE / C.E_alpha * fe
                    dep_i[z] += Wt * dE / C.E_alpha * (1.0 - fe)
                    E -= dE
                    L -= ds
                    if E <= Tth:
                        # thermalized: dump the remainder here with the same split
                        dep_e[z] += Wt * E / C.E_alpha * fe
                        dep_i[z] += Wt * E / C.E_alpha * (1.0 - fe)
                        E = 0.0
                        alive = False
                        break
                if not alive:
                    break
                t = t_exit
                z = znext
                if z >= N:
                    escaped += Wt * E / C.E_alpha
                    alive = False
                elif z < 0:
                    z = 0
    return escaped


def gauss_mu(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w / 2.0
