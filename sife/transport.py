"""Collisional plasma coefficients (CGS, T in eV).

* Coulomb logarithm: NRL formulary electron-ion form, floored at 2.
* Electron collision time: Braginskii tau_e with T replaced by the
  degeneracy-corrected T_eff (Lee-More-like saturation in cold dense fuel).
* Electron thermal conductivity: Spitzer-Harm with the Epperlein-Haines-type
  Z interpolation  gamma(Z) = 13.58 (Z+0.24)/(Z+4.2)  (3.16 at Z=1, Lorentz limit
  128/3pi at Z->inf).
* Ion conductivity: Braginskii 3.9 n_i T_i tau_i / m_i.
* e-i energy exchange: Q = 3 (m_e/m_i) n_e (Te-Ti)/tau_e.
* Free-free (bremsstrahlung) Planck and Rosseland mean opacities (Kramers,
  Planck/Rosseland ratio 30.26), plus Thomson scattering in the Rosseland mean.
* Inverse-bremsstrahlung absorption coefficient for the laser.
"""
import math

import numpy as np
from numba import njit

from .constants import CF, a_rad, c, e, eV, me, mu

SQ2PI = math.sqrt(2.0 * math.pi)
E4 = e ** 4
SQME = math.sqrt(me)
P_BREM = 1.69e-25        # erg/s cm^3 eV^-1/2  (NRL: 1.69e-32 W cm^-3)
SIGMA_T = 0.6652e-24
FF_RATIO = 1.0 / 30.262


@njit(cache=True)
def coulomb_log(ne, Te, Z):
    if Te < 10.0 * Z * Z:
        L = 23.0 - math.log(math.sqrt(ne) * Z * Te ** -1.5)
    else:
        L = 24.0 - math.log(math.sqrt(ne) / Te)
    return max(L, 2.0)


@njit(cache=True)
def teff(ne, Te):
    TF = 0.4 * CF * ne ** (2.0 / 3.0)
    return math.sqrt(Te * Te + TF * TF)


@njit(cache=True)
def tau_e(ne, Te, Zeff, lnL):
    T = teff(ne, Te) * eV
    return 3.0 * SQME * T ** 1.5 / (4.0 * SQ2PI * ne * Zeff * E4 * lnL)


@njit(cache=True)
def gamma_Z(Z):
    return 13.58 * (Z + 0.24) / (Z + 4.2)


@njit(cache=True)
def kappa_e(ne, Te, Zeff, lnL):
    """Spitzer-Harm conductivity, erg / (s cm eV)."""
    return gamma_Z(Zeff) * ne * (Te * eV) * tau_e(ne, Te, Zeff, lnL) / me * eV


@njit(cache=True)
def kappa_i(ni, Ti, Z, A_mu, lnL):
    mi = A_mu
    tau_i = 3.0 * math.sqrt(mi) * (Ti * eV) ** 1.5 / (4.0 * math.sqrt(math.pi) * ni * Z ** 4 * E4 * lnL)
    return 3.9 * ni * (Ti * eV) * tau_i / mi * eV


@njit(cache=True)
def q_free_stream(ne, Te):
    T = Te * eV
    return ne * T * math.sqrt(T / me)


@njit(cache=True)
def exchange_coef(ne, Te, Zeff, A_mu, lnL):
    """Return omega such that Q_ei = omega * (Te - Ti) [erg/cc/s/eV]."""
    return 3.0 * (me / A_mu) * ne * eV / tau_e(ne, Te, Zeff, lnL)


@njit(cache=True)
def opacities(ne, ni, Te, Z2, mult):
    """Planck and Rosseland mean absorption coefficients [1/cm]."""
    T = max(Te, 1.0e-2)
    Pbr = P_BREM * ne * math.sqrt(T) * ni * Z2 * mult
    kP = Pbr / (c * a_rad * T ** 4)
    kR = kP * FF_RATIO + SIGMA_T * ne
    return kP, kR


@njit(cache=True)
def digamma(x):
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    x2 = 1.0 / (x * x)
    return r + math.log(x) - 0.5 / x - x2 * (1.0 / 12.0 - x2 * (1.0 / 120.0 - x2 / 252.0))


@njit(cache=True)
def spectral_factor(Te, Tr):
    """Ratio of the free-free absorption coefficient averaged over a Planck
    spectrum at Tr to the Planck mean at Te:
        G = (Te/Tr)^3 [psi(1 + Tr/Te) + gamma_E]      (G = 1 when Tr = Te).
    Hard radiation from a hot spot (Tr >> Te) is absorbed far less by cold
    dense fuel than a one-temperature gray model would predict."""
    s = max(Tr, 1e-3) / max(Te, 1e-3)
    if s < 0.1:
        s = 0.1
    return (digamma(1.0 + s) + 0.5772156649015329) / (s * s * s)


@njit(cache=True)
def nu_ei(ne, Te, Zeff, lnL):
    return 1.0 / tau_e(ne, Te, Zeff, lnL)


@njit(cache=True)
def kappa_ib(ne, nc, Te, Zeff, lnL):
    """Inverse bremsstrahlung absorption coefficient [1/cm] (x = ne/nc < 1)."""
    x = ne / nc
    if x >= 1.0:
        return 1.0e30
    return nu_ei(ne, Te, Zeff, lnL) / c * x / math.sqrt(1.0 - x)
