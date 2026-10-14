"""Material definitions.

Each material is described by average ionic properties (fully ionized plasma),
an optional cold curve for solids (so that a cryogenic DT or CH layer sits at
zero pressure at its initial density), and the DT fuel fraction used by the burn
module.
"""
from dataclasses import dataclass

from . import constants as C


@dataclass
class Material:
    name: str
    Z: float            # mean ionic charge <Z> (full ionization)
    Z2: float           # <Z^2>
    A: float            # mean ion mass [amu]
    rho0: float         # reference (initial) density [g/cc]
    fDT: float = 0.0    # fraction of ions that are D or T (50/50 mix)
    solid: bool = False # apply cold-curve correction
    B0: float = 0.0     # bulk modulus at rho0 [erg/cc] (only for solids)
    opac_mult: float = 1.0  # multiplier on free-free opacity (crude bound-free/line enhancement)

    @property
    def Zeff(self):
        """Effective charge for collisions, <Z^2>/<Z>."""
        return self.Z2 / self.Z


def DT_ice(rho0=0.25):
    return Material("DT-ice", 1.0, 1.0, 2.5, rho0, fDT=1.0, solid=True, B0=0.005 * C.MBAR)


def DT_gas(rho0=3.0e-4):
    return Material("DT-gas", 1.0, 1.0, 2.5, rho0, fDT=1.0, solid=False)


def CH(rho0=1.05, opac_mult=1.0):
    # C:H = 1:1  ->  <Z> = 3.5, <Z^2> = 18.5, <A> = 6.5
    return Material("CH", 3.5, 18.5, 6.5, rho0, fDT=0.0, solid=True, B0=0.05 * C.MBAR, opac_mult=opac_mult)


def CH_foam_DT(rho_foam=0.18, rho_DT=0.25):
    """DT-wetted CH foam (NRL-type KrF ablator).  Mixture by number of ions."""
    nCH = rho_foam / (6.5 * C.mu)      # 'CH' pseudo-ions (one C or one H each)
    nDT = rho_DT / (2.5 * C.mu) * (1.0 - rho_foam / 1.05)
    n = nCH + nDT
    fCH, fDT = nCH / n, nDT / n
    Z = fCH * 3.5 + fDT * 1.0
    Z2 = fCH * 18.5 + fDT * 1.0
    A = fCH * 6.5 + fDT * 2.5
    rho0 = n * A * C.mu
    return Material("CH-foam+DT", Z, Z2, A, rho0, fDT=fDT, solid=True, B0=0.01 * C.MBAR)


def ideal(name, Z, A, rho0, Z2=None):
    """Generic ideal fully ionized gas (used by benchmarks)."""
    return Material(name, Z, Z if Z2 is None else Z2, A, rho0)
