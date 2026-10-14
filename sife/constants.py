"""Physical constants (CGS). Temperatures are carried in eV throughout SIFE."""
import math

c = 2.99792458e10          # cm/s
e = 4.80320471e-10         # statC
me = 9.1093837e-28         # g
mp = 1.67262192e-24        # g
mu = 1.66053907e-24        # g (atomic mass unit)
kB = 1.380649e-16          # erg/K
hbar = 1.054571817e-27     # erg s
eV = 1.602176634e-12       # erg
keV = 1.0e3 * eV
MeV = 1.0e6 * eV
K_per_eV = 11604.518

# radiation constant in erg cm^-3 eV^-4
a_rad = 7.5657e-15 * K_per_eV ** 4
sigma_sb_eV = a_rad * c / 4.0

# Fermi energy: E_F[eV] = CF * ne^(2/3), ne in cm^-3
CF = hbar ** 2 / (2.0 * me) * (3.0 * math.pi ** 2) ** (2.0 / 3.0) / eV

# DT fusion
E_alpha = 3.5 * MeV
E_neutron = 14.1 * MeV
E_fusion = E_alpha + E_neutron
m_alpha = 4.001506 * mu
Z_alpha = 2.0

# Kramers free-free Rosseland/Planck mean ratio (with stimulated emission), computed numerically
FF_ROSS_OVER_PLANCK = 1.0 / 30.262

MBAR = 1.0e12  # erg/cm^3
