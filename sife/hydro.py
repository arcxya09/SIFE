"""Staggered-mesh Lagrangian hydrodynamics in 1D (planar / cylindrical / spherical).

Node-centred velocities, zone-centred thermodynamics (von Neumann-Richtmyer).
Predictor-corrector time centring; momentum uses the compatible area
A_j = dV/dr_j.  Separate electron, ion and radiation internal energies:
    de_e = -P_e dV/m,   de_i = -(P_i + q) dV/m,   d(er) = -(Er/3) dV/m
(the shock-viscous heating goes to the ions, as in real shocks), while the
cold-curve energy is a function of density only.

Artificial viscosity: q = rho (C2 du^2 + C1 c_s |du|) in compressing zones.
"""
import math

import numpy as np
from numba import njit

from .eos import eos_update

FOUR_PI_3 = 4.0 * math.pi / 3.0


@njit(cache=True)
def volumes(r, geom, V):
    """geom: 0 planar (per unit area), 1 cylindrical (per unit length/rad), 2 spherical."""
    N = V.shape[0]
    for k in range(N):
        if geom == 2:
            V[k] = FOUR_PI_3 * (r[k + 1] ** 3 - r[k] ** 3)
        elif geom == 1:
            V[k] = math.pi * (r[k + 1] ** 2 - r[k] ** 2)
        else:
            V[k] = r[k + 1] - r[k]


@njit(cache=True)
def area(rj, geom):
    if geom == 2:
        return 4.0 * math.pi * rj * rj
    elif geom == 1:
        return 2.0 * math.pi * rj
    return 1.0


@njit(cache=True)
def viscosity(u, rho, cs2, C1, C2, q):
    N = rho.shape[0]
    for k in range(N):
        du = u[k + 1] - u[k]
        if du < 0.0:
            q[k] = rho[k] * (C2 * du * du - C1 * math.sqrt(cs2[k]) * du)
        else:
            q[k] = 0.0


@njit(cache=True)
def hydro_step(dt, geom, r, u, m, mnode, rho, ee, ei, er,
               Zb, Ab_mu, rho0, Pd0, B0, solid, deg,
               Te, Ti, Pe, Pi, Pc, cs2, Cve, Cvi, q, C1, C2, Pext, fix_outer):
    """Advance positions, velocities and energies by dt (in place).
    Returns boundary work done on the system by Pext."""
    N = rho.shape[0]
    V0 = np.empty(N)
    volumes(r, geom, V0)
    P0 = np.empty(N)
    for k in range(N):
        P0[k] = Pe[k] + Pi[k] + Pc[k] + er[k] * rho[k] / 3.0
    # ---- predictor: half step positions with u^n
    rh = np.empty(N + 1)
    for j in range(N + 1):
        rh[j] = r[j] + 0.5 * dt * u[j]
    Vh = np.empty(N)
    volumes(rh, geom, Vh)
    rhoh = np.empty(N)
    eeh = np.empty(N)
    eih = np.empty(N)
    erh = np.empty(N)
    for k in range(N):
        rhoh[k] = m[k] / Vh[k]
        dV = Vh[k] - V0[k]
        eeh[k] = ee[k] - Pe[k] * dV / m[k]
        eih[k] = ei[k] - (Pi[k] + q[k]) * dV / m[k]
        erh[k] = er[k] - (er[k] * rho[k] / 3.0) * dV / m[k]
        if erh[k] < 0.0:
            erh[k] = 0.0
    Teh = np.empty(N); Tih = np.empty(N); Peh = np.empty(N); Pih = np.empty(N)
    Pch = np.empty(N); cs2h = np.empty(N); Cveh = np.empty(N); Cvih = np.empty(N)
    eos_update(rhoh, eeh, eih, Zb, Ab_mu, rho0, Pd0, B0, solid, deg,
               Teh, Tih, Peh, Pih, Pch, cs2h, Cveh, Cvih)
    qh = np.empty(N)
    viscosity(u, rhoh, cs2h, C1, C2, qh)
    Ph = np.empty(N)
    for k in range(N):
        Ph[k] = Peh[k] + Pih[k] + Pch[k] + erh[k] * rhoh[k] / 3.0 + qh[k]
    # ---- corrector: momentum
    u0 = u.copy()
    for j in range(N + 1):
        A = area(rh[j], geom)
        Pl = Ph[j - 1] if j > 0 else Ph[0]
        Pr = Ph[j] if j < N else Pext
        if j == 0 and geom > 0:
            u[j] = 0.0
            continue
        if j == 0 and geom == 0:
            u[j] = 0.0  # reflecting wall
            continue
        if j == N and fix_outer:
            u[j] = 0.0
            continue
        u[j] = u0[j] + dt * A * (Pl - Pr) / mnode[j]
    for j in range(N + 1):
        r[j] = r[j] + dt * 0.5 * (u0[j] + u[j])
    V1 = np.empty(N)
    volumes(r, geom, V1)
    for k in range(N):
        dV = V1[k] - V0[k]
        ee[k] = ee[k] - Peh[k] * dV / m[k]
        ei[k] = ei[k] - (Pih[k] + qh[k]) * dV / m[k]
        er[k] = er[k] - (erh[k] * rhoh[k] / 3.0) * dV / m[k]
        if er[k] < 0.0:
            er[k] = 0.0
        rho[k] = m[k] / V1[k]
        q[k] = qh[k]
    eos_update(rho, ee, ei, Zb, Ab_mu, rho0, Pd0, B0, solid, deg,
               Te, Ti, Pe, Pi, Pc, cs2, Cve, Cvi)
    Aout = area(rh[N], geom)
    return -Pext * Aout * dt * 0.5 * (u0[N] + u[N])


@njit(cache=True)
def cfl_dt(r, u, cs2, rho, er, q, C2, cfl):
    N = rho.shape[0]
    dtmin = 1.0e30
    for k in range(N):
        dr = r[k + 1] - r[k]
        du = u[k + 1] - u[k]
        c = math.sqrt(cs2[k] + 4.0 / 9.0 * er[k])
        if du < 0.0:
            c = c + 2.0 * C2 * (-du)
        dtk = cfl * dr / c
        if dtk < dtmin:
            dtmin = dtk
    return dtmin
