"""Implicit three-temperature (Te, Ti, Er) diffusion / coupling step.

Solves, per zone k of mass m and volume V (Lagrangian, after the hydro step):

  m Cve dTe/dt = div(kappa_e grad Te) - V w_ei (Te-Ti) - V c kP (a Te^4 - Er) + P_laser
  m Cvi dTi/dt = div(kappa_i grad Ti) + V w_ei (Te-Ti)                       + S_i
  V dEr/dt     = div(D_r grad Er)                 + V c kP (a Te^4 - Er)

backward-Euler, with a Te^4 linearized about the latest iterate and the
coefficients (conductivities, opacities, exchange rate, flux limiters)
lagged in Picard iterations.  The coupled system is block-tridiagonal (3x3
blocks) and is solved directly.

Absorption uses the Planck mean at Te times a spectral correction G(Te, Tr)
(free-free absorption of a Planck spectrum at the radiation temperature), so
hard hot-spot / coronal x-rays are not over-absorbed by cold dense fuel, a
known deficiency of single-temperature gray diffusion.  Emission is the
Planck mean at Te; the two coincide at equilibrium.

Flux limiting:
  electrons: harmonic limiter  q = q_SH / (1 + |q_SH| / (f q_fs)),  q_fs = n_e T_e v_te
  radiation: Wilson sum limiter D = c / (3 kR + |grad Er| / Er)
Outer radiation boundary: vacuum (Marshak) F = c Er_b / 2.
"""
import math

import numpy as np
from numba import njit

from .constants import a_rad, c
from .hydro import area, volumes
from .transport import (coulomb_log, exchange_coef, kappa_e, kappa_i,
                        opacities, q_free_stream, spectral_factor)


@njit(cache=True)
def _solve3(M, B):
    """Solve M X = B for 3x3 M and 3xK B with partial pivoting (copies)."""
    A = M.copy()
    X = B.copy()
    K = X.shape[1]
    for col in range(3):
        piv = col
        amax = abs(A[col, col])
        for rr in range(col + 1, 3):
            if abs(A[rr, col]) > amax:
                amax = abs(A[rr, col])
                piv = rr
        if piv != col:
            for cc in range(3):
                tmp = A[col, cc]; A[col, cc] = A[piv, cc]; A[piv, cc] = tmp
            for cc in range(K):
                tmp = X[col, cc]; X[col, cc] = X[piv, cc]; X[piv, cc] = tmp
        for rr in range(col + 1, 3):
            f = A[rr, col] / A[col, col]
            if f != 0.0:
                for cc in range(col, 3):
                    A[rr, cc] -= f * A[col, cc]
                for cc in range(K):
                    X[rr, cc] -= f * X[col, cc]
    for col in range(2, -1, -1):
        for cc in range(K):
            s = X[col, cc]
            for j in range(col + 1, 3):
                s -= A[col, j] * X[j, cc]
            X[col, cc] = s / A[col, col]
    return X


@njit(cache=True)
def block_tridiag(Lo, Dg, Up, b):
    """Lo, Up: (N,3) diagonal off-blocks; Dg: (N,3,3); b: (N,3)."""
    N = b.shape[0]
    Cp = np.zeros((N, 3, 3))
    dp = np.zeros((N, 3))
    for k in range(N):
        M = Dg[k].copy()
        rhs = np.zeros((3, 4))
        for i in range(3):
            rhs[i, i] = Up[k, i]
            rhs[i, 3] = b[k, i]
        if k > 0:
            for i in range(3):
                for j in range(3):
                    M[i, j] -= Lo[k, i] * Cp[k - 1, i, j]
                rhs[i, 3] -= Lo[k, i] * dp[k - 1, i]
        X = _solve3(M, rhs)
        for i in range(3):
            for j in range(3):
                Cp[k, i, j] = X[i, j]
            dp[k, i] = X[i, 3]
    x = np.zeros((N, 3))
    for i in range(3):
        x[N - 1, i] = dp[N - 1, i]
    for k in range(N - 2, -1, -1):
        for i in range(3):
            s = dp[k, i]
            for j in range(3):
                s -= Cp[k, i, j] * x[k + 1, j]
            x[k, i] = s
    return x


@njit(cache=True)
def diffusion_step(dt, geom, r, m, rho, Te0, Ti0, Er0, Cve, Cvi,
                   Zb, Z2b, Ab_mu, opac_mult, Pdep, Sion,
                   flim, cond_e, cond_i, exch, rad, rad_outer_vacuum,
                   kappa_mode, kappa0, kappa_n, n_iter, Tfloor,
                   Te, Ti, Er, spectral):
    """Returns radiation energy lost through the outer boundary during dt."""
    N = rho.shape[0]
    V = np.empty(N)
    volumes(r, geom, V)
    rc = np.empty(N)
    for k in range(N):
        rc[k] = 0.5 * (r[k] + r[k + 1])
    ne = np.empty(N)
    ni = np.empty(N)
    Zeff = np.empty(N)
    for k in range(N):
        ni[k] = rho[k] / Ab_mu[k]
        ne[k] = Zb[k] * ni[k]
        Zeff[k] = Z2b[k] / Zb[k]
        Te[k] = Te0[k]
        Ti[k] = Ti0[k]
        Er[k] = Er0[k]
    Ge = np.zeros(N + 1)
    Gi = np.zeros(N + 1)
    Gr = np.zeros(N + 1)
    Gb = 0.0
    kP = np.zeros(N)
    kA = np.zeros(N)
    kR = np.zeros(N)
    om = np.zeros(N)
    Lo = np.zeros((N, 3))
    Up = np.zeros((N, 3))
    Dg = np.zeros((N, 3, 3))
    b = np.zeros((N, 3))
    for it in range(n_iter):
        # ---- coefficients at latest iterate
        for k in range(N):
            lnL = coulomb_log(ne[k], Te[k], Zeff[k])
            om[k] = exchange_coef(ne[k], Te[k], Zeff[k], Ab_mu[k], lnL) if exch else 0.0
            if rad:
                kp, kr = opacities(ne[k], ni[k], Te[k], Z2b[k], opac_mult[k])
                kP[k] = kp
                if spectral:
                    Tr = (max(Er[k], 1e-300) / a_rad) ** 0.25
                    G = spectral_factor(Te[k], Tr)
                    kA[k] = kp * G
                    if G < 1.0:
                        kr = (kr - 0.6652e-24 * ne[k]) * G + 0.6652e-24 * ne[k]
                else:
                    kA[k] = kp
                kR[k] = kr
        for j in range(1, N):
            l = j - 1
            rr = j
            A = area(r[j], geom)
            dist = rc[rr] - rc[l]
            Tf = 0.5 * (Te[l] + Te[rr])
            nef = math.sqrt(ne[l] * ne[rr])
            Zf = 0.5 * (Zeff[l] + Zeff[rr])
            if cond_e:
                if kappa_mode == 1:
                    kf = kappa0 * Tf ** kappa_n
                    Ge[j] = A * kf / dist
                else:
                    kf = kappa_e(nef, Tf, Zf, coulomb_log(nef, Tf, Zf))
                    qsh = kf * abs(Te[rr] - Te[l]) / dist
                    qfs = flim * q_free_stream(nef, Tf)
                    Ge[j] = A * kf / (1.0 + qsh / qfs) / dist
            if cond_i:
                Tif = 0.5 * (Ti[l] + Ti[rr])
                nif = math.sqrt(ni[l] * ni[rr])
                Zi = 0.5 * (Zb[l] + Zb[rr])
                Ai = 0.5 * (Ab_mu[l] + Ab_mu[rr])
                ki = kappa_i(nif, Tif, Zi, Ai, coulomb_log(nef, Tif, Zf))
                Gi[j] = A * ki / dist
            if rad:
                kRf = 0.5 * (kR[l] + kR[rr])
                Ef = 0.5 * (Er[l] + Er[rr]) + 1e-300
                D = c / (3.0 * kRf + abs(Er[rr] - Er[l]) / (dist * Ef))
                Gr[j] = A * D / dist
        if rad and rad_outer_vacuum:
            k = N - 1
            dr = r[N] - r[N - 1]
            D = c / (3.0 * kR[k] + 1.0 / (0.5 * dr))
            Gb = area(r[N], geom) / (0.5 * dr / D + 2.0 / c)
        # ---- assemble
        for k in range(N):
            for i in range(3):
                b[k, i] = 0.0
                Lo[k, i] = 0.0
                Up[k, i] = 0.0
                for j in range(3):
                    Dg[k, i, j] = 0.0
            Vk = V[k]
            # electrons
            Dg[k, 0, 0] = m[k] * Cve[k] / dt + Ge[k] + Ge[k + 1] + Vk * om[k]
            Dg[k, 0, 1] = -Vk * om[k]
            b[k, 0] = m[k] * Cve[k] * Te0[k] / dt + Pdep[k]
            Lo[k, 0] = -Ge[k]
            Up[k, 0] = -Ge[k + 1]
            # ions
            Dg[k, 1, 1] = m[k] * Cvi[k] / dt + Gi[k] + Gi[k + 1] + Vk * om[k]
            Dg[k, 1, 0] = -Vk * om[k]
            b[k, 1] = m[k] * Cvi[k] * Ti0[k] / dt + Sion[k]
            Lo[k, 1] = -Gi[k]
            Up[k, 1] = -Gi[k + 1]
            # radiation
            if rad:
                T3 = Te[k] ** 3
                cpl = Vk * c * kP[k]
                cpa = Vk * c * kA[k]
                Dg[k, 0, 0] += cpl * 4.0 * a_rad * T3
                Dg[k, 0, 2] = -cpa
                b[k, 0] += cpl * 3.0 * a_rad * T3 * Te[k]
                Dg[k, 2, 2] = Vk / dt + Gr[k] + Gr[k + 1] + cpa
                if k == N - 1:
                    Dg[k, 2, 2] += Gb
                Dg[k, 2, 0] = -cpl * 4.0 * a_rad * T3
                b[k, 2] = Vk * Er0[k] / dt - cpl * 3.0 * a_rad * T3 * Te[k]
                Lo[k, 2] = -Gr[k]
                Up[k, 2] = -Gr[k + 1]
            else:
                Dg[k, 2, 2] = 1.0
                b[k, 2] = Er0[k]
        Lo[0, 0] = 0.0; Lo[0, 1] = 0.0; Lo[0, 2] = 0.0
        x = block_tridiag(Lo, Dg, Up, b)
        for k in range(N):
            Te[k] = max(x[k, 0], Tfloor)
            Ti[k] = max(x[k, 1], Tfloor)
            Er[k] = max(x[k, 2], 0.0)
    lost = 0.0
    if rad and rad_outer_vacuum:
        lost = Gb * Er[N - 1] * dt
    return lost
