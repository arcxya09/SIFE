"""Implicit three-temperature diffusion on the 2D (z, R) Lagrangian mesh.

Same physics as the 1D `diffusion.py` (flux-limited Spitzer electron
conduction, Braginskii ion conduction, e-i exchange, gray flux-limited
radiation diffusion with emission/absorption, Marshak vacuum boundary).

Time integration uses dimensional (Lie) operator splitting, as in many 2D
ICF codes:
  1. angular sweep: for each ring i, backward-Euler diffusion of Te, Ti and Er
     separately across the angular faces (scalar tridiagonal per ring);
  2. radial sweep: for each angular column j, the fully coupled 3T backward-
     Euler system (radial diffusion + exchange + emission/absorption + laser
     and alpha sources), with Picard iterations on the coefficients, solved as a
     3x3 block-tridiagonal system exactly like the 1D code.
Both sweeps are unconditionally stable and conservative; columns and rings run
in parallel threads.

Face fluxes use two-point differences between zone centroids,
F = A_f k_f (T_a - T_b) / d_ab, with A_f the true (rotated) face area.
"""
import math

import numpy as np
from numba import njit, prange

from .constants import a_rad, c
from .diffusion import block_tridiag
from .transport import (coulomb_log, exchange_coef, kappa_e, kappa_i,
                        opacities, q_free_stream, spectral_factor)

TWO_PI = 2.0 * math.pi


@njit(cache=True)
def face_geometry(z, R, zc, Rc, Arad, drad, Aang, dang):
    """Arad[i, j]: area of the face at node row i of column j (i = 0 .. Ni; i = 0 is
    the origin, i = Ni the outer surface); drad[i, j]: centroid distance across it
    (for i = Ni: centroid to face midpoint).  Aang[i, j]: area of the face at node
    column j of ring i (j = 0 .. Nj; ends are axis / symmetry plane), dang likewise."""
    Ni, Nj = zc.shape
    for j in range(Nj):
        for i in range(Ni + 1):
            za = z[i, j]; Ra = R[i, j]; zb = z[i, j + 1]; Rb = R[i, j + 1]
            L = math.sqrt((zb - za) ** 2 + (Rb - Ra) ** 2)
            Arad[i, j] = math.pi * (Ra + Rb) * L
            if i == 0:
                Arad[i, j] = 0.0
                drad[i, j] = 1.0
            elif i == Ni:
                drad[i, j] = math.sqrt((0.5 * (za + zb) - zc[i - 1, j]) ** 2 + (0.5 * (Ra + Rb) - Rc[i - 1, j]) ** 2)
            else:
                drad[i, j] = math.sqrt((zc[i, j] - zc[i - 1, j]) ** 2 + (Rc[i, j] - Rc[i - 1, j]) ** 2)
    for i in range(Ni):
        for j in range(Nj + 1):
            za = z[i, j]; Ra = R[i, j]; zb = z[i + 1, j]; Rb = R[i + 1, j]
            L = math.sqrt((zb - za) ** 2 + (Rb - Ra) ** 2)
            if j == 0 or j == Nj:
                Aang[i, j] = 0.0
                dang[i, j] = 1.0
            else:
                Aang[i, j] = math.pi * (Ra + Rb) * L
                dang[i, j] = math.sqrt((zc[i, j] - zc[i, j - 1]) ** 2 + (Rc[i, j] - Rc[i, j - 1]) ** 2)


@njit(cache=True)
def _thomas(a, b, cc, d, x):
    """Solve a_k x_{k-1} + b_k x_k + c_k x_{k+1} = d_k."""
    n = b.shape[0]
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = cc[0] / b[0]
    dp[0] = d[0] / b[0]
    for k in range(1, n):
        den = b[k] - a[k] * cp[k - 1]
        cp[k] = cc[k] / den
        dp[k] = (d[k] - a[k] * dp[k - 1]) / den
    x[n - 1] = dp[n - 1]
    for k in range(n - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]


@njit(cache=True)
def _zone_coefs(ne, ni, Te, Er, Z2b, opac, rad, spectral):
    kp, kr = opacities(ne, ni, Te, Z2b, opac)
    ka = kp
    if rad and spectral:
        Tr = (max(Er, 1e-300) / a_rad) ** 0.25
        G = spectral_factor(Te, Tr)
        ka = kp * G
        if G < 1.0:
            kr = (kr - 0.6652e-24 * ne) * G + 0.6652e-24 * ne
    return kp, ka, kr


@njit(parallel=True, cache=True)
def angular_sweep(dt, V, m, rho, Te, Ti, Er, Cve, Cvi, Zb, Z2b, Ab_mu, opac,
                  Aang, dang, flim, cond_e, cond_i, rad, spectral, Tfloor, kappa_mode, kappa0, kappa_n):
    """In-place implicit angular diffusion of Te, Ti, Er (each ring independently)."""
    Ni, Nj = V.shape
    for i in prange(Ni):
        ne = np.empty(Nj); ni = np.empty(Nj); Zeff = np.empty(Nj); kR = np.empty(Nj)
        for j in range(Nj):
            ni[j] = rho[i, j] / Ab_mu[i, j]
            ne[j] = Zb[i, j] * ni[j]
            Zeff[j] = Z2b[i, j] / Zb[i, j]
            if rad:
                kR[j] = _zone_coefs(ne[j], ni[j], Te[i, j], Er[i, j], Z2b[i, j], opac[i, j], rad, spectral)[2]
        a = np.zeros(Nj); b = np.zeros(Nj); cc = np.zeros(Nj); d = np.zeros(Nj); x = np.zeros(Nj)
        G = np.zeros(Nj + 1)
        for field in range(3):
            if field == 0 and not cond_e:
                continue
            if field == 1 and not cond_i:
                continue
            if field == 2 and not rad:
                continue
            for j in range(1, Nj):
                l = j - 1
                A = Aang[i, j]
                dist = dang[i, j]
                if field == 0:
                    Tf = 0.5 * (Te[i, l] + Te[i, j])
                    if kappa_mode == 1:
                        G[j] = A * kappa0 * Tf ** kappa_n / dist
                    else:
                        nef = math.sqrt(ne[l] * ne[j])
                        Zf = 0.5 * (Zeff[l] + Zeff[j])
                        kf = kappa_e(nef, Tf, Zf, coulomb_log(nef, Tf, Zf))
                        qsh = kf * abs(Te[i, j] - Te[i, l]) / dist
                        qfs = flim * q_free_stream(nef, Tf)
                        G[j] = A * kf / (1.0 + qsh / qfs) / dist
                elif field == 1:
                    Tif = 0.5 * (Ti[i, l] + Ti[i, j])
                    nef = math.sqrt(ne[l] * ne[j])
                    nif = math.sqrt(ni[l] * ni[j])
                    Zf = 0.5 * (Zeff[l] + Zeff[j])
                    Zi = 0.5 * (Zb[i, l] + Zb[i, j])
                    Ai = 0.5 * (Ab_mu[i, l] + Ab_mu[i, j])
                    G[j] = A * kappa_i(nif, Tif, Zi, Ai, coulomb_log(nef, Tif, Zf)) / dist
                else:
                    kRf = 0.5 * (kR[l] + kR[j])
                    Ef = 0.5 * (Er[i, l] + Er[i, j]) + 1e-300
                    D = c / (3.0 * kRf + abs(Er[i, j] - Er[i, l]) / (dist * Ef))
                    G[j] = A * D / dist
            G[0] = 0.0
            G[Nj] = 0.0
            for j in range(Nj):
                if field == 0:
                    cap = m[i, j] * Cve[i, j] / dt
                    x0 = Te[i, j]
                elif field == 1:
                    cap = m[i, j] * Cvi[i, j] / dt
                    x0 = Ti[i, j]
                else:
                    cap = V[i, j] / dt
                    x0 = Er[i, j]
                a[j] = -G[j]
                cc[j] = -G[j + 1]
                b[j] = cap + G[j] + G[j + 1]
                d[j] = cap * x0
            _thomas(a, b, cc, d, x)
            for j in range(Nj):
                if field == 0:
                    Te[i, j] = max(x[j], Tfloor)
                elif field == 1:
                    Ti[i, j] = max(x[j], Tfloor)
                else:
                    Er[i, j] = max(x[j], 0.0)


@njit(cache=True)
def line_3t(dt, V, m, Af, df, rho, Te0, Ti0, Er0, Cve, Cvi, Zb, Z2b, Ab_mu, opac,
            Pdep, Sion, flim, cond_e, cond_i, exch, rad, n_iter, Tfloor, spectral,
            Te, Ti, Er, kappa_mode, kappa0, kappa_n):
    """Coupled 3T backward-Euler solve along one radial column.  Af[0..N] are face
    areas (Af[0] inner = 0, Af[N] outer vacuum boundary), df the matching
    distances.  Returns the radiation energy lost through the outer face."""
    N = V.shape[0]
    ne = np.empty(N); ni = np.empty(N); Zeff = np.empty(N)
    for k in range(N):
        ni[k] = rho[k] / Ab_mu[k]
        ne[k] = Zb[k] * ni[k]
        Zeff[k] = Z2b[k] / Zb[k]
        Te[k] = Te0[k]; Ti[k] = Ti0[k]; Er[k] = Er0[k]
    Ge = np.zeros(N + 1); Gi = np.zeros(N + 1); Gr = np.zeros(N + 1)
    Gb = 0.0
    kP = np.zeros(N); kA = np.zeros(N); kR = np.zeros(N); om = np.zeros(N)
    Lo = np.zeros((N, 3)); Up = np.zeros((N, 3)); Dg = np.zeros((N, 3, 3)); b = np.zeros((N, 3))
    for it in range(n_iter):
        for k in range(N):
            lnL = coulomb_log(ne[k], Te[k], Zeff[k])
            om[k] = exchange_coef(ne[k], Te[k], Zeff[k], Ab_mu[k], lnL) if exch else 0.0
            if rad:
                kp, ka, kr = _zone_coefs(ne[k], ni[k], Te[k], Er[k], Z2b[k], opac[k], rad, spectral)
                kP[k] = kp; kA[k] = ka; kR[k] = kr
        for j in range(1, N):
            l = j - 1
            A = Af[j]
            dist = df[j]
            Tf = 0.5 * (Te[l] + Te[j])
            nef = math.sqrt(ne[l] * ne[j])
            Zf = 0.5 * (Zeff[l] + Zeff[j])
            if cond_e:
                if kappa_mode == 1:
                    Ge[j] = A * kappa0 * Tf ** kappa_n / dist
                else:
                    kf = kappa_e(nef, Tf, Zf, coulomb_log(nef, Tf, Zf))
                    qsh = kf * abs(Te[j] - Te[l]) / dist
                    qfs = flim * q_free_stream(nef, Tf)
                    Ge[j] = A * kf / (1.0 + qsh / qfs) / dist
            if cond_i:
                Tif = 0.5 * (Ti[l] + Ti[j])
                nif = math.sqrt(ni[l] * ni[j])
                Zi = 0.5 * (Zb[l] + Zb[j])
                Ai = 0.5 * (Ab_mu[l] + Ab_mu[j])
                Gi[j] = A * kappa_i(nif, Tif, Zi, Ai, coulomb_log(nef, Tif, Zf)) / dist
            if rad:
                kRf = 0.5 * (kR[l] + kR[j])
                Ef = 0.5 * (Er[l] + Er[j]) + 1e-300
                D = c / (3.0 * kRf + abs(Er[j] - Er[l]) / (dist * Ef))
                Gr[j] = A * D / dist
        if rad:
            D = c / (3.0 * kR[N - 1] + 1.0 / df[N])
            Gb = Af[N] / (df[N] / D + 2.0 / c)
        for k in range(N):
            for i in range(3):
                b[k, i] = 0.0; Lo[k, i] = 0.0; Up[k, i] = 0.0
                for j in range(3):
                    Dg[k, i, j] = 0.0
            Vk = V[k]
            Dg[k, 0, 0] = m[k] * Cve[k] / dt + Ge[k] + Ge[k + 1] + Vk * om[k]
            Dg[k, 0, 1] = -Vk * om[k]
            b[k, 0] = m[k] * Cve[k] * Te0[k] / dt + Pdep[k]
            Lo[k, 0] = -Ge[k]; Up[k, 0] = -Ge[k + 1]
            Dg[k, 1, 1] = m[k] * Cvi[k] / dt + Gi[k] + Gi[k + 1] + Vk * om[k]
            Dg[k, 1, 0] = -Vk * om[k]
            b[k, 1] = m[k] * Cvi[k] * Ti0[k] / dt + Sion[k]
            Lo[k, 1] = -Gi[k]; Up[k, 1] = -Gi[k + 1]
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
                Lo[k, 2] = -Gr[k]; Up[k, 2] = -Gr[k + 1]
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
    if rad:
        lost = Gb * Er[N - 1] * dt
    return lost


@njit(parallel=True, cache=True)
def radial_sweep(dt, V, m, rho, Te, Ti, Er, Cve, Cvi, Zb, Z2b, Ab_mu, opac, Pdep, Sion,
                 Arad, drad, flim, cond_e, cond_i, exch, rad, n_iter, Tfloor, spectral, lost,
                 kappa_mode, kappa0, kappa_n):
    Ni, Nj = V.shape
    for j in prange(Nj):
        Te1 = np.empty(Ni); Ti1 = np.empty(Ni); Er1 = np.empty(Ni)
        lost[j] = line_3t(dt, V[:, j].copy(), m[:, j].copy(), Arad[:, j].copy(), drad[:, j].copy(),
                          rho[:, j].copy(), Te[:, j].copy(), Ti[:, j].copy(), Er[:, j].copy(),
                          Cve[:, j].copy(), Cvi[:, j].copy(), Zb[:, j].copy(), Z2b[:, j].copy(),
                          Ab_mu[:, j].copy(), opac[:, j].copy(), Pdep[:, j].copy(), Sion[:, j].copy(),
                          flim, cond_e, cond_i, exch, rad, n_iter, Tfloor, spectral, Te1, Ti1, Er1,
                          kappa_mode, kappa0, kappa_n)
        for i in range(Ni):
            Te[i, j] = Te1[i]
            Ti[i, j] = Ti1[i]
            Er[i, j] = Er1[i]


def diffusion2d_step(dt, geo, m, rho, Te, Ti, Er, Cve, Cvi, Zb, Z2b, Ab_mu, opac, Pdep, Sion,
                     flim, cond_e, cond_i, exch, rad, n_iter, Tfloor, spectral,
                     kappa_mode=0, kappa0=0.0, kappa_n=0.0):
    """Te, Ti, Er (2D arrays) are updated in place.  geo: dict with V, Arad, drad,
    Aang, dang.  Returns the radiation energy lost through the outer surface."""
    angular_sweep(dt, geo["V"], m, rho, Te, Ti, Er, Cve, Cvi, Zb, Z2b, Ab_mu, opac,
                  geo["Aang"], geo["dang"], flim, cond_e, cond_i, rad, spectral, Tfloor,
                  kappa_mode, kappa0, kappa_n)
    lost = np.zeros(rho.shape[1])
    radial_sweep(dt, geo["V"], m, rho, Te, Ti, Er, Cve, Cvi, Zb, Z2b, Ab_mu, opac, Pdep, Sion,
                 geo["Arad"], geo["drad"], flim, cond_e, cond_i, exch, rad, n_iter, Tfloor,
                 spectral, lost, kappa_mode, kappa0, kappa_n)
    return float(lost.sum())
