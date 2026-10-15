"""2D axisymmetric Godunov ALE hydrodynamics on a spherical-polar grid.

Grid: rings of radius r_i(t) (the same for every polar angle) and fixed spokes
theta_j.  Every cell is an exact spherical sector, so the geometry has no zero-energy
(hourglass) modes and needs no artificial stiffness.  The rings move radially with
the ring-averaged mass velocity, which keeps the 1D Lagrangian zoning (the shell
stays resolved through a 20-30x convergence); fluid crosses the rings only through
departures from spherical symmetry.  All angular motion is Eulerian.

Scheme (finite volume, cell-centred):
  * second-order MUSCL reconstruction (van Leer limiter) of the primitive variables
    in r and theta,
  * HLLC Riemann solver; on a moving face the flux is taken in the state at
    x/t = w (Godunov ALE flux F(U) - w U),
  * Heun (RK2) time stepping with the face areas of the swept volumes from t^n to
    t^{n+1}, so the geometric conservation law holds exactly,
  * well-balanced geometric sources P (A_out - A_in), P (S_j+1 - S_j) and the
    centrifugal / Coriolis terms rho u_t^2 / r, -rho u_r u_t / r.

Three temperatures: total energy is conserved; electron and radiation specific
energies are advected and receive -p dV with the Riemann contact velocities, so
shock heating goes to the ions.  A separately advected ion energy (pdV only) is used
outside shocks (dual-energy formalism), which keeps the entropy of the cold, fast
imploding shell free of the kinetic-energy truncation error.

Materials are carried as mass fractions X_k (mixed cells at interfaces); the DT
number per gram y is a passive scalar.
"""
import math

import numpy as np
from numba import njit, prange

from .eos import cold_curve

TWO_PI = 2.0 * math.pi

# primitive variable slots in Q[q, i, j]
QR, QUR, QUT, QP, QEE, QEI, QER, QEC, QY, QX = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
# conserved / rate slots in U[q, i, j]
UM, UMR, UMT, UE, UEE, UEI, UER, UY, UX = 0, 1, 2, 3, 4, 5, 6, 7, 8


def n_prim(nmat):
    return QX + nmat


def n_cons(nmat):
    return UX + nmat


@njit(cache=True)
def ring_geometry(rn0, rn1, theta, V0, V1, Aeff, Seff, rc):
    """Volumes at t^n and t^n+1, swept-volume face areas of the rings Aeff[i, j]
    (Aeff * (r1 - r0) = exact swept volume), spoke face areas Seff[i, j] at the
    time-averaged radii and cell centroid radii rc (mid geometry)."""
    Ni = V0.shape[0]
    Nj = V0.shape[1]
    for j in range(Nj):
        dmu = math.cos(theta[j]) - math.cos(theta[j + 1])
        for i in range(Ni):
            V0[i, j] = TWO_PI / 3.0 * (rn0[i + 1] ** 3 - rn0[i] ** 3) * dmu
            V1[i, j] = TWO_PI / 3.0 * (rn1[i + 1] ** 3 - rn1[i] ** 3) * dmu
        for i in range(Ni + 1):
            Aeff[i, j] = TWO_PI / 3.0 * (rn0[i] ** 2 + rn0[i] * rn1[i] + rn1[i] ** 2) * dmu
    for i in range(Ni):
        a2 = 0.5 * (rn0[i] ** 2 + rn1[i] ** 2)
        b2 = 0.5 * (rn0[i + 1] ** 2 + rn1[i + 1] ** 2)
        for j in range(Nj + 1):
            Seff[i, j] = math.pi * math.sin(theta[j]) * (b2 - a2)
        ra = 0.5 * (rn0[i] + rn1[i])
        rb = 0.5 * (rn0[i + 1] + rn1[i + 1])
        c = 0.75 * (rb ** 4 - ra ** 4) / max(rb ** 3 - ra ** 3, 1e-300)
        for j in range(Nj):
            rc[i, j] = c


@njit(cache=True)
def _vl(a, b, c):
    d1 = b - a
    d2 = c - b
    if d1 * d2 <= 0.0:
        return 0.0
    return 2.0 * d1 * d2 / (d1 + d2)


@njit(cache=True)
def _slopes(Q, half, core, second, Gr, Gt):
    """Limited half-cell increments in r (Gr) and theta (Gt) of every primitive.
    Axis / equator neighbours are mirrors (u_theta changes sign); the outer radial
    neighbour is a copy (outflow); the centre has no inward neighbour."""
    nq, Ni, Nj = Q.shape
    for q in range(nq):
        for i in range(Ni):
            for j in range(Nj):
                Gr[q, i, j] = 0.0
                Gt[q, i, j] = 0.0
    if not second:
        return
    for q in range(nq):
        sg = -1.0 if q == QUT else 1.0
        for i in range(1, Ni - 1):
            for j in range(Nj):
                Gr[q, i, j] = 0.5 * _vl(Q[q, i - 1, j], Q[q, i, j], Q[q, i + 1, j])
        for i in range(core, Ni):
            for j in range(Nj):
                a = Q[q, i, j - 1] if j > 0 else sg * Q[q, i, 0]
                if j < Nj - 1:
                    c = Q[q, i, j + 1]
                else:
                    c = sg * Q[q, i, Nj - 1]     # equator mirror (half) or far axis mirror (full)
                Gt[q, i, j] = 0.5 * _vl(a, Q[q, i, j], c)


@njit(cache=True)
def _hllc(rL, unL, utL, pL, eL, cL, rR, unR, utR, pR, eR, cR, w, out):
    """HLLC flux through a face moving with normal speed w.
    e = specific total energy (internal + kinetic).  out = [F_mass, F_mom_n,
    F_mom_t, F_E, S*, side] with side = 0 (left state upwind) or 1 (right)."""
    SL = min(unL - cL, unR - cR)
    SR = max(unL + cL, unR + cR)
    den = rL * (SL - unL) - rR * (SR - unR)
    if abs(den) < 1e-300:
        Ss = 0.5 * (unL + unR)
    else:
        Ss = (pR - pL + rL * unL * (SL - unL) - rR * unR * (SR - unR)) / den
    EL = rL * eL
    ER = rR * eR
    if w <= SL:
        out[0] = rL * unL - w * rL
        out[1] = rL * unL * unL + pL - w * rL * unL
        out[2] = rL * unL * utL - w * rL * utL
        out[3] = (EL + pL) * unL - w * EL
        side = 0
    elif w >= SR:
        out[0] = rR * unR - w * rR
        out[1] = rR * unR * unR + pR - w * rR * unR
        out[2] = rR * unR * utR - w * rR * utR
        out[3] = (ER + pR) * unR - w * ER
        side = 1
    elif w <= Ss:
        f = rL * (SL - unL) / (SL - Ss)
        Es = f * (eL + (Ss - unL) * (Ss + pL / (rL * (SL - unL))))
        # F*L = F_L + SL (U*L - U_L);  ALE flux = F*L - w U*L
        out[0] = rL * unL + SL * (f - rL) - w * f
        out[1] = rL * unL * unL + pL + SL * (f * Ss - rL * unL) - w * f * Ss
        out[2] = rL * unL * utL + SL * (f * utL - rL * utL) - w * f * utL
        out[3] = (EL + pL) * unL + SL * (Es - EL) - w * Es
        side = 0
    else:
        f = rR * (SR - unR) / (SR - Ss)
        Es = f * (eR + (Ss - unR) * (Ss + pR / (rR * (SR - unR))))
        out[0] = rR * unR + SR * (f - rR) - w * f
        out[1] = rR * unR * unR + pR + SR * (f * Ss - rR * unR) - w * f * Ss
        out[2] = rR * unR * utR + SR * (f * utR - rR * utR) - w * f * utR
        out[3] = (ER + pR) * unR + SR * (Es - ER) - w * Es
        side = 1
    out[4] = Ss
    out[5] = side


@njit(parallel=True, cache=True)
def ale_rates(Q, c, Pe, Pi, Pr, w, Aeff, Seff, V, rc, half, core, second, grav, wall, L, div, flux_out):
    """Time derivatives L[q, i, j] of the conserved cell totals (mass, radial and
    polar momentum, total energy, electron / pdV-ion / radiation energy, DT number,
    material masses).  div[i, j] = sum over faces of A * u_contact (Lagrangian
    volume rate).  flux_out[q] = rate leaving through the outer boundary, which is
    a vacuum free surface, or a reflecting wall if wall is True."""
    nq, Ni, Nj = Q.shape
    nu = L.shape[0]
    nmat = nu - UX
    Gr = np.empty_like(Q)
    Gt = np.empty_like(Q)
    _slopes(Q, half, core, second, Gr, Gt)
    for q in range(nu):
        for i in range(Ni):
            for j in range(Nj):
                L[q, i, j] = 0.0
    for i in range(Ni):
        for j in range(Nj):
            div[i, j] = 0.0
    for q in range(nu):
        flux_out[q] = 0.0
    # ---- radial faces: face i between cells i-1 and i (i = 1..Ni); one column per thread
    FR = np.zeros((Ni + 1, Nj, nu))
    SR_ = np.zeros((Ni + 1, Nj))
    for j in prange(Nj):
        out = np.empty(6)
        for i in range(1, Ni + 1):
            iL = i - 1
            rL = Q[QR, iL, j] + Gr[QR, iL, j]
            uL = Q[QUR, iL, j] + Gr[QUR, iL, j]
            tL = Q[QUT, iL, j] + Gr[QUT, iL, j]
            pL = Q[QP, iL, j] + Gr[QP, iL, j]
            eiL = (Q[QEE, iL, j] + Gr[QEE, iL, j] + Q[QEI, iL, j] + Gr[QEI, iL, j]
                   + Q[QER, iL, j] + Gr[QER, iL, j] + Q[QEC, iL, j])
            if rL <= 0.0 or pL <= 0.0 or eiL <= 0.0:
                rL = Q[QR, iL, j]; uL = Q[QUR, iL, j]; tL = Q[QUT, iL, j]; pL = Q[QP, iL, j]
                eiL = Q[QEE, iL, j] + Q[QEI, iL, j] + Q[QER, iL, j] + Q[QEC, iL, j]
            if i < Ni:
                iR = i
                rR = Q[QR, iR, j] - Gr[QR, iR, j]
                uR = Q[QUR, iR, j] - Gr[QUR, iR, j]
                tR = Q[QUT, iR, j] - Gr[QUT, iR, j]
                pR = Q[QP, iR, j] - Gr[QP, iR, j]
                eiR = (Q[QEE, iR, j] - Gr[QEE, iR, j] + Q[QEI, iR, j] - Gr[QEI, iR, j]
                       + Q[QER, iR, j] - Gr[QER, iR, j] + Q[QEC, iR, j])
                if rR <= 0.0 or pR <= 0.0 or eiR <= 0.0:
                    rR = Q[QR, iR, j]; uR = Q[QUR, iR, j]; tR = Q[QUT, iR, j]; pR = Q[QP, iR, j]
                    eiR = Q[QEE, iR, j] + Q[QEI, iR, j] + Q[QER, iR, j] + Q[QEC, iR, j]
                cR = c[iR, j]
            else:                       # outer boundary: free surface against vacuum (P = 0)
                iR = iL
                cR = c[iL, j]
                rR = rL; uR = uL; tR = tL; pR = pL; eiR = eiL
            if i == Ni and wall:
                uR = -uL
            if i == Ni and not wall:
                dv_ = max(uL - w[i], 0.0)       # material leaving through the outer ring; nothing enters
                out[0] = rL * dv_
                out[1] = rL * uL * dv_
                out[2] = rL * tL * dv_
                out[3] = rL * (eiL + 0.5 * (uL * uL + tL * tL)) * dv_
                out[4] = uL
                out[5] = 0
            else:
                _hllc(rL, uL, tL, pL, eiL + 0.5 * (uL * uL + tL * tL), c[iL, j],
                      rR, uR, tR, pR, eiR + 0.5 * (uR * uR + tR * tR), cR, w[i], out)
            FR[i, j, UM] = out[0]
            FR[i, j, UMR] = out[1]
            FR[i, j, UMT] = out[2]
            FR[i, j, UE] = out[3]
            k = iL if out[5] == 0 else iR
            for (qq, uu) in ((QEE, UEE), (QEI, UEI), (QER, UER), (QY, UY)):
                phi = Q[qq, k, j] + (Gr[qq, k, j] if out[5] == 0 else -Gr[qq, k, j])
                FR[i, j, uu] = out[0] * phi
            for m in range(nmat):
                phi = Q[QX + m, k, j] + (Gr[QX + m, k, j] if out[5] == 0 else -Gr[QX + m, k, j])
                FR[i, j, UX + m] = out[0] * phi
            SR_[i, j] = out[4]
    # ---- spoke faces: face j between cells j-1 and j (j = 1..Nj-1, plus the equator if half)
    FT = np.zeros((Ni, Nj + 1, nu))
    ST_ = np.zeros((Ni, Nj + 1))
    jmax = Nj if half else Nj - 1
    for i in prange(core, Ni):
        out = np.empty(6)
        for j in range(1, jmax + 1):
            jL = j - 1
            rL = Q[QR, i, jL] + Gt[QR, i, jL]
            uL = Q[QUT, i, jL] + Gt[QUT, i, jL]      # normal = theta
            tL = Q[QUR, i, jL] + Gt[QUR, i, jL]      # tangential = r
            pL = Q[QP, i, jL] + Gt[QP, i, jL]
            eiL = (Q[QEE, i, jL] + Gt[QEE, i, jL] + Q[QEI, i, jL] + Gt[QEI, i, jL]
                   + Q[QER, i, jL] + Gt[QER, i, jL] + Q[QEC, i, jL])
            if rL <= 0.0 or pL <= 0.0 or eiL <= 0.0:
                rL = Q[QR, i, jL]; uL = Q[QUT, i, jL]; tL = Q[QUR, i, jL]; pL = Q[QP, i, jL]
                eiL = Q[QEE, i, jL] + Q[QEI, i, jL] + Q[QER, i, jL] + Q[QEC, i, jL]
            if j < Nj:
                jR = j
                rR = Q[QR, i, jR] - Gt[QR, i, jR]
                uR = Q[QUT, i, jR] - Gt[QUT, i, jR]
                tR = Q[QUR, i, jR] - Gt[QUR, i, jR]
                pR = Q[QP, i, jR] - Gt[QP, i, jR]
                eiR = (Q[QEE, i, jR] - Gt[QEE, i, jR] + Q[QEI, i, jR] - Gt[QEI, i, jR]
                       + Q[QER, i, jR] - Gt[QER, i, jR] + Q[QEC, i, jR])
                if rR <= 0.0 or pR <= 0.0 or eiR <= 0.0:
                    rR = Q[QR, i, jR]; uR = Q[QUT, i, jR]; tR = Q[QUR, i, jR]; pR = Q[QP, i, jR]
                    eiR = Q[QEE, i, jR] + Q[QEI, i, jR] + Q[QER, i, jR] + Q[QEC, i, jR]
                cR = c[i, jR]
            else:                       # reflecting equator
                jR = jL
                rR = rL; uR = -uL; tR = tL; pR = pL; eiR = eiL
                cR = c[i, jL]
            _hllc(rL, uL, tL, pL, eiL + 0.5 * (uL * uL + tL * tL), c[i, jL],
                  rR, uR, tR, pR, eiR + 0.5 * (uR * uR + tR * tR), cR, 0.0, out)
            FT[i, j, UM] = out[0]
            FT[i, j, UMT] = out[1]
            FT[i, j, UMR] = out[2]
            FT[i, j, UE] = out[3]
            if j < Nj:
                k = jL if out[5] == 0 else jR
                sg = 1.0 if out[5] == 0 else -1.0
                for (qq, uu) in ((QEE, UEE), (QEI, UEI), (QER, UER), (QY, UY)):
                    FT[i, j, uu] = out[0] * (Q[qq, i, k] + sg * Gt[qq, i, k])
                for m in range(nmat):
                    FT[i, j, UX + m] = out[0] * (Q[QX + m, i, k] + sg * Gt[QX + m, i, k])
            ST_[i, j] = out[4]
    # ---- accumulate
    for i in prange(Ni):
        for j in range(Nj):
            Ai = Aeff[i, j]
            Ao = Aeff[i + 1, j]
            for q in range(nu):
                L[q, i, j] = Ai * FR[i, j, q] - Ao * FR[i + 1, j, q]
            dv = Ao * SR_[i + 1, j] - Ai * SR_[i, j]
            if i >= core:
                Sl = Seff[i, j]
                Sr = Seff[i, j + 1]
                for q in range(nu):
                    L[q, i, j] += Sl * FT[i, j, q] - Sr * FT[i, j + 1, q]
                dv += Sr * ST_[i, j + 1] - Sl * ST_[i, j]
            div[i, j] = dv
            rho = Q[QR, i, j]
            ur = Q[QUR, i, j]
            ut = Q[QUT, i, j]
            P = Q[QP, i, j]
            Vc = V[i, j]
            r = rc[i, j]
            # well-balanced geometric sources
            L[UMR, i, j] += P * (Ao - Ai) + rho * ut * ut * Vc / r
            if i >= core:
                L[UMT, i, j] += P * (Seff[i, j + 1] - Seff[i, j]) - rho * ur * ut * Vc / r
            if grav != 0.0:
                L[UMR, i, j] -= rho * Vc * grav
                L[UE, i, j] -= rho * Vc * grav * ur
            # p dV of electrons, ions (pdV-only copy) and radiation
            L[UEE, i, j] -= Pe[i, j] * dv
            L[UEI, i, j] -= Pi[i, j] * dv
            L[UER, i, j] -= Pr[i, j] * dv
    for j in range(Nj):
        A = Aeff[Ni, j]
        for q in range(nu):
            flux_out[q] += A * FR[Ni, j, q]


@njit(parallel=True, cache=True)
def radial_contact(Q, c, half, core, second, theta, w):
    """Ring velocities from the radial Riemann problems: w[i] = solid-angle mean
    over the columns of the HLLC contact speed S* at ring i (w[0] = 0).  With
    this choice the mesh is exactly Lagrangian in a spherically symmetric flow
    (no mass crosses a ring, material interfaces stay sharp); only the angular
    deviations of S* produce radial mass fluxes."""
    nq, Ni, Nj = Q.shape
    Gr = np.empty_like(Q)
    Gt = np.empty_like(Q)
    _slopes(Q, half, core, second, Gr, Gt)
    S = np.zeros((Ni + 1, Nj))
    for j in prange(Nj):
        for i in range(1, Ni + 1):
            iL = i - 1
            rL = Q[QR, iL, j] + Gr[QR, iL, j]
            uL = Q[QUR, iL, j] + Gr[QUR, iL, j]
            pL = Q[QP, iL, j] + Gr[QP, iL, j]
            if rL <= 0.0 or pL <= 0.0:
                rL = Q[QR, iL, j]; uL = Q[QUR, iL, j]; pL = Q[QP, iL, j]
            cL = c[iL, j]
            if i < Ni:
                rR = Q[QR, i, j] - Gr[QR, i, j]
                uR = Q[QUR, i, j] - Gr[QUR, i, j]
                pR = Q[QP, i, j] - Gr[QP, i, j]
                if rR <= 0.0 or pR <= 0.0:
                    rR = Q[QR, i, j]; uR = Q[QUR, i, j]; pR = Q[QP, i, j]
                cR = c[i, j]
            else:                       # free surface: the outer ring moves with the fluid
                S[i, j] = uL
                continue
            SL = min(uL - cL, uR - cR)
            SR = max(uL + cL, uR + cR)
            den = rL * (SL - uL) - rR * (SR - uR)
            if abs(den) < 1e-300:
                S[i, j] = 0.5 * (uL + uR)
            else:
                S[i, j] = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / den
    tot = 0.0
    for j in range(Nj):
        tot += np.cos(theta[j]) - np.cos(theta[j + 1])
    w[0] = 0.0
    for i in range(1, Ni + 1):
        a = 0.0
        for j in range(Nj):
            a += S[i, j] * (np.cos(theta[j]) - np.cos(theta[j + 1]))
        w[i] = a / tot


@njit(cache=True)
def regularize_mesh(rn, w, ratio0, qmin):
    """Keep neighbouring cell-size ratios within [qmin, 1/qmin] x their initial
    values: where a cell has become too small, the shared ring is slowed so the
    small cell is compressed no faster (relatively) than its neighbour."""
    Ni = rn.shape[0] - 1
    for i in range(1, Ni):
        a = rn[i] - rn[i - 1]
        b = rn[i + 1] - rn[i]
        q = (a / b) / ratio0[i - 1]
        if q < qmin:
            w[i] = max(w[i], (w[i - 1] * b + w[i + 1] * a) / (a + b))
        elif q > 1.0 / qmin:
            w[i] = min(w[i], (w[i + 1] * a + w[i - 1] * b) / (a + b))


@njit(cache=True)
def mesh_velocity(rn, m, ur, ratio0, qmin):
    """Ring velocities: mass-weighted average radial velocity of the two adjacent
    rings (outer face: of the last ring); w[0] = 0.

    Mesh regularisation: where the size ratio of two neighbouring cells,
    dr_i / dr_{i+1}, has dropped below qmin times its initial value (or risen
    above 1/qmin times it), the shared ring is slowed so that the smaller cell
    is compressed no faster (in relative terms) than its neighbour.  This keeps
    a stagnating cell (centre of a Noh implosion, hot spot at stagnation) from
    collapsing while the mesh stays Lagrangian everywhere else."""
    Ni, Nj = m.shape
    w = np.zeros(Ni + 1)
    for i in range(1, Ni + 1):
        num = 0.0
        den = 0.0
        for j in range(Nj):
            num += m[i - 1, j] * ur[i - 1, j]
            den += m[i - 1, j]
            if i < Ni:
                num += m[i, j] * ur[i, j]
                den += m[i, j]
        w[i] = num / max(den, 1e-300)
    if qmin > 0.0:
        regularize_mesh(rn, w, ratio0, qmin)
    return w


@njit(cache=True)
def ale_cfl(rn, rc, theta, ur, ut, c, w, core, cfl, loc):
    """Stable time step; loc = [i, j, kind] of the limiting cell
    (kind 0: converging rings, 1: radial CFL, 2: polar CFL)."""
    Ni, Nj = c.shape
    dt = 1e30
    for i in range(Ni):
        dr = rn[i + 1] - rn[i]
        dw = w[i] - w[i + 1]
        if dw > 0.0:                    # converging rings
            d = 0.3 * dr / dw
            if d < dt:
                dt = d; loc[0] = i; loc[1] = 0; loc[2] = 0
        for j in range(Nj):
            wc = 0.5 * (w[i] + w[i + 1])
            d = cfl * dr / (abs(ur[i, j] - wc) + c[i, j])
            if d < dt:
                dt = d; loc[0] = i; loc[1] = j; loc[2] = 1
            if i >= core:
                dl = rc[i, j] * (theta[j + 1] - theta[j])
                d = cfl * dl / (abs(ut[i, j]) + c[i, j])
                if d < dt:
                    dt = d; loc[0] = i; loc[1] = j; loc[2] = 2
    return dt


@njit(cache=True)
def cold_energy_cells(rho, rho0, Pd0, B0, solid, out):
    Ni, Nj = rho.shape
    for i in range(Ni):
        for j in range(Nj):
            out[i, j] = cold_curve(rho[i, j], rho0[i, j], Pd0[i, j], B0[i, j], solid[i, j])[1]
