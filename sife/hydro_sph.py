"""Two-dimensional hydrodynamics in spherical polar coordinates (r, theta),
axisymmetric about the z axis: radially Lagrangian, angularly ALE.

Mesh: nodes (i, j) on fixed "spokes" theta_j (j = 0..Nj), at radii r[i, j]
(i = 0..Ni; i = 0 is the origin).  Zone (i, j) lies between spokes j, j+1 and
node rows i, i+1, its inner and outer boundaries interpolated linearly in theta,
so its volume is
    V = (2 pi / 3) Int_{theta_j}^{theta_j+1} (r_out(theta)^3 - r_in(theta)^3) sin(theta) d theta
(Gauss-Legendre in theta, exact for spherical sectors).  Each step:

1. Lagrangian phase (predictor-corrector, like the 1D VNR scheme):
   * radial node force  F_r = sum_z (P + q_r)_z dV_z/dr_p  (+ centrifugal term),
     nodal masses consistent with dV/dr, so a spherically symmetric state
     evolves exactly as in the 1D code on every spoke, the axis included;
   * angular force from the pressure difference across each spoke face,
     integrated as angular momentum  d(M r u_theta)/dt = r F_theta;
   * zone volume change = radial part (from the new radii) + angular part
     (spoke faces swept by u_theta); compatible P dV work split between
     electrons, ions and radiation; viscous heating (radial and angular edge
     viscosities, only in compression) goes to the ions.
2. Remap phase: the spokes return to their fixed angles, i.e. mass, internal
   energies, radiation energy and fuel are advected across the spoke faces with
   the swept volumes (second-order, van Leer limited), nodal momentum likewise;
   the kinetic energy lost in the momentum remap is returned to the ions so that
   total energy is conserved.
Because zones never leave their ring of radial index, materials (defined per
ring) never mix.  Optionally the innermost `core` rings (tiny cones around the
origin, whose angular width r dtheta would set the time step at stagnation) are
treated as a quasi-1D core: no angular motion, ring-averaged thermodynamics.
A fourth-difference anti-hourglass stiffness acts on the angular zig-zag mode of
the node radii (a zero-energy mode of the linear-in-theta zone boundaries).
"""
import math

import numpy as np
from numba import njit

from .eos import eos_update

TWO_PI = 2.0 * math.pi
NG = 4


def gauss_theta(theta, ng=NG):
    """Gauss points in each angular zone: s (Nj, ng) in [0, 1] and weights
    W (Nj, ng) = 2 pi * w * dtheta * sin(theta_g)."""
    x, w = np.polynomial.legendre.leggauss(ng)
    Nj = len(theta) - 1
    s = np.empty((Nj, ng)); W = np.empty((Nj, ng))
    for j in range(Nj):
        a, b = theta[j], theta[j + 1]
        th = a + (b - a) * 0.5 * (x + 1.0)
        s[j] = 0.5 * (x + 1.0)
        W[j] = TWO_PI * 0.5 * (b - a) * w * np.sin(th)
    return s, W


@njit(cache=True)
def sph_volumes(r, s, W, V):
    Ni, Nj = V.shape
    ng = s.shape[1]
    for i in range(Ni):
        for j in range(Nj):
            v = 0.0
            for g in range(ng):
                ro = r[i + 1, j] * (1.0 - s[j, g]) + r[i + 1, j + 1] * s[j, g]
                ri = r[i, j] * (1.0 - s[j, g]) + r[i, j + 1] * s[j, g]
                v += W[j, g] * (ro ** 3 - ri ** 3)
            V[i, j] = v / 3.0


@njit(cache=True)
def _dVdr(r, s, W, i, j):
    """dV_z/dr for the 4 nodes of zone (i, j): (inner j, inner j+1, outer j, outer j+1)."""
    ng = s.shape[1]
    a0 = 0.0; a1 = 0.0; b0 = 0.0; b1 = 0.0
    for g in range(ng):
        sg = s[j, g]
        ro = r[i + 1, j] * (1.0 - sg) + r[i + 1, j + 1] * sg
        ri = r[i, j] * (1.0 - sg) + r[i, j + 1] * sg
        b0 += W[j, g] * ro * ro * (1.0 - sg)
        b1 += W[j, g] * ro * ro * sg
        a0 -= W[j, g] * ri * ri * (1.0 - sg)
        a1 -= W[j, g] * ri * ri * sg
    return a0, a1, b0, b1


@njit(cache=True)
def spoke_areas(r, theta, S):
    """Area of the spoke faces: S[i, j] between node rows i, i+1 on spoke j."""
    Ni1, Nj1 = r.shape
    for i in range(Ni1 - 1):
        for j in range(Nj1):
            S[i, j] = math.pi * math.sin(theta[j]) * (r[i + 1, j] ** 2 - r[i, j] ** 2)


@njit(cache=True)
def nodal_masses_sph(r, s, W, m, M):
    """Each zone gives half its mass to its inner and half to its outer node pair,
    shared in proportion to dV/dr (consistent with the radial forces)."""
    Ni, Nj = m.shape
    M[:, :] = 0.0
    for i in range(Ni):
        for j in range(Nj):
            a0, a1, b0, b1 = _dVdr(r, s, W, i, j)
            if i > 0:
                M[i, j] += 0.5 * m[i, j] * a0 / (a0 + a1)
                M[i, j + 1] += 0.5 * m[i, j] * a1 / (a0 + a1)
            else:
                M[0, j] += 0.25 * m[i, j]       # origin (fixed), as in the 1D scheme
                M[0, j + 1] += 0.25 * m[i, j]
            M[i + 1, j] += 0.5 * m[i, j] * b0 / (b0 + b1)
            M[i + 1, j + 1] += 0.5 * m[i, j] * b1 / (b0 + b1)


@njit(cache=True)
def sph_geometry(r, theta, s, W, V, Arad, drad, Aang, dang, rc):
    """Zone volumes and centre radii, and the face areas / centre distances used by
    the diffusion solver: radial faces Arad[i, j] (node row i, i = 0..Ni) and angular
    (spoke) faces Aang[i, j] (spoke j = 0..Nj; the axis / equator faces are closed)."""
    Ni, Nj = V.shape
    ng = s.shape[1]
    sph_volumes(r, s, W, V)
    for i in range(Ni):
        for j in range(Nj):
            rc[i, j] = 0.25 * (r[i, j] + r[i, j + 1] + r[i + 1, j] + r[i + 1, j + 1])
    for i in range(Ni + 1):
        for j in range(Nj):
            a = 0.0
            for g in range(ng):
                rg = r[i, j] * (1.0 - s[j, g]) + r[i, j + 1] * s[j, g]
                a += W[j, g] * rg * rg
            Arad[i, j] = a
            if i == 0:
                drad[i, j] = 1.0
            elif i == Ni:
                drad[i, j] = max(0.5 * (r[i, j] + r[i, j + 1]) - rc[i - 1, j], 1e-300)
            else:
                drad[i, j] = max(rc[i, j] - rc[i - 1, j], 1e-300)
    for i in range(Ni):
        for j in range(Nj + 1):
            if j == 0 or j == Nj:
                Aang[i, j] = 0.0
                dang[i, j] = 1.0
            else:
                Aang[i, j] = math.pi * math.sin(theta[j]) * (r[i + 1, j] ** 2 - r[i, j] ** 2)
                dc = 0.5 * (theta[j + 1] - theta[j - 1])
                dang[i, j] = max(0.5 * (rc[i, j - 1] + rc[i, j]) * dc, 1e-300)


@njit(cache=True)
def _zone_q(r, ur, ut, rho, cs2, C1, C2, qr, qe, qt):
    """Edge viscosities (compression only).  Radial: one per spoke edge of the zone,
    qe[i, j, 0/1] for spokes j / j+1, so that a compression confined to one spoke is
    still captured (a zone-average q lets a single spoke node overrun its
    neighbour).  qr = max of the two (for the time step).  Angular: zone average."""
    Ni, Nj = rho.shape
    for i in range(Ni):
        for j in range(Nj):
            cs = math.sqrt(cs2[i, j])
            qm = 0.0
            for e in range(2):
                du = ur[i + 1, j + e] - ur[i, j + e]
                if du < 0.0:
                    q = rho[i, j] * (C2 * du * du - C1 * cs * du)
                else:
                    q = 0.0
                qe[i, j, e] = q
                if q > qm:
                    qm = q
            qr[i, j] = qm
            dut = 0.5 * (ut[i, j + 1] + ut[i + 1, j + 1]) - 0.5 * (ut[i, j] + ut[i + 1, j])
            if dut < 0.0:
                qt[i, j] = rho[i, j] * (C2 * dut * dut - C1 * cs * dut)
            else:
                qt[i, j] = 0.0


@njit(cache=True)
def _mir(j, Nj):
    """Spoke index mirrored at the axis (j < 0) and at the equator / far axis (j > Nj)."""
    if j < 0:
        return -j
    if j > Nj:
        return 2 * Nj - j
    return j


@njit(cache=True)
def _forces_sph(r, theta, s, W, P, qe, qt, rho, cs2, hg, Fr, Ft, S, core, PfL, PfR, hgo):
    """Radial nodal forces, angular nodal forces and spoke-face areas."""
    Ni, Nj = P.shape
    Fr[:, :] = 0.0
    Ft[:, :] = 0.0
    for i in range(Ni):
        for j in range(Nj):
            a0, a1, b0, b1 = _dVdr(r, s, W, i, j)
            P0 = P[i, j] + qe[i, j, 0]
            P1 = P[i, j] + qe[i, j, 1]
            Fr[i, j] += P0 * a0
            Fr[i, j + 1] += P1 * a1
            Fr[i + 1, j] += P0 * b0
            Fr[i + 1, j + 1] += P1 * b1
    spoke_areas(r, theta, S)
    # zone pressures at their spoke faces.  The zones of one ring sit at different
    # radii once the ring is tilted, so the raw difference P[i, j-1] - P[i, j] contains
    # the background radial gradient times the tilt, a spurious tangential force of the
    # same size as the Rayleigh-Taylor drive (it stabilises unstable and destabilises
    # stable stratification).  The pressure is therefore carried to the face mid-radius
    # with its limited radial gradient along the zone's own column.
    for i in range(Ni):
        for j in range(Nj):
            rc = 0.25 * (r[i, j] + r[i, j + 1] + r[i + 1, j] + r[i + 1, j + 1])
            g = 0.0
            if 0 < i < Ni - 1:
                rm = 0.25 * (r[i - 1, j] + r[i - 1, j + 1] + r[i, j] + r[i, j + 1])
                rp = 0.25 * (r[i + 1, j] + r[i + 1, j + 1] + r[i + 2, j] + r[i + 2, j + 1])
                gm = (P[i, j] - P[i - 1, j]) / max(rc - rm, 1e-300)
                gp = (P[i + 1, j] - P[i, j]) / max(rp - rc, 1e-300)
                if gm * gp > 0.0:
                    g = gm if abs(gm) < abs(gp) else gp
            PfL[i, j] = P[i, j] + g * (0.5 * (r[i, j] + r[i + 1, j]) - rc)
            PfR[i, j] = P[i, j] + g * (0.5 * (r[i, j + 1] + r[i + 1, j + 1]) - rc)
    for i in range(core, Ni):
        for j in range(1, Nj):
            # spoke face (i, j) between zones (i, j-1) and (i, j); half to each end node
            dP = (PfR[i, j - 1] + qt[i, j - 1]) - (PfL[i, j] + qt[i, j])
            f = 0.5 * S[i, j] * dP
            Ft[i, j] += f
            Ft[i + 1, j] += f
    # anti-hourglass: the angular zig-zag of node radii does not change zone volumes
    # (zero-energy mode of the linear-in-theta zone boundaries).  Resist the part of
    # r_j that a 4th-order interpolation from its neighbours does not explain, with
    # a stiffness comparable to the radial one (rho cs^2 A / dr); smooth modes l
    # are affected only at order (l dtheta)^4.
    if hg > 0.0:
        Ni1 = Ni + 1
        for i in range(1, Ni1):
            iz = i - 1 if i == Ni1 - 1 else i
            izm = i - 1
            for j in range(0, Nj + 1):
                # neighbours, mirrored at the axis (j = 0) and at the equator / far axis (j = Nj)
                if hgo == 6:
                    # 6th difference, scaled so that the sawtooth sees the same force as
                    # with the 4th difference; smooth modes feel (k dtheta)^2 / 4 of it
                    d6 = (r[i, _mir(j - 3, Nj)] - 6.0 * r[i, _mir(j - 2, Nj)] + 15.0 * r[i, _mir(j - 1, Nj)]
                          - 20.0 * r[i, j] + 15.0 * r[i, _mir(j + 1, Nj)] - 6.0 * r[i, _mir(j + 2, Nj)]
                          + r[i, _mir(j + 3, Nj)])
                    h = -d6 / 24.0
                else:
                    interp = (-r[i, _mir(j - 2, Nj)] + 4.0 * r[i, _mir(j - 1, Nj)] + 4.0 * r[i, _mir(j + 1, Nj)]
                              - r[i, _mir(j + 2, Nj)]) / 6.0
                    h = r[i, j] - interp
                rr = r[i, j]
                ja = j - 1 if j >= 1 else 0
                jb = j if j < Nj else Nj - 1
                th0 = theta[ja]; th1 = theta[jb + 1]
                kz = 0.5 * (rho[iz, ja] * cs2[iz, ja] + rho[iz, jb] * cs2[iz, jb])
                drl = 0.5 * (r[iz + 1, j] - r[iz, j]) + 0.5 * (r[izm + 1, j] - r[izm, j])
                # area of the spoke's dual cell (half cells at the axis / equator)
                if j == 0 or j == Nj:
                    area = TWO_PI * rr * rr * abs(math.cos(theta[j]) - math.cos(0.5 * (th0 + th1)))
                else:
                    area = TWO_PI * rr * rr * abs(math.cos(0.5 * (theta[j - 1] + theta[j])) -
                                                  math.cos(0.5 * (theta[j] + theta[j + 1])))
                Fr[i, j] -= hg * kz * area * h / max(abs(drl), 1e-300)


@njit(cache=True)
def hydro_sph_step(dt, r, theta, s, W, ur, ut, m, M, rho, ee, ei, er,
                   Zb, Ab_mu, rho0, Pd0, B0, solid, deg,
                   Te, Ti, Pe, Pi, Pc, cs2, Cve, Cvi, qr, qt, C1, C2, hg, half, fix_outer, dVt, core, tie=True, grav=0.0, hgt=0.5, hgo=4, hgv=0.0):
    """Lagrangian phase.  Updates r, ur, ut and the energies in place, fills the
    angular swept volumes dVt[i, j] (face j of ring i) for the remap and returns the
    viscous heating [erg].  rho on exit is the Lagrangian density.  Rings i < core
    form a quasi-1D core with no angular motion (the driver ring-averages them),
    which removes the angular Courant limit of the tiny central cones."""
    Ni, Nj = rho.shape
    n = Ni * Nj
    V0 = np.empty((Ni, Nj))
    sph_volumes(r, s, W, V0)
    Pe2 = Pe.reshape((Ni, Nj)); Pi2 = Pi.reshape((Ni, Nj))
    # ---- predictor (half step, radial + angular volume change with u^n)
    rh = r + 0.5 * dt * ur
    for j in range(Nj + 1):
        rh[0, j] = 0.0
    Vh = np.empty((Ni, Nj))
    sph_volumes(rh, s, W, Vh)
    S = np.empty((Ni, Nj + 1))
    spoke_areas(rh, theta, S)
    for i in range(Ni):
        for j in range(Nj):
            Vh[i, j] += 0.5 * dt * (S[i, j + 1] * 0.5 * (ut[i, j + 1] + ut[i + 1, j + 1]) -
                                    S[i, j] * 0.5 * (ut[i, j] + ut[i + 1, j]))
    rhoh = np.empty((Ni, Nj)); eeh = np.empty((Ni, Nj)); eih = np.empty((Ni, Nj)); erh = np.empty((Ni, Nj))
    for i in range(Ni):
        for j in range(Nj):
            rhoh[i, j] = m[i, j] / Vh[i, j]
            dV = Vh[i, j] - V0[i, j]
            eeh[i, j] = ee[i, j] - Pe2[i, j] * dV / m[i, j]
            eih[i, j] = ei[i, j] - Pi2[i, j] * dV / m[i, j]
            erh[i, j] = max(er[i, j] - er[i, j] * rho[i, j] / 3.0 * dV / m[i, j], 0.0)
    Teh = np.empty(n); Tih = np.empty(n); Peh = np.empty(n); Pih = np.empty(n)
    Pch = np.empty(n); cs2h = np.empty(n); Cveh = np.empty(n); Cvih = np.empty(n)
    eos_update(rhoh.reshape(n), eeh.reshape(n), eih.reshape(n), Zb, Ab_mu, rho0, Pd0, B0, solid, deg,
               Teh, Tih, Peh, Pih, Pch, cs2h, Cveh, Cvih)
    Ph = np.empty((Ni, Nj))
    for i in range(Ni):
        for j in range(Nj):
            kk = i * Nj + j
            Ph[i, j] = Peh[kk] + Pih[kk] + Pch[kk] + erh[i, j] * rhoh[i, j] / 3.0
    cs2h2 = cs2h.reshape((Ni, Nj))
    qe = np.empty((Ni, Nj, 2))
    _zone_q(r, ur, ut, rhoh, cs2h2, C1, C2, qr, qe, qt)
    Fr = np.zeros(r.shape); Ft = np.zeros(r.shape)
    PfL = np.empty((Ni, Nj)); PfR = np.empty((Ni, Nj))
    _forces_sph(rh, theta, s, W, Ph, qe, qt, rhoh, cs2h2, hg, Fr, Ft, S, core, PfL, PfR, hgo)
    if grav != 0.0:       # uniform inward radial body force (test problems only)
        for i in range(1, Ni + 1):
            for j in range(Nj + 1):
                Fr[i, j] -= M[i, j] * grav
    # tangential-velocity hourglass: a radial zig-zag of u_theta along a spoke does
    # not change any zone volume (the spoke faces see the average of their two end
    # nodes), so it is damped explicitly (viscous Flanagan-Belytschko-type force on
    # the part of u_theta not explained by a 4th-order fit of its radial neighbours).
    Fd = np.zeros(r.shape)
    if hgt > 0.0:
        for i in range(max(core, 1) + 1, Ni):
            for j in range(1, Nj):
                im2 = i - 2 if i - 2 >= 1 else 1
                ip2 = i + 2 if i + 2 <= Ni else Ni
                h = ut[i, j] - (-ut[im2, j] + 4.0 * ut[i - 1, j] + 4.0 * ut[i + 1, j] - ut[ip2, j]) / 6.0
                rz = 0.25 * (rhoh[i - 1, j - 1] + rhoh[i - 1, j] + rhoh[i, j - 1] + rhoh[i, j])
                cz = math.sqrt(0.25 * (cs2h2[i - 1, j - 1] + cs2h2[i - 1, j] + cs2h2[i, j - 1] + cs2h2[i, j]))
                Sf = 0.5 * (S[i - 1, j] + S[i, j])
                Fd[i, j] = -hgt * rz * cz * Sf * h
                Ft[i, j] += Fd[i, j]
    # viscous anti-hourglass on the angular sawtooth of u_r: damps the grid-scale
    # mode without the elastic restoring force of the stiffness version (an elastic
    # force stabilises low-l Rayleigh-Taylor modes; a viscous one only slows them).
    Fv = np.zeros(r.shape)
    if hgv > 0.0:
        for i in range(1, Ni + 1):
            iz = i - 1 if i == Ni else i
            for j in range(0, Nj + 1):
                hu = (ur[i, _mir(j - 2, Nj)] - 4.0 * ur[i, _mir(j - 1, Nj)] + 6.0 * ur[i, j]
                      - 4.0 * ur[i, _mir(j + 1, Nj)] + ur[i, _mir(j + 2, Nj)]) / 6.0
                ja = j - 1 if j >= 1 else 0
                jb = j if j < Nj else Nj - 1
                rz = 0.5 * (rhoh[iz, ja] + rhoh[iz, jb])
                cz = math.sqrt(0.5 * (cs2h2[iz, ja] + cs2h2[iz, jb]))
                # (the node mass over its radial extent) * c, i.e. rho c A
                drn = 0.5 * (r[min(i + 1, Ni), j] - r[i - 1, j])
                Fv[i, j] = -hgv * M[i, j] * cz / max(drn, 1e-300) * hu
                Fr[i, j] += Fv[i, j]
    # ---- corrector: velocities
    ur0 = ur.copy(); ut0 = ut.copy()
    for i in range(1, Ni + 1):
        for j in range(Nj + 1):
            if M[i, j] <= 0.0:
                continue
            # centrifugal term
            Fc = M[i, j] * ut0[i, j] ** 2 / max(rh[i, j], 1e-300)
            ur[i, j] = ur0[i, j] + dt * (Fr[i, j] + Fc) / M[i, j]
    # axis nodes: tied rigidly to their neighbour spoke (sin(theta) -> 0 gives them
    # almost no pressure area, so as free nodes they could overrun their radial
    # neighbours and seed axis jets).  Forces and masses are combined.
    for i in range(1, Ni + 1 if tie else 1):
        for (ja, jb) in ((0, 1), (Nj, Nj - 1)):
            if ja == Nj and half:
                continue
            Mt = M[i, ja] + M[i, jb]
            if Mt <= 0.0:
                continue
            u = (M[i, ja] * ur0[i, ja] + M[i, jb] * ur0[i, jb] + dt * (Fr[i, ja] + Fr[i, jb] +
                 M[i, jb] * ut0[i, jb] ** 2 / max(rh[i, jb], 1e-300))) / Mt
            ur[i, ja] = u
            ur[i, jb] = u
    if fix_outer:
        for j in range(Nj + 1):
            ur[Ni, j] = 0.0
    for j in range(Nj + 1):
        ur[0, j] = 0.0
    r_old = r.copy()
    for i in range(Ni + 1):
        for j in range(Nj + 1):
            r[i, j] = r_old[i, j] + dt * 0.5 * (ur0[i, j] + ur[i, j])
            if i == 0:
                r[i, j] = 0.0
    for i in range(1, Ni + 1):
        for j in range(Nj + 1):
            if M[i, j] <= 0.0 or j == 0 or j == Nj or i <= core:
                ut[i, j] = 0.0
                continue
            Lm = r_old[i, j] * ut0[i, j] + dt * rh[i, j] * Ft[i, j] / M[i, j]
            ut[i, j] = Lm / max(r[i, j], 1e-300)
    for j in range(Nj + 1):
        ut[0, j] = 0.0
    # ---- energies with the time-centred velocities (compatible work)
    heat = 0.0
    if hgv > 0.0:
        for i in range(1, Ni + 1):
            for j in range(Nj + 1):
                if Fv[i, j] != 0.0:
                    wq = -Fv[i, j] * 0.5 * (ur0[i, j] + ur[i, j]) * dt
                    ja = j - 1 if j >= 1 else 0
                    jb = j if j < Nj else Nj - 1
                    na = 2 if i < Ni else 1
                    nb = 2 if ja != jb else 1
                    for a in range(i - 1, i - 1 + na):
                        for b in range(ja, ja + nb):
                            ei[a, b] += wq / (na * nb) / m[a, b]
                    heat += wq
    if hgt > 0.0:
        for i in range(1, Ni):
            for j in range(1, Nj):
                if Fd[i, j] != 0.0:
                    wq = -Fd[i, j] * 0.5 * (ut0[i, j] + ut[i, j]) * dt
                    for a in (i - 1, i):
                        for b in (j - 1, j):
                            ei[a, b] += 0.25 * wq / m[a, b]
                    heat += wq
    for i in range(Ni):
        for j in range(Nj):
            a0, a1, b0, b1 = _dVdr(rh, s, W, i, j)
            dV0 = dt * (a0 * 0.5 * (ur0[i, j] + ur[i, j]) + b0 * 0.5 * (ur0[i + 1, j] + ur[i + 1, j]))
            dV1 = dt * (a1 * 0.5 * (ur0[i, j + 1] + ur[i, j + 1]) + b1 * 0.5 * (ur0[i + 1, j + 1] + ur[i + 1, j + 1]))
            dVr = dV0 + dV1
            uL = 0.25 * (ut0[i, j] + ut[i, j] + ut0[i + 1, j] + ut[i + 1, j])
            uR = 0.25 * (ut0[i, j + 1] + ut[i, j + 1] + ut0[i + 1, j + 1] + ut[i + 1, j + 1])
            dVtL = dt * S[i, j] * uL
            dVtR = dt * S[i, j + 1] * uR
            dVth = dVtR - dVtL
            dVt[i, j] = dVtL
            if j == Nj - 1:
                dVt[i, Nj] = dVtR
            kk = i * Nj + j
            dV = dVr + dVth
            ee[i, j] -= Peh[kk] * dV / m[i, j]
            hq = -(qe[i, j, 0] * dV0 + qe[i, j, 1] * dV1 + qt[i, j] * dVth)
            ei[i, j] += (-Pih[kk] * dV + hq) / m[i, j]
            if i >= core:
                # angular work done with the face pressures (compatible with the forces)
                wc = -((PfR[i, j] - Ph[i, j]) * dVtR - (PfL[i, j] - Ph[i, j]) * dVtL)
                pe = max(Peh[kk], 0.0); pt = pe + max(Pih[kk], 0.0)
                fe = pe / pt if pt > 0.0 else 0.5
                ee[i, j] += fe * wc / m[i, j]
                ei[i, j] += (1.0 - fe) * wc / m[i, j]
            er[i, j] = max(er[i, j] - erh[i, j] * rhoh[i, j] / 3.0 * dV / m[i, j], 0.0)
            V0[i, j] += dV
            rho[i, j] = m[i, j] / V0[i, j]
            heat += hq
    eos_update(rho.reshape(n), ee.reshape(n), ei.reshape(n), Zb, Ab_mu, rho0, Pd0, B0, solid, deg,
               Te, Ti, Pe, Pi, Pc, cs2, Cve, Cvi)
    return heat


@njit(cache=True)
def _vl_slope(a, b, c):
    """van Leer limited slope from three successive values."""
    d1 = b - a
    d2 = c - b
    if d1 * d2 <= 0.0:
        return 0.0
    return 2.0 * d1 * d2 / (d1 + d2)


@njit(cache=True)
def remap_theta(r, theta, s, W, m, M, rho, ee, ei, er, yDT, ur, ut, dVt, second_order, half):
    """Return the spokes to their fixed angles: advect zone and node quantities
    across the spoke faces with the swept volumes dVt[i, j] (face j of ring i,
    positive = toward larger theta).  Returns the kinetic energy given to the ions."""
    Ni, Nj = m.shape
    if Nj < 2:
        return 0.0
    Fm = np.zeros((Ni, Nj + 1))
    Fe = np.zeros((Ni, Nj + 1)); Fi = np.zeros((Ni, Nj + 1)); Fr_ = np.zeros((Ni, Nj + 1)); Fy = np.zeros((Ni, Nj + 1))
    for i in range(Ni):
        for j in range(1, Nj):
            dv = dVt[i, j]
            if dv == 0.0:
                continue
            k = j - 1 if dv > 0.0 else j          # donor zone
            # reconstructed face values (density and specific quantities)
            rf = rho[i, k]; ef = ee[i, k]; xf = ei[i, k]; gf = er[i, k]; yf = yDT[i, k]
            if second_order and 0 < k < Nj - 1:
                sgn = 0.5 if dv > 0.0 else -0.5
                rf += sgn * _vl_slope(rho[i, k - 1], rho[i, k], rho[i, k + 1])
                ef += sgn * _vl_slope(ee[i, k - 1], ee[i, k], ee[i, k + 1])
                xf += sgn * _vl_slope(ei[i, k - 1], ei[i, k], ei[i, k + 1])
                gf += sgn * _vl_slope(er[i, k - 1], er[i, k], er[i, k + 1])
                yf += sgn * _vl_slope(yDT[i, k - 1], yDT[i, k], yDT[i, k + 1])
            fm = rf * dv
            # never take more than half of the donor's mass
            if abs(fm) > 0.5 * m[i, k]:
                fm = math.copysign(0.5 * m[i, k], fm)
            Fm[i, j] = fm
            Fe[i, j] = fm * ef; Fi[i, j] = fm * xf; Fr_[i, j] = fm * gf; Fy[i, j] = fm * yf
    # ---- node momentum (before zone masses change)
    KE0 = 0.0
    for i in range(1, Ni + 1):
        for j in range(Nj + 1):
            KE0 += 0.5 * M[i, j] * (ur[i, j] ** 2 + ut[i, j] ** 2)
    Pr = np.zeros(M.shape); Pt = np.zeros(M.shape); Mn = M.copy()
    for i in range(1, Ni + 1):
        for j in range(Nj + 1):
            Pr[i, j] = M[i, j] * ur[i, j]
            Pt[i, j] = M[i, j] * ut[i, j]
    for i in range(1, Ni + 1):
        for jd in range(Nj):     # dual face between node j=jd and jd+1
            f = 0.0
            nrow = 0
            for iz in (i - 1, i):
                if iz < 0 or iz >= Ni:
                    continue
                f += 0.5 * (Fm[iz, jd] + Fm[iz, jd + 1])
                nrow += 1
            f *= 0.5 if nrow == 2 else 1.0
            if f == 0.0:
                continue
            d = jd if f > 0.0 else jd + 1
            if abs(f) > 0.5 * Mn[i, d]:
                f = math.copysign(0.5 * Mn[i, d], f)
            M[i, jd] -= f; M[i, jd + 1] += f
            Pr[i, jd] -= f * ur[i, d]; Pr[i, jd + 1] += f * ur[i, d]
            Pt[i, jd] -= f * ut[i, d]; Pt[i, jd + 1] += f * ut[i, d]
    for i in range(1, Ni + 1):
        for j in range(Nj + 1):
            if M[i, j] > 0.0:
                ur[i, j] = Pr[i, j] / M[i, j]
                ut[i, j] = Pt[i, j] / M[i, j] if (j > 0 and j < Nj) else 0.0
        # keep the axis node tied to its neighbour (momentum-conserving average)
        for (ja, jb) in ((0, 1), (Nj, Nj - 1)):
            if ja == Nj and half:
                continue
            Mt = M[i, ja] + M[i, jb]
            if Mt > 0.0:
                u = (M[i, ja] * ur[i, ja] + M[i, jb] * ur[i, jb]) / Mt
                ur[i, ja] = u
                ur[i, jb] = u
    KE1 = 0.0
    for i in range(1, Ni + 1):
        for j in range(Nj + 1):
            KE1 += 0.5 * M[i, j] * (ur[i, j] ** 2 + ut[i, j] ** 2)
    # ---- zone quantities
    V = np.empty((Ni, Nj))
    sph_volumes(r, s, W, V)
    for i in range(Ni):
        for j in range(Nj):
            dm = Fm[i, j] - Fm[i, j + 1]
            mn = m[i, j] + dm
            ee[i, j] = (m[i, j] * ee[i, j] + Fe[i, j] - Fe[i, j + 1]) / mn
            ei[i, j] = (m[i, j] * ei[i, j] + Fi[i, j] - Fi[i, j + 1]) / mn
            er[i, j] = max((m[i, j] * er[i, j] + Fr_[i, j] - Fr_[i, j + 1]) / mn, 0.0)
            yDT[i, j] = max((m[i, j] * yDT[i, j] + Fy[i, j] - Fy[i, j + 1]) / mn, 0.0)
            m[i, j] = mn
            rho[i, j] = mn / V[i, j]
    # kinetic energy lost by the momentum remap -> ions (mass-weighted)
    dKE = KE0 - KE1
    if dKE != 0.0:
        mt = 0.0
        for i in range(Ni):
            for j in range(Nj):
                mt += m[i, j]
        # distribute where the remap acted (zones with flux), else uniformly
        wsum = 0.0
        for i in range(Ni):
            for j in range(Nj):
                wsum += abs(Fm[i, j]) + abs(Fm[i, j + 1])
        for i in range(Ni):
            for j in range(Nj):
                if wsum > 0.0:
                    w = (abs(Fm[i, j]) + abs(Fm[i, j + 1])) / wsum
                else:
                    w = m[i, j] / mt
                ei[i, j] += dKE * w / m[i, j]
    return dKE


@njit(cache=True)
def cfl_sph(r, theta, ur, ut, cs2, er, qr, qt, C2, cfl, core):
    Ni = cs2.shape[0]
    Nj = cs2.shape[1]
    dtmin = 1e30
    for i in range(Ni):
        for j in range(Nj):
            dr = 0.5 * (r[i + 1, j] + r[i + 1, j + 1] - r[i, j] - r[i, j + 1])
            rm = 0.25 * (r[i + 1, j] + r[i + 1, j + 1] + r[i, j] + r[i, j + 1])
            dl = rm * (theta[j + 1] - theta[j])
            c = math.sqrt(cs2[i, j] + 4.0 / 9.0 * er[i, j])
            dur = abs(0.5 * (ur[i + 1, j] + ur[i + 1, j + 1] - ur[i, j] - ur[i, j + 1]))
            dut = abs(0.5 * (ut[i, j + 1] + ut[i + 1, j + 1] - ut[i, j] - ut[i + 1, j]))
            utm = 0.25 * (abs(ut[i, j]) + abs(ut[i, j + 1]) + abs(ut[i + 1, j]) + abs(ut[i + 1, j + 1]))
            dt1 = cfl * dr / (c + (2.0 * C2 * dur if qr[i, j] > 0 else 0.0))
            dt2 = cfl * dl / (c + utm + (2.0 * C2 * dut if qt[i, j] > 0 else 0.0))
            d = min(dt1, dt2) if i >= core else dt1
            if d < dtmin:
                dtmin = d
    return dtmin
