"""Laser energy deposition by 1D spherical ray tracing (inverse bremsstrahlung).

In a spherically symmetric plasma each ray conserves  n(r) r sin(psi) = b
(Bouguer's law), b being its impact parameter in vacuum, n^2 = 1 - ne/nc.
The IB optical depth along the ray is
    tau = Int kappa_ib ds = Int nu_ei(ne) (ne/nc) r dr / (c sqrt(r^2 n^2 - b^2)),
(kappa_ib = nu_ei X / (c n), ds = n r dr / sqrt(n^2 r^2 - b^2)).  The ray turns
where r n(r) = b.  ne is interpolated linearly between zone centres and the
integral is done analytically per zone, which removes the spurious long
grazing chords of a piecewise-constant-index tracer.  The return path is the
mirror image of the inward one.

Direct-drive illumination by many overlapping beams is represented, as in
1D codes such as LILAC/MULTI, by a set of rays whose impact parameters
follow the focal-spot intensity profile I(b) (super-Gaussian of radius R_b).
KrF lasers can "zoom" (shrink R_b as the target implodes), which is supported
through a user-supplied R_b(t).
"""
import math

import numpy as np
from numba import njit

from . import constants as C
from .transport import coulomb_log, nu_ei
from .constants import c


def critical_density(lambda_um):
    """n_c [cm^-3] for vacuum wavelength in microns."""
    return 1.1148e21 / lambda_um ** 2


@njit(cache=True)
def _tau_segment(ra, rb, ga, gb, ha, hb):
    """Integral of h(r)/sqrt(g(r)) dr over [ra, rb] with g, h linear in r (g >= 0)."""
    dr = rb - ra
    if dr <= 0.0:
        return 0.0
    G1 = (gb - ga) / dr
    gm = max(ga, gb)
    if abs(G1) * dr < 1e-6 * gm:
        return 0.5 * (ha + hb) * dr / math.sqrt(0.5 * (ga + gb))
    sa = math.sqrt(max(ga, 0.0))
    sb = math.sqrt(max(gb, 0.0))
    I0 = 2.0 * (sb - sa) / G1
    I1 = ((2.0 / 3.0) * (sb ** 3 - sa ** 3) - 2.0 * ga * (sb - sa)) / (G1 * G1)
    return ha * I0 + (hb - ha) / dr * I1


@njit(cache=True)
def trace_rays(r, ne, Te, Zeff, nc, b, w, power, dump_frac, dep):
    """Deposit `power` (erg/s) carried by rays (impact parameter b, weight w).

    The electron density is interpolated linearly between zone centres
    (node values), so that  g(r) = r^2 (1 - ne/nc) - b^2  is piecewise linear and
    the optical depth  tau = Int nu_ib(ne) (ne/nc) r dr / (c sqrt(g))  is
    integrated analytically through each zone, including the integrable
    square-root singularity at the turning point.
    dep[N] (erg/s per zone) is filled.  Returns (absorbed, missed, escaped).
    """
    N = ne.shape[0]
    for k in range(N):
        dep[k] = 0.0
    X = np.empty(N + 1)     # ne/nc at nodes
    X[0] = ne[0] / nc
    X[N] = ne[N - 1] / nc
    for j in range(1, N):
        X[j] = 0.5 * (ne[j - 1] + ne[j]) / nc
    nu_over_ne = np.empty(N)
    for k in range(N):
        nu_over_ne[k] = nu_ei(ne[k], Te[k], Zeff[k], coulomb_log(ne[k], Te[k], Zeff[k])) / ne[k]
    tau = np.zeros(N)
    absorbed = 0.0
    missed = 0.0
    escaped = 0.0
    R = r[N]
    for j in range(b.shape[0]):
        P = power * w[j]
        if P <= 0.0:
            continue
        bb = b[j] * b[j]
        if b[j] >= R:
            missed += P
            continue
        if X[N] >= 1.0:
            # no underdense plasma yet (cold solid surface): the light is absorbed
            # in the skin layer of the outermost zone (start-up / breakdown model)
            dep[N - 1] += P
            continue
        if R * R * (1.0 - X[N]) - bb <= 0.0:
            escaped += P
            continue
        last = N
        for k in range(N - 1, -1, -1):
            ro = r[k + 1]
            ri = r[k]
            go = ro * ro * (1.0 - X[k + 1]) - bb
            gi = ri * ri * (1.0 - X[k]) - bb
            c0 = nu_over_ne[k] * nc / c
            ho = c0 * X[k + 1] * X[k + 1] * ro
            last = k
            if gi > 0.0 and k > 0:
                tau[k] = _tau_segment(ri, ro, gi, go, c0 * X[k] * X[k] * ri, ho)
                dP = P * (1.0 - math.exp(-tau[k]))
                dep[k] += dP
                P -= dP
            else:
                # turning point inside this zone (or ray through the centre)
                if gi >= go:
                    rt = ri
                else:
                    rt = ro - go * (ro - ri) / (go - gi)
                    if rt < ri:
                        rt = ri
                Xt = X[k] + (X[k + 1] - X[k]) * (rt - ri) / (ro - ri)
                tau[k] = _tau_segment(rt, ro, 0.0 if gi <= 0.0 else gi, go, c0 * Xt * Xt * rt, ho)
                dP = P * (1.0 - math.exp(-2.0 * tau[k]))
                dep[k] += dP
                P -= dP
                break
        if dump_frac > 0.0:
            dP = P * dump_frac
            dep[last] += dP
            P -= dP
        for k in range(last + 1, N):
            dP = P * (1.0 - math.exp(-tau[k]))
            dep[k] += dP
            P -= dP
        escaped += P
    for k in range(N):
        absorbed += dep[k]
    return absorbed, missed, escaped


def make_rays(n_rays, R_beam, profile_order=4.0, b_max_factor=2.0):
    """Impact parameters and power weights for a super-Gaussian focal spot
    I(b) ~ exp(-(b/R_beam)^m); weights include the 2*pi*b annulus area and sum to 1."""
    bmax = b_max_factor * R_beam
    edges = np.linspace(0.0, bmax, n_rays + 1)
    b = 0.5 * (edges[1:] + edges[:-1])
    # integrate exactly-ish on a fine grid per bin
    w = np.empty(n_rays)
    for j in range(n_rays):
        bb = np.linspace(edges[j], edges[j + 1], 21)
        f = np.exp(-(bb / R_beam) ** profile_order) * bb
        w[j] = np.trapezoid(f, bb)
    w /= w.sum()
    return b, w
