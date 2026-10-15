"""SIFE-2D driver: axisymmetric (z, R) Lagrangian three-temperature radiation
hydrodynamics for laser direct-drive capsules.

Physics per step (operator split, same order as the 1D code):
  1. laser: inverse-bremsstrahlung ray tracing along each angular sector
     (each column of zones is traced as a locally spherical atmosphere with the
     1D tracer; the sector power follows the illumination pattern I(theta))
  2. 2D hydro in spherical polar coordinates: radially Lagrangian, angularly
     ALE with remap (hydro_sph.py)
  3. 3T conduction / exchange / radiation (diffusion2d.py, angular + radial sweeps)
  4. DT burn per zone; alpha transport along each sector with the 1D multi-angle
     slowing-down model
  5. EOS, diagnostics, next dt

Asymmetry seeds: illumination non-uniformity  I(theta) = 1 + sum_l a_l P_l(cos theta),
and surface perturbations  dr(theta) = sum_l eps_l P_l(cos theta)  on any material
interface (displacement tapered to zero at the neighbouring interfaces).

`half=True` simulates 0 <= theta <= pi/2 with a reflecting equatorial plane
(only even Legendre modes); tallies are always reported for the whole capsule.
"""
import math
import time as _time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from numba import njit, prange

from . import constants as C
from .diffusion2d import diffusion2d_step
from .eos import cold_energy, energies_from_T, eos_update, fermi_T
from .fusion import alpha_transport_core, burn_step, gauss_mu
from .hydro_sph import (cfl_sph, gauss_theta, hydro_sph_step, nodal_masses_sph, remap_theta,
                        sph_geometry, sph_volumes)
from .laser import make_rays, trace_rays
from .simulation import Laser, Options, build_mesh


@njit(parallel=True, cache=True)
def _laser_columns(rcol, ne, Te, Zeff, nc, b, w, Pcol, dump, dep):
    Ni, Nj = ne.shape
    out = np.zeros((Nj, 3))
    for j in prange(Nj):
        d = np.zeros(Ni)
        a, mi, es = trace_rays(rcol[:, j].copy(), ne[:, j].copy(), Te[:, j].copy(), Zeff[:, j].copy(),
                               nc, b, w, Pcol[j], dump, d)
        for i in range(Ni):
            dep[i, j] = d[i]
        out[j, 0] = a; out[j, 1] = mi; out[j, 2] = es
    return out


@njit(parallel=True, cache=True)
def _alpha_columns(rcol, Sa, ne, Te, ni, Ti, Z2b, Ab_mu, mus, wts, de, di):
    Ni, Nj = Sa.shape
    esc = np.zeros(Nj)
    for j in prange(Nj):
        e1 = np.zeros(Ni); i1 = np.zeros(Ni)
        esc[j] = alpha_transport_core(rcol[:, j].copy(), Sa[:, j].copy(), ne[:, j].copy(), Te[:, j].copy(),
                                      ni[:, j].copy(), Ti[:, j].copy(), Z2b[:, j].copy(), Ab_mu[:, j].copy(),
                                      mus, wts, e1, i1)
        for i in range(Ni):
            de[i, j] = e1[i]
            di[i, j] = i1[i]
    return esc


def legendre(l, x):
    return np.polynomial.legendre.legval(x, np.eye(l + 1)[l])


@dataclass
class Options2D(Options):
    gravity: float = 0.0          # uniform inward radial acceleration [cm/s^2] (test problems; not in E budget)
    dual_energy: bool = True      # ALE: pdV ion energy outside shocks (else total-energy based everywhere)
    dual_eta: float = 0.01        # ALE: internal / kinetic ratio below which the pdV ion energy is used
    shock_jump: float = 0.3       # ALE: relative pressure jump that marks a shock (dual energy)
    mesh_motion: str = "contact"  # ALE ring motion: "contact" (mean HLLC S*, Lagrangian in 1D) or "mass"
    shock_hlle: float = 0.5       # ALE: HLLE on spoke faces where the radial pressure jump exceeds this (0: off)
    mesh_relax: float = 5.0       # ALE: steps over which a ring relaxes to its initial enclosed mass (0: off)
    mesh_qmin: float = 0.2        # ALE: limit on neighbouring cell-size ratio relative to the initial mesh
    axis_tie: bool = True         # axis nodes move rigidly with their neighbour spoke
    hourglass_visc: float = 0.0   # viscous anti-hourglass on the angular sawtooth of u_r
    hourglass_order: int = 4      # 4 or 6: order of the difference that detects the angular sawtooth
    hourglass_ut: float = 0.5     # viscous damping of the radial zig-zag of u_theta along a spoke
    hourglass: float = 0.05       # anti-hourglass stiffness on the angular sawtooth of node radii
    remap_order: int = 2          # 1: donor cell, 2: van Leer limited angular remap
    core_rings: int = 0           # innermost rings treated as a quasi-1D core (no angular motion,
                                  # ring-averaged state); removes the angular CFL of the central cones
    n_picard: int = 2
    snap_float32: bool = True


class Simulation2D:
    def __init__(self, layers, n_theta=32, half=True, opts: Options2D = None, laser: Laser = None,
                 illumination: Optional[Dict[int, float]] = None,
                 perturbations: Optional[List[tuple]] = None):
        """layers: list of simulation.Layer (radial structure, as in 1D).
        illumination: {l: a_l} relative Legendre amplitudes of the laser intensity.
        perturbations: [(interface_index, l, amplitude_cm), ...]; interface k is the
        outer surface of layer k (k = len(layers)-1 is the capsule surface)."""
        self.o = opts or Options2D()
        self.laser = laser
        self.half = 1 if half else 0
        self.sym = 2.0 if half else 1.0
        self.illum = dict(illumination or {})
        r, mat_idx = build_mesh(layers)
        Ni = self.Ni = len(r) - 1
        Nj = self.Nj = n_theta
        self.materials = [L.material for L in layers]
        th = np.linspace(0.0, 0.5 * np.pi if half else np.pi, Nj + 1)
        self.theta = th
        self.theta_c = 0.5 * (th[1:] + th[:-1])
        rr = np.repeat(r[:, None], Nj + 1, 1)
        # interface perturbations
        layer_edges = np.concatenate([[0], np.cumsum([L.nzones for L in layers])])  # node index of interfaces
        for (k, l, eps) in (perturbations or []):
            i_s = layer_edges[k + 1]
            i_lo = layer_edges[k]
            i_hi = layer_edges[k + 2] if k + 2 < len(layer_edges) else i_s
            shape = eps * legendre(l, np.cos(th))
            for i in range(i_lo + 1, Ni + 1):
                if i <= i_s:
                    wgt = (r[i] - r[i_lo]) / (r[i_s] - r[i_lo])
                elif i_hi > i_s:
                    wgt = max(0.0, (r[i_hi] - r[i]) / (r[i_hi] - r[i_s]))
                else:
                    wgt = 1.0
                rr[i] += wgt * shape
        rr[0] = 0.0
        self.r = rr
        self.ur = np.zeros_like(rr)
        self.ut = np.zeros_like(rr)
        self.gs, self.gW = gauss_theta(th)
        self.mat = np.repeat(mat_idx[:, None], Nj, 1)
        M = [self.materials[k] for k in mat_idx]
        rep = lambda a: np.ascontiguousarray(np.repeat(np.array(a, float)[:, None], Nj, 1))
        self.Zb = rep([x.Z for x in M]); self.Z2b = rep([x.Z2 for x in M])
        self.Ab_mu = rep([x.A * C.mu for x in M]); self.rho0 = rep([x.rho0 for x in M])
        self.solid = np.ascontiguousarray(np.repeat(np.array([1 if x.solid else 0 for x in M], np.int64)[:, None], Nj, 1))
        self.B0 = rep([x.B0 for x in M]); self.fDT = rep([x.fDT for x in M]); self.opac = rep([x.opac_mult for x in M])
        self.deg = 1 if self.o.degeneracy else 0
        ne0 = self.Zb * self.rho0 / self.Ab_mu
        self.Pd0 = np.vectorize(lambda n: fermi_T(n, self.deg))(ne0) * ne0 * C.eV
        rho_init = rep([layers[k].rho or layers[k].material.rho0 for k in mat_idx])
        T0 = rep([layers[k].T0 for k in mat_idx])
        V = np.empty((Ni, Nj))
        sph_volumes(self.r, self.gs, self.gW, V)
        self.rho = rho_init.copy()
        self.m = self.rho * V
        self.Mnode = np.zeros_like(self.r)
        nodal_masses_sph(self.r, self.gs, self.gW, self.m, self.Mnode)
        self.qr = np.zeros((Ni, Nj)); self.qt = np.zeros((Ni, Nj))
        self.Te = T0.copy(); self.Ti = T0.copy()
        self.ee = np.empty((Ni, Nj)); self.ei = np.empty((Ni, Nj))
        n = Ni * Nj
        F = lambda a: a.reshape(n)
        energies_from_T(F(self.rho), F(self.Te), F(self.Ti), F(self.Zb), F(self.Ab_mu), self.deg, F(self.ee), F(self.ei))
        self.er = C.a_rad * self.Te ** 4 / self.rho if self.o.radiation else np.zeros((Ni, Nj))
        for k in ["Pe", "Pi", "Pc", "cs2", "Cve", "Cvi"]:
            setattr(self, k, np.zeros((Ni, Nj)))
        self._eos()
        self.yDT = self.fDT / self.Ab_mu
        self.yDT0 = self.yDT.copy()
        self.t = 0.0; self.dt = self.o.dt_init; self.nstep = 0
        self.E_laser = 0.0; self.E_abs = 0.0; self.E_missed = 0.0; self.E_escaped_laser = 0.0
        self.E_rad_lost = 0.0; self.E_fus = 0.0; self.E_alpha_dep = 0.0; self.E_alpha_esc = 0.0
        self.E_neutron = 0.0; self.N_reac = 0.0; self.E_visc = 0.0; self.E_remap_heat = 0.0
        self.E0 = self.total_energy()
        self.hist = {}
        self.snaps = []
        self.Pdep = np.zeros((Ni, Nj)); self.Sion = np.zeros((Ni, Nj))
        self.reac_rate = 0.0
        if self.o.alpha_n_mu > 0:
            self.mus, self.wmu = gauss_mu(self.o.alpha_n_mu)
        # illumination weights per sector (solid-angle fraction x relative intensity)
        dOm = np.cos(th[:-1]) - np.cos(th[1:])
        I = 1.0 + sum(a * legendre(l, np.cos(self.theta_c)) for l, a in self.illum.items())
        I = np.maximum(I, 0.0)
        self.sector_w = dOm * I / np.sum(dOm * I)
        self.wall = 0.0

    # ------------------------------------------------------------------ helpers
    def _flat(self, a):
        return a.reshape(self.Ni * self.Nj)

    def _eos(self):
        F = self._flat
        eos_update(F(self.rho), F(self.ee), F(self.ei), F(self.Zb), F(self.Ab_mu), F(self.rho0), F(self.Pd0),
                   F(self.B0), F(self.solid), self.deg, F(self.Te), F(self.Ti), F(self.Pe), F(self.Pi),
                   F(self.Pc), F(self.cs2), F(self.Cve), F(self.Cvi))

    def geometry(self):
        Ni, Nj = self.Ni, self.Nj
        V = np.empty((Ni, Nj)); rc = np.empty((Ni, Nj))
        Arad = np.empty((Ni + 1, Nj)); drad = np.empty((Ni + 1, Nj))
        Aang = np.empty((Ni, Nj + 1)); dang = np.empty((Ni, Nj + 1))
        sph_geometry(self.r, self.theta, self.gs, self.gW, V, Arad, drad, Aang, dang, rc)
        return dict(V=V, Arad=Arad, drad=drad, Aang=Aang, dang=dang, rc=rc,
                    zc=rc * np.cos(self.theta_c)[None, :], Rc=rc * np.sin(self.theta_c)[None, :])

    @property
    def z(self):
        return self.r * np.cos(self.theta)[None, :]

    @property
    def R(self):
        return self.r * np.sin(self.theta)[None, :]

    @property
    def P(self):
        return self.Pe + self.Pi + self.Pc + self.er * self.rho / 3.0

    def column_radii(self):
        """Node radii along each sector (Ni+1, Nj), made monotonic."""
        rcol = 0.5 * (self.r[:, 1:] + self.r[:, :-1])
        rcol[0] = 0.0
        rcol = np.maximum.accumulate(rcol + 1e-12 * np.arange(self.Ni + 1)[:, None], axis=0)
        return np.ascontiguousarray(rcol)

    def kinetic_energy(self):
        return 0.5 * np.sum(self.Mnode * (self.ur ** 2 + self.ut ** 2))

    def internal_energy(self):
        F = self._flat
        ec = cold_energy(F(self.rho), F(self.rho0), F(self.Pd0), F(self.B0), F(self.solid))
        return np.sum(F(self.m) * (F(self.ee) + F(self.ei) + ec))

    def radiation_energy(self):
        return np.sum(self.m * self.er)

    def total_energy(self):
        return self.kinetic_energy() + self.internal_energy() + self.radiation_energy()

    def energy_error(self):
        s = self.sym
        src = (self.E_abs + self.E_alpha_dep) / s
        sink = self.E_rad_lost / s
        err = self.total_energy() - self.E0 - src + sink
        return err / max(abs(self.E0), abs(self.total_energy()), src, 1e-30)

    def critical_radius(self, nc):
        """Sector-averaged radius of the critical surface."""
        ne = self.Zb * self.rho / self.Ab_mu
        rc = self.geometry()["rc"]
        out = []
        for j in range(self.Nj):
            idx = np.where(ne[:, j] >= nc)[0]
            out.append(rc[idx[-1], j] if len(idx) else 0.0)
        return float(np.sum(np.array(out) * self.sector_w))

    def column_rhoR(self, fuel_only=False):
        rcol = self.column_radii()
        dr = np.diff(rcol, axis=0)
        w = (self.fDT > 0.99) if fuel_only else np.ones_like(self.rho, bool)
        return np.sum(self.rho * dr * w, axis=0)

    def _domega(self):
        return np.cos(self.theta[:-1]) - np.cos(self.theta[1:])

    # ------------------------------------------------------------------ records
    def _record(self, P_laser, P_abs):
        h = self.hist
        s = self.sym
        fuel = self.fDT > 0.99
        dOm = self._domega()
        rr = self.column_rhoR(True)
        rad = self.column_rhoR(False)
        vals = dict(t=self.t, dt=self.dt, P_laser=P_laser, P_abs=P_abs, E_laser=self.E_laser, E_abs=self.E_abs,
                    KE=s * self.kinetic_energy(), IE=s * self.internal_energy(), Erad=s * self.radiation_energy(),
                    E_rad_lost=self.E_rad_lost, E_fus=self.E_fus, E_alpha_dep=self.E_alpha_dep, N_reac=self.N_reac,
                    reac_rate=self.reac_rate,
                    rhoR=float(np.sum(rad * dOm) / dOm.sum()),
                    rhoR_fuel=float(np.sum(rr * dOm) / dOm.sum()),
                    rhoR_fuel_min=float(rr.min()), rhoR_fuel_max=float(rr.max()),
                    rho_max=float(self.rho.max()), Ti_max=float(self.Ti.max()), Te_max=float(self.Te.max()))
        hs = fuel & (self.Ti > 1000.0)
        vals["Ti_hs"] = float(np.sum(self.m[hs] * self.Ti[hs]) / np.sum(self.m[hs])) if hs.any() else 0.0
        rn = self.r
        ur = self.ur
        vals["KE_in"] = s * float(0.5 * np.sum(self.Mnode * np.minimum(ur, 0.0) ** 2))
        vals["R_out"] = float(np.mean(rn[-1]))
        kf = np.where(fuel[:, 0])[0]
        vals["R_fuel"] = float(np.mean(rn[kf[-1] + 1])) if len(kf) else 0.0
        if len(kf):
            nodes = slice(1, kf[-1] + 1)
            mn = self.Mnode[nodes]
            vals["v_imp"] = float(-np.sum(mn * np.minimum(ur[nodes], 0)) / np.sum(mn))
            # fuel inner surface (ice/gas boundary, layer 1) shape: mean radius and P2/P4 amplitudes
            k1 = np.where(self.mat[:, 0] == 1)[0]
            k1 = k1[0] if len(k1) else kf[0]
            ri = rn[k1]
            th = self.theta
            dOmn = np.sin(th)
            r0 = np.trapezoid(ri * dOmn, th) / np.trapezoid(dOmn, th)
            vals["R_ice_in"] = float(r0)
            for l in (2, 4, 6):
                Pl = legendre(l, np.cos(th))
                vals[f"ice_in_P{l}"] = float(np.trapezoid((ri - r0) * Pl * dOmn, th) /
                                             np.trapezoid(Pl * Pl * dOmn, th) / max(r0, 1e-30))
        else:
            vals["v_imp"] = 0.0
        vals["u_min"] = float(ur.min())
        vals["E_err"] = self.energy_error()
        vals["R_crit"] = self.critical_radius(self.laser.nc) if self.laser is not None else 0.0
        for k, v in vals.items():
            h.setdefault(k, []).append(v)

    def snapshot(self):
        g = self.geometry()
        f = np.float32 if self.o.snap_float32 else np.float64
        burn = 1.0 - np.divide(self.yDT, self.yDT0, out=np.ones_like(self.yDT), where=self.yDT0 > 0)
        self.snaps.append(dict(t=self.t, z=self.z.astype(f), R=self.R.astype(f), zc=g["zc"].astype(f),
                               ur=self.ur.astype(f), ut=self.ut.astype(f),
                               Rc=g["Rc"].astype(f), rho=self.rho.astype(f), Te=self.Te.astype(f),
                               Ti=self.Ti.astype(f), P=self.P.astype(f), Pdep=self.Pdep.astype(f),
                               burn=burn.astype(f), m=self.m.astype(f)))

    def save(self, path):
        keys = ["z", "R", "zc", "Rc", "ur", "ut", "rho", "Te", "Ti", "P", "Pdep", "burn", "m"]
        np.savez_compressed(path, t=np.array([s["t"] for s in self.snaps]), mat=self.mat, theta=self.theta,
                            half=self.half, **{k: np.array([s[k] for s in self.snaps]) for k in keys})

    def history(self):
        return {k: np.array(v) for k, v in self.hist.items()}

    # ------------------------------------------------------------------ step
    def _laser(self, dt):
        """Laser deposition for this step (fills self.Pdep); returns (P_laser, P_abs)."""
        s = self.sym
        # 1. laser (power per sector; the domain receives 1/sym of the total)
        P_laser = 0.0; P_abs = 0.0
        if self.laser is not None:
            P_laser = self.laser.power(self.t + 0.5 * dt) * 1e7
            if P_laser > 0:
                Rb = self.laser.beam_radius(self.t) if self.laser.beam_radius else self._R_beam0
                key = round(Rb, 7)
                if getattr(self, "_ray_key", None) != key:
                    self._rays = make_rays(self.laser.n_rays, Rb, self.laser.profile_order)
                    self._ray_key = key
                b, w = self._rays
                ne = self.Zb * self.rho / self.Ab_mu
                out = _laser_columns(self.column_radii(), ne, self.Te, self.Z2b / self.Zb, self.laser.nc,
                                     b, w, P_laser / s * self.sector_w, self.laser.dump_frac, self.Pdep)
                P_abs = s * out[:, 0].sum()
                self.E_missed += s * out[:, 1].sum() * dt
                self.E_escaped_laser += s * out[:, 2].sum() * dt
            else:
                self.Pdep[:] = 0.0
            self.E_laser += P_laser * dt
            self.E_abs += P_abs * dt
        return P_laser, P_abs

    def _diffuse_burn(self, dt, P_abs):
        """3T diffusion with the laser source, then burn and alpha deposition."""
        o = self.o
        Ni, Nj = self.Ni, self.Nj
        s = self.sym
        # 3. diffusion
        if o.cond_e or o.cond_i or o.exchange or o.radiation or P_abs > 0:
            geo = self.geometry()
            Te0 = self.Te.copy(); Ti0 = self.Ti.copy()
            Er = self.er * self.rho
            Te = self.Te.copy(); Ti = self.Ti.copy()
            lost = diffusion2d_step(dt, geo, self.m, self.rho, Te, Ti, Er, self.Cve, self.Cvi, self.Zb,
                                    self.Z2b, self.Ab_mu, self.opac, self.Pdep, self.Sion, o.flux_limiter,
                                    o.cond_e, o.cond_i, o.exchange, o.radiation, o.n_picard, 1e-4,
                                    o.rad_spectral, o.kappa_mode, o.kappa0, o.kappa_n)
            self.ee += self.Cve * (Te - Te0)
            self.ei += self.Cvi * (Ti - Ti0)
            if o.radiation:
                self.er = Er / self.rho
                self.E_rad_lost += s * lost
            self._eos()
        # 4. burn
        self.reac_rate = 0.0
        if o.burn and self.Ti.max() > 300.0:
            F = self._flat
            Sa = np.zeros((Ni, Nj))
            nreac = s * burn_step(F(self.rho), F(self.m), F(self.Ti), F(self.yDT), dt, F(Sa))
            self.reac_rate = nreac / dt
            self.N_reac += nreac
            self.E_fus += nreac * C.E_fusion
            self.E_neutron += nreac * C.E_neutron
            if Sa.sum() > 0:
                if o.alpha_n_mu > 0:
                    de = np.zeros((Ni, Nj)); di = np.zeros((Ni, Nj))
                    ne = self.Zb * self.rho / self.Ab_mu
                    ni = self.rho / self.Ab_mu
                    esc = _alpha_columns(self.column_radii(), Sa, ne, self.Te, ni, self.Ti, self.Z2b,
                                         self.Ab_mu, self.mus, self.wmu, de, di).sum()
                else:
                    de = 0.7 * Sa; di = 0.3 * Sa; esc = 0.0
                self.ee += de / self.m
                self.ei += di / self.m
                self.E_alpha_dep += s * (de.sum() + di.sum())
                self.E_alpha_esc += s * esc
                self._eos()

    def step(self):
        o = self.o
        Ni, Nj = self.Ni, self.Nj
        dt = self.dt
        s = self.sym
        Te_old = self.Te.copy(); Ti_old = self.Ti.copy()
        V_old = self.m / self.rho
        P_laser, P_abs = self._laser(dt)
        # 2. hydro (Lagrangian phase + angular remap)
        if o.hydro:
            F = self._flat
            dVt = np.zeros((Ni, Nj + 1))
            self.E_visc += s * hydro_sph_step(dt, self.r, self.theta, self.gs, self.gW, self.ur, self.ut, self.m,
                                              self.Mnode, self.rho, self.ee, self.ei, self.er, F(self.Zb),
                                              F(self.Ab_mu), F(self.rho0), F(self.Pd0), F(self.B0), F(self.solid),
                                              self.deg, F(self.Te), F(self.Ti), F(self.Pe), F(self.Pi), F(self.Pc),
                                              F(self.cs2), F(self.Cve), F(self.Cvi), self.qr, self.qt, o.C1, o.C2,
                                              o.hourglass, self.half, o.fix_outer, dVt, o.core_rings, o.axis_tie,
                                              o.gravity, o.hourglass_ut, o.hourglass_order, o.hourglass_visc)
            if o.core_rings > 0:
                self._mix_core()
            if Nj > 1:
                self.E_remap_heat += s * remap_theta(self.r, self.theta, self.gs, self.gW, self.m, self.Mnode,
                                                     self.rho, self.ee, self.ei, self.er, self.yDT, self.ur,
                                                     self.ut, dVt, o.remap_order == 2, self.half)
                self._eos()
        self._diffuse_burn(dt, P_abs)
        self.t += dt
        self.nstep += 1
        # 5. next dt
        dtc = cfl_sph(self.r, self.theta, self.ur, self.ut, self.cs2, self.er, self.qr, self.qt, o.C2, o.cfl,
                      o.core_rings)
        rel = max(np.max(np.abs(self.Te - Te_old) / (Te_old + 1.0)),
                  np.max(np.abs(self.Ti - Ti_old) / (Ti_old + 1.0)))
        relV = np.max(np.abs(self.m / self.rho - V_old) / V_old)
        f = min(1.25, o.dT_rel / max(rel, 1e-12), o.dV_rel / max(relV, 1e-12))
        self.dt = max(min(dtc, dt * f, o.dt_max), o.dt_min)
        if self.laser is not None and hasattr(self.laser.power, "t_ns"):
            tb = self.laser.power.t_ns * 1e-9
            nxt = tb[tb > self.t + 1e-18]
            if len(nxt) and self.t + self.dt > nxt[0]:
                self.dt = max(nxt[0] - self.t, o.dt_min)
        return P_laser, P_abs

    def _mix_core(self):
        """Ring-average (mass weighted) the specific energies and fuel of the core rings."""
        k = self.o.core_rings
        m = self.m[:k]
        mt = m.sum(axis=1, keepdims=True)
        for a in (self.ee, self.ei, self.er, self.yDT):
            a[:k] = (np.sum(a[:k] * m, axis=1, keepdims=True) / mt) * np.ones((1, self.Nj))
        self._eos()

    def _snap_dt(self):
        sd = self.o.snap_dt
        return sd(self.t) if callable(sd) else sd

    def run(self, t_end, stop: Optional[Callable] = None, max_steps=10_000_000, wall_limit=None):
        o = self.o
        if self.laser is not None:
            self._R_beam0 = float(self.r[-1].mean())
        t0 = _time.time()
        self.snapshot(); self._next_snap = self.t + self._snap_dt()
        while self.t < t_end and self.nstep < max_steps:
            if self.t + self.dt > t_end:
                self.dt = t_end - self.t
            P_laser, P_abs = self.step()
            if self.nstep % o.history_every == 0:
                self._record(P_laser, P_abs)
            if self.t >= self._next_snap:
                self.snapshot()
                self._next_snap += self._snap_dt()
            if o.verbose and self.nstep % 1000 == 0:
                print(f"  step {self.nstep:8d} t={self.t*1e9:8.4f} ns dt={self.dt:.2e} "
                      f"rho_max={self.rho.max():8.2f} Ti_max={self.Ti.max()/1e3:7.3f} keV "
                      f"E_abs={self.E_abs/1e13:8.4f} MJ Yield={self.E_fus/1e13:9.4f} MJ "
                      f"Eerr={self.energy_error():+.2e}  ({_time.time()-t0:.0f}s)"
                      + (f" dt@{tuple(int(x) for x in self._dtloc)}" if hasattr(self, "_dtloc") else ""), flush=True)
            if stop is not None and stop(self):
                break
            if wall_limit and _time.time() - t0 > wall_limit:
                print("  wall-clock limit reached", flush=True)
                break
        self._record(0.0, 0.0)
        self.snapshot()
        self.wall = _time.time() - t0
        return self
