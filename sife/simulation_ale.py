"""SIFE-2D with the Godunov ALE hydro solver (hydro_ale.py).

Same physics modules and interface as simulation2d.Simulation2D (laser sectors,
3T diffusion, burn, alpha transport, diagnostics, snapshots); only the hydro step
and the state representation differ:

  * the rings are spheres r_i(t) moving with the ring-averaged mass velocity,
  * velocities are cell-centred (vr, vt),
  * materials are mass fractions X[k] (interfaces may cross cells; interface
    perturbations are initialised as volume fractions).
"""
import math
from typing import Dict, List, Optional

import numpy as np

from . import constants as C
from .eos import energies_from_T, fermi_T
from .hydro_ale import (QEC, QEE, QEI, QER, QP, QR, QUR, QUT, QX, QY, UE, UEE, UEI, UER, UM, UMR, UMT, UX, UY,
                        ale_cfl, ale_rates, radial_contact, regularize_mesh, cold_energy_cells, mesh_velocity, n_cons, n_prim, ring_geometry)
from .simulation import Laser
from .simulation2d import Options2D, Simulation2D, legendre
from .hydro_sph import nodal_masses_sph


class SimulationALE(Simulation2D):
    def __init__(self, layers, n_theta=32, half=True, opts: Options2D = None, laser: Laser = None,
                 illumination: Optional[Dict[int, float]] = None,
                 perturbations: Optional[List[tuple]] = None):
        super().__init__(layers, n_theta, half, opts, laser, illumination, None)
        o = self.o
        Ni, Nj = self.Ni, self.Nj
        self.rn = self.r[:, 0].copy()
        nm = self.nmat = len(layers)
        # material property tables
        mats = self.materials
        self._mZ = np.array([x.Z for x in mats]); self._mZ2 = np.array([x.Z2 for x in mats])
        self._mA = np.array([x.A * C.mu for x in mats]); self._mfDT = np.array([x.fDT for x in mats])
        self._mop = np.array([x.opac_mult for x in mats]); self._mrho0 = np.array([x.rho0 for x in mats])
        self._mB0 = np.array([x.B0 for x in mats]); self._msolid = np.array([1 if x.solid else 0 for x in mats])
        ne0 = self._mZ * self._mrho0 / self._mA
        self._mPd0 = np.array([fermi_T(n, self.deg) * n * C.eV for n in ne0])
        # layer outer radii, perturbed: R_k(theta) at the cell-centre angles
        edges = np.concatenate([[0], np.cumsum([L.nzones for L in layers])])
        Rk = np.array([self.rn[edges[k + 1]] for k in range(nm)])[:, None] * np.ones((1, Nj))
        for (k, l, eps) in (perturbations or []):
            Rk[k] += eps * legendre(l, np.cos(self.theta_c))
        rho_k = np.array([L.rho or L.material.rho0 for L in layers])
        T_k = np.array([L.T0 for L in layers])
        X = np.zeros((nm, Ni, Nj))
        mass = np.zeros((Ni, Nj))
        T = np.zeros((Ni, Nj))
        r3 = self.rn ** 3
        for k in range(nm):
            lo = (Rk[k - 1] if k > 0 else np.zeros(Nj)) ** 3
            hi = Rk[k] ** 3 if k < nm - 1 else np.full(Nj, np.inf)
            a = np.maximum(r3[:-1, None], lo[None, :])
            b = np.minimum(r3[1:, None], hi[None, :])
            f = np.clip(b - a, 0.0, None) / (r3[1:, None] - r3[:-1, None])
            X[k] = f * rho_k[k]
            mass += f * rho_k[k]
            T += f * T_k[k]
        # the outer surface is the mesh boundary (no vacuum cells): its displacement eps(theta)
        # becomes an equal areal-mass perturbation of the outermost cells, spread over a
        # layer at least 4 max|eps| thick
        epsR = Rk[nm - 1] - self.rn[-1]
        if np.any(epsR != 0.0):
            D = 4.0 * np.max(np.abs(epsR))
            i0 = max(int(np.searchsorted(self.rn, self.rn[-1] - D)) - 1, edges[nm - 1])
            Rin = self.rn[i0]
            fac = ((self.rn[-1] + epsR) ** 3 - Rin ** 3) / (self.rn[-1] ** 3 - Rin ** 3)
            X[:, i0:] *= fac[None, None, :]
            mass[i0:] *= fac[None, :]
        self.X = X / mass[None]
        self.rho = mass                          # mass-weighted mixture density
        V = self._volumes(self.rn)
        self.m = self.rho * V
        self._mix_props()
        self.Te = T.copy(); self.Ti = T.copy()
        n = Ni * Nj
        F = lambda a: a.reshape(n)
        energies_from_T(F(self.rho), F(self.Te), F(self.Ti), F(self.Zb), F(self.Ab_mu), self.deg, F(self.ee),
                        F(self.ei))
        self.er = C.a_rad * self.Te ** 4 / self.rho if o.radiation else np.zeros((Ni, Nj))
        self._eos()
        self.yDT = self.fDT / self.Ab_mu
        self.yDT0 = self.yDT.copy()
        self.vr = np.zeros((Ni, Nj)); self.vt = np.zeros((Ni, Nj))
        self.E_out = 0.0; self.M_out = 0.0
        dr0 = np.diff(self.rn)
        self._M0 = np.cumsum(self.m.sum(1))[:-1]
        self._ratio0 = dr0[:-1] / dr0[1:]
        self.E0 = self.total_energy()
        self.hist = {}
        self.snaps = []

    # ------------------------------------------------------------------ state helpers
    def _volumes(self, rn):
        dmu = np.cos(self.theta[:-1]) - np.cos(self.theta[1:])
        return 2 * np.pi / 3 * np.diff(rn ** 3)[:, None] * dmu[None, :]

    def _mix_props(self):
        """Mixture properties from the mass fractions (ions per gram add up);
        cold-curve parameters of the dominant material."""
        X = self.X
        nA = X / self._mA[:, None, None]
        s = nA.sum(axis=0)
        self.Ab_mu = 1.0 / s
        self.Zb = np.einsum("k,kij->ij", self._mZ, nA) / s
        self.Z2b = np.einsum("k,kij->ij", self._mZ2, nA) / s
        self.fDT = np.einsum("k,kij->ij", self._mfDT, nA) / s
        self.opac = np.einsum("k,kij->ij", self._mop, X)
        dom = np.argmax(X, axis=0)
        self.mat = dom
        self.rho0 = self._mrho0[dom]; self.Pd0 = self._mPd0[dom]
        self.B0 = self._mB0[dom]; self.solid = np.ascontiguousarray(self._msolid[dom].astype(np.int64))
        for a in ("Ab_mu", "Zb", "Z2b", "fDT", "opac", "rho0", "Pd0", "B0"):
            setattr(self, a, np.ascontiguousarray(getattr(self, a)))

    def kinetic_energy(self):
        if not hasattr(self, "vr"):              # during the base-class constructor
            return super().kinetic_energy()
        return 0.5 * np.sum(self.m * (self.vr ** 2 + self.vt ** 2))

    def energy_error(self):
        if not hasattr(self, "E_out"):
            return super().energy_error()
        s = self.sym
        src = (self.E_abs + self.E_alpha_dep) / s
        sink = (self.E_rad_lost + self.E_out) / s
        err = self.total_energy() - self.E0 - src + sink
        return err / max(abs(self.E0), abs(self.total_energy()), src, 1e-30)

    def column_rhoR(self, fuel_only=False):
        dr = np.diff(self.rn)[:, None]
        w = self.fDT if fuel_only else 1.0
        return np.sum(self.rho * dr * w, axis=0)

    def _prims(self):
        """Primitive array Q and sound speed for the hydro solver."""
        Ni, Nj = self.Ni, self.Nj
        Q = np.empty((n_prim(self.nmat), Ni, Nj))
        ec = np.empty((Ni, Nj))
        cold_energy_cells(self.rho, self.rho0, self.Pd0, self.B0, self.solid, ec)
        Q[QR] = self.rho; Q[QUR] = self.vr; Q[QUT] = self.vt; Q[QP] = self.P
        Q[QEE] = self.ee; Q[QEI] = self.ei; Q[QER] = self.er; Q[QEC] = ec; Q[QY] = self.yDT
        Q[QX:] = self.X
        c = np.sqrt(np.maximum(self.cs2, 0.0) + 4.0 / 9.0 * self.er + 1e-30)
        return Q, c

    def _cons(self):
        Ni, Nj = self.Ni, self.Nj
        U = np.empty((n_cons(self.nmat), Ni, Nj))
        ec = np.empty((Ni, Nj))
        cold_energy_cells(self.rho, self.rho0, self.Pd0, self.B0, self.solid, ec)
        m = self.m
        U[UM] = m; U[UMR] = m * self.vr; U[UMT] = m * self.vt
        U[UE] = m * (self.ee + self.ei + self.er + ec + 0.5 * (self.vr ** 2 + self.vt ** 2))
        U[UEE] = m * self.ee; U[UEI] = m * self.ei; U[UER] = m * self.er; U[UY] = m * self.yDT
        U[UX:] = m[None] * self.X
        return U

    def _recover(self, U, V, shock):
        """Cell state from conserved totals (in place); dual energy for the ions."""
        o = self.o
        m = np.maximum(U[UM], 1e-300)
        self.m = m
        self.rho = m / V
        self.vr = U[UMR] / m; self.vt = U[UMT] / m
        X = np.maximum(U[UX:], 0.0)
        self.X = X / np.maximum(X.sum(axis=0), 1e-300)[None]
        self._mix_props()
        self.yDT = np.maximum(U[UY] / m, 0.0)
        ec = np.empty_like(m)
        cold_energy_cells(self.rho, self.rho0, self.Pd0, self.B0, self.solid, ec)
        ee = U[UEE] / m
        er = np.maximum(U[UER] / m, 0.0) if o.radiation else np.zeros_like(m)
        ei_tot = U[UE] / m - 0.5 * (self.vr ** 2 + self.vt ** 2) - ee - er - ec
        ei_pdv = U[UEI] / m
        # dual energy: the separately advected (pdV) ion energy is used only where the
        # flow is kinetic-energy dominated and smooth (cold imploding shell), where
        # E - KE carries the truncation error of KE
        ke = 0.5 * (self.vr ** 2 + self.vt ** 2)
        use_pdv = (~shock) & (ee + ei_tot + er < o.dual_eta * ke)
        ei = np.where(use_pdv, ei_pdv, ei_tot)
        # floors: a cold (1 meV) electron / ion gas at this density
        n = m.size
        F = lambda a: a.reshape(n)
        Tf = np.full_like(m, 1e-3)
        eef = np.empty_like(m); eif = np.empty_like(m)
        energies_from_T(F(self.rho), F(Tf), F(Tf), F(self.Zb), F(self.Ab_mu), self.deg, F(eef), F(eif))
        self.ee = np.maximum(ee, eef)
        self.ei = np.maximum(ei, eif)
        self.er = er
        self._eos()

    def _shock_flag(self, div, Q):
        P = Q[QP]
        jump = np.zeros_like(P)
        jump[1:] = np.maximum(jump[1:], np.abs(P[1:] - P[:-1]))
        jump[:-1] = np.maximum(jump[:-1], np.abs(P[1:] - P[:-1]))
        jump[:, 1:] = np.maximum(jump[:, 1:], np.abs(P[:, 1:] - P[:, :-1]))
        jump[:, :-1] = np.maximum(jump[:, :-1], np.abs(P[:, 1:] - P[:, :-1]))
        if not self.o.dual_energy:
            return np.ones(P.shape, bool)
        return (div < 0.0) & (jump > self.o.shock_jump * np.maximum(np.abs(P), 1e-300))

    def _mix_core_ale(self):
        k = self.o.core_rings
        if k <= 0:
            return
        m = self.m[:k]
        mt = m.sum(axis=1, keepdims=True)
        V = self._volumes(self.rn)[:k]
        ke0 = 0.5 * np.sum(m * (self.vr[:k] ** 2 + self.vt[:k] ** 2), axis=1, keepdims=True)
        avg = lambda a: np.sum(a[:k] * m, axis=1, keepdims=True) / mt
        vr = avg(self.vr)
        for a in (self.ee, self.ei, self.er, self.yDT):
            a[:k] = avg(a) * np.ones((1, self.Nj))
        for q in range(self.nmat):
            self.X[q, :k] = avg(self.X[q]) * np.ones((1, self.Nj))
        self.vr[:k] = vr * np.ones((1, self.Nj)); self.vt[:k] = 0.0
        rho = mt / V.sum(axis=1, keepdims=True)
        self.rho[:k] = rho * np.ones((1, self.Nj))
        self.m[:k] = self.rho[:k] * V
        ke1 = 0.5 * np.sum(self.m[:k] * self.vr[:k] ** 2, axis=1, keepdims=True)
        self.ei[:k] += (ke0 - ke1) / mt
        self._mix_props()
        self._eos()

    # ------------------------------------------------------------------ hydro step
    def _mesh_w(self, Q=None, c=None):
        o = self.o
        if o.mesh_motion == "contact":
            if Q is None:
                Q, c = self._prims()
            w = np.empty(self.Ni + 1)
            radial_contact(Q, c, self.half, o.core_rings, o.remap_order == 2, self.theta, w)
            if o.mesh_relax > 0 and hasattr(self, "_M0") and self.dt > 0:
                # pull each ring back toward the radius that encloses its initial mass, when it has drifted more than a quarter cell
                # (averaged over angle): stops slow drift and odd-even ring spacing
                # where columns disagree (shock fronts not aligned with rings)
                Ni = self.Ni
                M = np.cumsum(self.m.sum(1))[:-1]
                dmu = np.cos(self.theta[:-1]) - np.cos(self.theta[1:])
                rr = self.rn[1:Ni]
                rho_f = 0.5 * (self.rho[:-1] + self.rho[1:])
                dMdr = 2 * np.pi * rr ** 2 * (rho_f * dmu[None, :]).sum(1)
                dr = np.diff(self.rn)
                dmin = np.minimum(dr[:-1], dr[1:])
                x = (self._M0 - M) / dMdr          # ring displacement to its initial enclosed mass
                x = np.sign(x) * np.maximum(np.abs(x) - 0.25 * dmin, 0.0)   # dead band: act only on large drift
                d = np.clip(x / o.mesh_relax, -0.1 * dmin, 0.1 * dmin)
                w[1:Ni] += d / self.dt
            if o.mesh_qmin > 0:
                regularize_mesh(self.rn, w, self._ratio0, o.mesh_qmin)
            return w
        return mesh_velocity(self.rn, self.m, self.vr, self._ratio0, o.mesh_qmin)

    def _hydro(self, dt):
        o = self.o
        s = self.sym
        Ni, Nj = self.Ni, self.Nj
        Q, c = self._prims()
        w = self._mesh_w(Q, c)
        if o.fix_outer:
            w[-1] = 0.0
        rn1 = self.rn + w * dt
        rn1[0] = 0.0
        V0 = np.empty((Ni, Nj)); V1 = np.empty((Ni, Nj)); Aeff = np.empty((Ni + 1, Nj))
        Seff = np.empty((Ni, Nj + 1)); rc = np.empty((Ni, Nj))
        ring_geometry(self.rn, rn1, self.theta, V0, V1, Aeff, Seff, rc)
        Vm = 0.5 * (V0 + V1)
        U0 = self._cons()
        nu = U0.shape[0]
        L1 = np.empty_like(U0); L2 = np.empty_like(U0)
        div = np.empty((Ni, Nj)); fo1 = np.empty(nu); fo2 = np.empty(nu)
        second = o.remap_order == 2
        Pr = self.er * self.rho / 3.0
        ale_rates(Q, c, self.Pe, self.Pi, Pr, w, Aeff, Seff, Vm, rc, self.half, o.core_rings, second,
                  o.gravity, o.fix_outer, o.shock_hlle, L1, div, fo1)
        sh = self._shock_flag(div, Q)
        self._recover(U0 + dt * L1, V1, sh)
        Q2, c2 = self._prims()
        Pr = self.er * self.rho / 3.0
        ale_rates(Q2, c2, self.Pe, self.Pi, Pr, w, Aeff, Seff, Vm, rc, self.half, o.core_rings, second,
                  o.gravity, o.fix_outer, o.shock_hlle, L2, div, fo2)
        sh = sh | self._shock_flag(div, Q2)
        self.rn = rn1
        self.r = np.ascontiguousarray(np.repeat(rn1[:, None], Nj + 1, 1))
        self._recover(U0 + 0.5 * dt * (L1 + L2), V1, sh)
        self._mix_core_ale()
        fo = 0.5 * (fo1 + fo2)
        self.E_out += s * fo[UE] * dt
        self.M_out += s * fo[UM] * dt
        self._w = w

    # ------------------------------------------------------------------ step
    def step(self):
        o = self.o
        Ni, Nj = self.Ni, self.Nj
        dt = self.dt
        s = self.sym
        Te_old = self.Te.copy(); Ti_old = self.Ti.copy()
        P_laser, P_abs = self._laser(dt)
        if o.hydro:
            self._hydro(dt)
        self._diffuse_burn(dt, P_abs)
        self.t += dt
        self.nstep += 1
        rc = np.repeat((0.5 * (self.rn[1:] + self.rn[:-1]))[:, None], Nj, 1)
        c = np.sqrt(np.maximum(self.cs2, 0.0) + 4.0 / 9.0 * self.er + 1e-30)
        w = self._mesh_w()
        self._dtloc = np.zeros(3, np.int64)
        dtc = ale_cfl(self.rn, rc, self.theta, self.vr, self.vt, c, w, o.core_rings, o.cfl, self._dtloc)
        rel = max(np.max(np.abs(self.Te - Te_old) / (Te_old + 1.0)),
                  np.max(np.abs(self.Ti - Ti_old) / (Ti_old + 1.0)))
        f = min(1.25, o.dT_rel / max(rel, 1e-12))
        self.dt = max(min(dtc, dt * f, o.dt_max), o.dt_min)
        if self.laser is not None and hasattr(self.laser.power, "t_ns"):
            tb = self.laser.power.t_ns * 1e-9
            nxt = tb[tb > self.t + 1e-18]
            if len(nxt) and self.t + self.dt > nxt[0]:
                self.dt = max(nxt[0] - self.t, o.dt_min)
        return P_laser, P_abs

    # ------------------------------------------------------------------ records
    def interface_radius(self, k):
        """Radius per column where the mass fraction of materials <= k falls to 1/2
        (interface k = outer surface of layer k)."""
        f = self.X[:k + 1].sum(axis=0)
        out = np.empty(self.Nj)
        rn = self.rn
        for j in range(self.Nj):
            col = f[:, j]
            idx = np.where(col < 0.5)[0]
            if len(idx) == 0:
                out[j] = rn[-1]; continue
            i = idx[0]
            if i == 0:
                out[j] = 0.0; continue
            # linear in the fraction between cell centres i-1 and i
            ra = 0.5 * (rn[i - 1] + rn[i]); rb = 0.5 * (rn[i] + rn[i + 1])
            fa, fb = col[i - 1], col[i]
            out[j] = ra + (fa - 0.5) / max(fa - fb, 1e-30) * (rb - ra)
        return out

    def legendre_amp(self, rj, l):
        th = self.theta_c
        dOm = self._domega()
        r0 = np.sum(rj * dOm) / dOm.sum()
        Pl = legendre(l, np.cos(th))
        return float(np.sum((rj - r0) * Pl * dOm) / np.sum(Pl * Pl * dOm)), float(r0)

    def _record(self, P_laser, P_abs):
        h = self.hist
        s = self.sym
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
        fw = self.fDT * self.m
        hs = (self.fDT > 0.5) & (self.Ti > 1000.0)
        vals["Ti_hs"] = float(np.sum(self.m[hs] * self.Ti[hs]) / np.sum(self.m[hs])) if hs.any() else 0.0
        vals["KE_in"] = s * float(0.5 * np.sum(self.m * np.minimum(self.vr, 0.0) ** 2))
        vals["R_out"] = float(self.rn[-1])
        vals["v_imp"] = float(-np.sum(fw * np.minimum(self.vr, 0.0)) / max(np.sum(fw), 1e-300))
        nL = self.nmat
        rf = self.interface_radius(min(1, nL - 1))
        vals["R_fuel"] = float(np.sum(rf * dOm) / dOm.sum())
        ri = self.interface_radius(0)
        a0 = None
        for l in (2, 4, 6):
            a, r0 = self.legendre_amp(ri, l)
            vals[f"ice_in_P{l}"] = a / max(r0, 1e-30)
            a0 = r0
        vals["R_ice_in"] = a0
        vals["u_min"] = float(self.vr.min())
        vals["E_err"] = self.energy_error()
        vals["R_crit"] = self.critical_radius(self.laser.nc) if self.laser is not None else 0.0
        for k, v in vals.items():
            h.setdefault(k, []).append(v)

    def snapshot(self):
        g = self.geometry()
        f = np.float32 if self.o.snap_float32 else np.float64
        burn = 1.0 - np.divide(self.yDT, self.yDT0, out=np.ones_like(self.yDT), where=self.yDT0 > 0)
        self.snaps.append(dict(t=self.t, z=self.z.astype(f), R=self.R.astype(f), zc=g["zc"].astype(f),
                               ur=self.vr.astype(f), ut=self.vt.astype(f),
                               Rc=g["Rc"].astype(f), rho=self.rho.astype(f), Te=self.Te.astype(f),
                               Ti=self.Ti.astype(f), P=self.P.astype(f), Pdep=self.Pdep.astype(f),
                               burn=burn.astype(f), m=self.m.astype(f), X=self.X.astype(f)))

    def save(self, path):
        keys = ["z", "R", "zc", "Rc", "ur", "ut", "rho", "Te", "Ti", "P", "Pdep", "burn", "m", "X"]
        np.savez_compressed(path, t=np.array([s["t"] for s in self.snaps]), mat=self.mat, theta=self.theta,
                            half=self.half, **{k: np.array([s[k] for s in self.snaps]) for k in keys})
