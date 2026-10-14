"""SIFE driver: 1D Lagrangian three-temperature radiation hydrodynamics with
laser ray tracing, DT burn and alpha transport.

One time step (operator split):
  1. laser ray trace on the current state      -> P_dep (electrons)
  2. Lagrangian hydro (predictor-corrector)    -> r, u, rho, e_e, e_i, e_r
  3. implicit 3T conduction/exchange/radiation  (laser source included)
  4. DT burn + alpha slowing-down transport     -> e_e, e_i, fuel depletion
  5. EOS update, diagnostics, next dt
"""
import math
import time as _time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from . import constants as C
from .diffusion import diffusion_step
from .eos import cold_energy, energies_from_T, eos_update, fermi_T
from .fusion import alpha_transport_core, burn_step, gauss_mu
from .hydro import cfl_dt, hydro_step, viscosity, volumes
from .laser import critical_density, make_rays, trace_rays


# --------------------------------------------------------------------------- mesh
@dataclass
class Layer:
    material: object
    thickness: float          # cm
    nzones: int
    ratio: float = 1.0        # dr_{k+1}/dr_k inside the layer (outward)
    T0: float = 1.6e-3        # initial temperature [eV]
    rho: Optional[float] = None
    fine_in: float = 1.0      # if > 1: zones grow geometrically by this factor from the inner edge
    fine_out: float = 1.0     # if > 1: zones grow geometrically by this factor from the outer edge


def build_mesh(layers: List[Layer], r_inner=0.0):
    r = [r_inner]
    mat_idx = []
    for li, L in enumerate(layers):
        n = L.nzones
        if L.fine_in > 1.0 or L.fine_out > 1.0:
            k = np.arange(n)
            w = np.minimum(L.fine_in ** k, L.fine_out ** (n - 1 - k))
            dr = L.thickness * w / w.sum()
        elif abs(L.ratio - 1.0) < 1e-12:
            dr = np.full(n, L.thickness / n)
        else:
            dr0 = L.thickness * (L.ratio - 1.0) / (L.ratio ** n - 1.0)
            dr = dr0 * L.ratio ** np.arange(n)
        for d in dr:
            r.append(r[-1] + d)
            mat_idx.append(li)
    return np.array(r), np.array(mat_idx, dtype=np.int64)


# --------------------------------------------------------------------------- laser
@dataclass
class Laser:
    wavelength_um: float = 0.248                  # KrF
    power: Callable[[float], float] = None        # W as a function of t [s]
    n_rays: int = 64
    beam_radius: Optional[Callable[[float], float]] = None  # cm; None -> 'zoom' to initial radius
    profile_order: float = 4.0
    dump_frac: float = 0.0

    @property
    def nc(self):
        return critical_density(self.wavelength_um)


def tabulated_pulse(t_ns, P_TW):
    """Piecewise-linear pulse from (t [ns], P [TW]) tables -> P(t[s]) in W."""
    t_ns = np.asarray(t_ns, float)
    P_TW = np.asarray(P_TW, float)

    def P(t):
        return float(np.interp(t * 1e9, t_ns, P_TW, left=0.0, right=0.0)) * 1e12
    P.t_ns = t_ns
    P.P_TW = P_TW
    return P


# --------------------------------------------------------------------------- options
@dataclass
class Options:
    geom: int = 2
    hydro: bool = True
    degeneracy: bool = True
    cond_e: bool = True
    cond_i: bool = True
    exchange: bool = True
    radiation: bool = True
    rad_spectral: bool = True   # two-temperature (Te, Tr) free-free absorption correction
    burn: bool = True
    alpha_n_mu: int = 16        # 0 -> local deposition
    flux_limiter: float = 0.06
    C1: float = 0.25
    C2: float = 1.5
    cfl: float = 0.3
    dT_rel: float = 0.15
    dV_rel: float = 0.08
    dt_init: float = 1e-14
    dt_max: float = 2e-12
    dt_min: float = 1e-19
    n_picard: int = 3
    kappa_mode: int = 0         # 1 -> kappa = kappa0 T^n (benchmarks)
    kappa0: float = 0.0
    kappa_n: float = 2.5
    fix_outer: bool = False
    P_ext: float = 0.0
    history_every: int = 10
    snap_dt: float = 0.1e-9     # or a callable t -> interval (denser frames for animations)
    verbose: int = 1


class Simulation:
    def __init__(self, layers=None, opts: Options = None, laser: Laser = None,
                 r=None, mat_idx=None, materials=None, rho=None, Te=None, Ti=None, u=None):
        self.o = opts or Options()
        self.laser = laser
        if layers is not None:
            r, mat_idx = build_mesh(layers)
            materials = [L.material for L in layers]
            rho = np.array([layers[i].rho or layers[i].material.rho0 for i in mat_idx])
            Te = np.array([layers[i].T0 for i in mat_idx])
            Ti = Te.copy()
        self.materials = materials
        self.r = np.array(r, float)
        N = self.N = len(self.r) - 1
        self.mat = np.array(mat_idx, dtype=np.int64)
        M = [materials[i] for i in self.mat]
        self.Zb = np.array([x.Z for x in M])
        self.Z2b = np.array([x.Z2 for x in M])
        self.Ab_mu = np.array([x.A * C.mu for x in M])
        self.rho0 = np.array([x.rho0 for x in M])
        self.solid = np.array([1 if x.solid else 0 for x in M], dtype=np.int64)
        self.B0 = np.array([x.B0 for x in M])
        self.fDT = np.array([x.fDT for x in M])
        self.opac = np.array([x.opac_mult for x in M])
        self.deg = 1 if self.o.degeneracy else 0
        ne0 = self.Zb * self.rho0 / self.Ab_mu
        self.Pd0 = np.array([fermi_T(n, self.deg) for n in ne0]) * ne0 * C.eV
        V = np.empty(N)
        volumes(self.r, self.o.geom, V)
        self.rho = np.array(rho, float)
        self.m = self.rho * V
        self.mnode = np.zeros(N + 1)
        self.mnode[:-1] += 0.5 * self.m
        self.mnode[1:] += 0.5 * self.m
        self.u = np.zeros(N + 1) if u is None else np.array(u, float)
        self.Te = np.array(Te, float)
        self.Ti = np.array(Ti, float)
        self.ee = np.empty(N)
        self.ei = np.empty(N)
        energies_from_T(self.rho, self.Te, self.Ti, self.Zb, self.Ab_mu, self.deg, self.ee, self.ei)
        self.er = np.zeros(N)
        if self.o.radiation:
            self.er = C.a_rad * self.Te ** 4 / self.rho
        self.Pe = np.empty(N); self.Pi = np.empty(N); self.Pc = np.empty(N)
        self.cs2 = np.empty(N); self.Cve = np.empty(N); self.Cvi = np.empty(N)
        self.q = np.zeros(N)
        self._eos()
        self.yDT = self.fDT / self.Ab_mu         # DT ions per gram
        self.yDT0 = self.yDT.copy()
        self.t = 0.0
        self.dt = self.o.dt_init
        self.nstep = 0
        # tallies [erg]
        self.E_laser = 0.0; self.E_abs = 0.0; self.E_missed = 0.0; self.E_escaped_laser = 0.0
        self.E_rad_lost = 0.0; self.E_fus = 0.0; self.E_alpha_dep = 0.0
        self.E_alpha_esc = 0.0; self.E_neutron = 0.0; self.N_reac = 0.0
        self.E_bnd = 0.0
        self.E0 = self.total_energy()
        self.hist = {k: [] for k in self._hist_keys()}
        self.snaps = []
        self._next_snap = 0.0
        self.Pdep = np.zeros(N)
        self.Sion = np.zeros(N)
        self.reac_rate = 0.0
        if self.o.alpha_n_mu > 0:
            self.mus, self.wmu = gauss_mu(self.o.alpha_n_mu)

    # ------------------------------------------------------------------ helpers
    def _eos(self):
        eos_update(self.rho, self.ee, self.ei, self.Zb, self.Ab_mu, self.rho0, self.Pd0, self.B0,
                   self.solid, self.deg, self.Te, self.Ti, self.Pe, self.Pi, self.Pc,
                   self.cs2, self.Cve, self.Cvi)

    @property
    def rc(self):
        return 0.5 * (self.r[1:] + self.r[:-1])

    @property
    def P(self):
        return self.Pe + self.Pi + self.Pc + self.er * self.rho / 3.0

    def kinetic_energy(self):
        return 0.5 * np.sum(self.mnode * self.u ** 2)

    def internal_energy(self):
        ec = cold_energy(self.rho, self.rho0, self.Pd0, self.B0, self.solid)
        return np.sum(self.m * (self.ee + self.ei + ec))

    def radiation_energy(self):
        return np.sum(self.m * self.er)

    def total_energy(self):
        return self.kinetic_energy() + self.internal_energy() + self.radiation_energy()

    def energy_error(self):
        """(E_now - E0 - sources + sinks) / max(E_now, E_abs)."""
        src = self.E_abs + self.E_alpha_dep + self.E_bnd
        sink = self.E_rad_lost
        err = self.total_energy() - self.E0 - src + sink
        return err / max(abs(self.E0), abs(self.total_energy()), src, 1e-30)

    def critical_radius(self, nc):
        """Outermost radius where ne crosses nc (log-linear interpolation between zone centres)."""
        ne = self.Zb * self.rho / self.Ab_mu
        idx = np.where(ne >= nc)[0]
        if len(idx) == 0:
            return 0.0
        k = idx[-1]
        if k == self.N - 1:
            return float(self.r[-1])
        rc = self.rc
        f = np.log(ne[k] / nc) / max(np.log(ne[k] / ne[k + 1]), 1e-30)
        return float(rc[k] + f * (rc[k + 1] - rc[k]))

    def areal_density(self, fuel_only=False):
        dr = np.diff(self.r)
        w = self.fDT > 0.99 if fuel_only else np.ones(self.N, bool)
        return float(np.sum(self.rho[w] * dr[w]))

    def _hist_keys(self):
        return ["t", "dt", "P_laser", "P_abs", "E_laser", "E_abs", "KE", "IE", "Erad", "E_rad_lost",
                "E_fus", "E_alpha_dep", "N_reac", "reac_rate", "rhoR", "rhoR_fuel", "rho_max",
                "Ti_max", "Te_max", "Ti_hs", "KE_in", "R_out", "R_fuel", "u_min", "v_imp", "E_err",
                "R_crit"]

    def _record(self, P_laser, P_abs):
        h = self.hist
        fuel = self.fDT > 0.99
        h["t"].append(self.t); h["dt"].append(self.dt)
        h["P_laser"].append(P_laser); h["P_abs"].append(P_abs)
        h["E_laser"].append(self.E_laser); h["E_abs"].append(self.E_abs)
        h["KE"].append(self.kinetic_energy()); h["IE"].append(self.internal_energy())
        h["Erad"].append(self.radiation_energy()); h["E_rad_lost"].append(self.E_rad_lost)
        h["E_fus"].append(self.E_fus); h["E_alpha_dep"].append(self.E_alpha_dep)
        h["N_reac"].append(self.N_reac); h["reac_rate"].append(self.reac_rate)
        h["rhoR"].append(self.areal_density()); h["rhoR_fuel"].append(self.areal_density(True))
        h["rho_max"].append(float(self.rho.max()))
        h["KE_in"].append(float(0.5 * np.sum(self.mnode * np.minimum(self.u, 0.0) ** 2)))
        h["Ti_max"].append(float(self.Ti.max())); h["Te_max"].append(float(self.Te.max()))
        hs = fuel & (self.Ti > 1000.0)
        h["Ti_hs"].append(float(np.sum(self.m[hs] * self.Ti[hs]) / np.sum(self.m[hs])) if hs.any() else 0.0)
        h["R_out"].append(float(self.r[-1]))
        kf = np.where(fuel)[0]
        h["R_fuel"].append(float(self.r[kf[-1] + 1]) if len(kf) else 0.0)
        h["u_min"].append(float(self.u.min()))
        # implosion velocity: mass-averaged inward velocity of the dense unablated shell
        if len(kf):
            nodes = np.arange(1, kf[-1] + 1)
            un = self.u[nodes]
            mn = self.mnode[nodes]
            h["v_imp"].append(float(-np.sum(mn * np.minimum(un, 0)) / np.sum(mn)))
        else:
            h["v_imp"].append(0.0)
        h["E_err"].append(self.energy_error())
        if self.laser is not None:
            h["R_crit"].append(self.critical_radius(self.laser.nc))
        else:
            h["R_crit"].append(0.0)

    def snapshot(self):
        self.snaps.append(dict(t=self.t, r=self.r.copy(), u=self.u.copy(), rho=self.rho.copy(),
                               Te=self.Te.copy(), Ti=self.Ti.copy(), Tr=(self.er * self.rho / C.a_rad) ** 0.25,
                               P=self.P.copy(), mat=self.mat.copy(), Pdep=self.Pdep.copy(),
                               burn=1.0 - np.divide(self.yDT, self.yDT0, out=np.ones(self.N), where=self.yDT0 > 0)))

    # ------------------------------------------------------------------ step
    def step(self):
        o = self.o
        N = self.N
        dt = self.dt
        Te_old = self.Te.copy(); Ti_old = self.Ti.copy()
        V_old = self.m / self.rho
        # 1. laser
        P_laser = 0.0
        P_abs = 0.0
        if self.laser is not None:
            P_laser = self.laser.power(self.t + 0.5 * dt) * 1e7  # W -> erg/s
            if P_laser > 0.0:
                Rb = self.laser.beam_radius(self.t) if self.laser.beam_radius else self._R_beam0
                b, w = make_rays(self.laser.n_rays, Rb, self.laser.profile_order)
                ne = self.Zb * self.rho / self.Ab_mu
                ab, mi, es = trace_rays(self.r, ne, self.Te, self.Z2b / self.Zb, self.laser.nc,
                                        b, w, P_laser, self.laser.dump_frac, self.Pdep)
                P_abs = ab
                self.E_missed += mi * dt
                self.E_escaped_laser += es * dt
            else:
                self.Pdep[:] = 0.0
            self.E_laser += P_laser * dt
            self.E_abs += P_abs * dt
        # 2. hydro
        if o.hydro:
          self.E_bnd += hydro_step(dt, o.geom, self.r, self.u, self.m, self.mnode, self.rho, self.ee,
                                 self.ei, self.er, self.Zb, self.Ab_mu, self.rho0, self.Pd0, self.B0,
                                 self.solid, self.deg, self.Te, self.Ti, self.Pe, self.Pi, self.Pc,
                                 self.cs2, self.Cve, self.Cvi, self.q, o.C1, o.C2, o.P_ext, o.fix_outer)
        # 3. diffusion / coupling
        if o.cond_e or o.cond_i or o.exchange or o.radiation or P_abs > 0:
            Te0 = self.Te.copy(); Ti0 = self.Ti.copy(); Er0 = self.er * self.rho
            Te1 = np.empty(N); Ti1 = np.empty(N); Er1 = np.empty(N)
            lost = diffusion_step(dt, o.geom, self.r, self.m, self.rho, Te0, Ti0, Er0, self.Cve, self.Cvi,
                                  self.Zb, self.Z2b, self.Ab_mu, self.opac, self.Pdep, self.Sion,
                                  o.flux_limiter, o.cond_e, o.cond_i, o.exchange, o.radiation, True,
                                  o.kappa_mode, o.kappa0, o.kappa_n, o.n_picard, 1e-4, Te1, Ti1, Er1,
                                  o.rad_spectral)
            self.ee += self.Cve * (Te1 - Te0)
            self.ei += self.Cvi * (Ti1 - Ti0)
            if o.radiation:
                self.er = Er1 / self.rho
                self.E_rad_lost += lost
            self._eos()
        # 4. burn
        self.reac_rate = 0.0
        if o.burn and np.any(self.yDT > 0) and self.Ti.max() > 300.0:
            Sa = np.zeros(N)
            nreac = burn_step(self.rho, self.m, self.Ti, self.yDT, dt, Sa)
            self.reac_rate = nreac / dt
            self.N_reac += nreac
            self.E_fus += nreac * C.E_fusion
            self.E_neutron += nreac * C.E_neutron
            Ea = Sa.sum()
            if Ea > 0:
                if o.alpha_n_mu > 0:
                    de = np.zeros(N); di = np.zeros(N)
                    ne = self.Zb * self.rho / self.Ab_mu
                    ni = self.rho / self.Ab_mu
                    esc = alpha_transport_core(self.r, Sa, ne, self.Te, ni, self.Ti, self.Z2b, self.Ab_mu,
                                               self.mus, self.wmu, de, di)
                else:
                    de = Sa * 0.7; di = Sa * 0.3; esc = 0.0
                self.ee += de / self.m
                self.ei += di / self.m
                self.E_alpha_dep += de.sum() + di.sum()
                self.E_alpha_esc += esc
                self._eos()
        self.t += dt
        self.nstep += 1
        # 5. next dt
        dtc = cfl_dt(self.r, self.u, self.cs2, self.rho, self.er, self.q, o.C2, o.cfl)
        Tref = 1.0
        rel = max(np.max(np.abs(self.Te - Te_old) / (Te_old + Tref)),
                  np.max(np.abs(self.Ti - Ti_old) / (Ti_old + Tref)))
        relV = np.max(np.abs(self.m / self.rho - V_old) / V_old)
        f = min(1.25, o.dT_rel / max(rel, 1e-12), o.dV_rel / max(relV, 1e-12))
        self.dt = max(min(dtc, dt * f, o.dt_max), o.dt_min)
        if self.laser is not None and hasattr(self.laser.power, "t_ns"):
            # do not step over pulse breakpoints
            tb = self.laser.power.t_ns * 1e-9
            nxt = tb[tb > self.t + 1e-18]
            if len(nxt) and self.t + self.dt > nxt[0]:
                self.dt = max(nxt[0] - self.t, o.dt_min)
        return P_laser, P_abs

    def run(self, t_end, stop: Optional[Callable] = None, max_steps=10_000_000):
        o = self.o
        if self.laser is not None:
            self._R_beam0 = self.r[-1]
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
            if o.verbose and self.nstep % 5000 == 0:
                print(f"  step {self.nstep:8d} t={self.t*1e9:8.4f} ns dt={self.dt:.2e} "
                      f"rho_max={self.rho.max():8.2f} Ti_max={self.Ti.max()/1e3:7.3f} keV "
                      f"E_abs={self.E_abs/1e13:8.4f} MJ Yield={self.E_fus/1e13:9.4f} MJ "
                      f"Eerr={self.energy_error():+.2e}  ({_time.time()-t0:.0f}s)", flush=True)
            if stop is not None and stop(self):
                break
        self._record(0.0, 0.0)
        self.snapshot()
        self.wall = _time.time() - t0
        return self

    def _snap_dt(self):
        s = self.o.snap_dt
        return s(self.t) if callable(s) else s

    def history(self):
        return {k: np.array(v) for k, v in self.hist.items()}
