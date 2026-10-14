"""2D (axisymmetric) version of the KrF direct-drive ignition target.

Same capsule and pulse as examples/krf_target.py, on a polar Lagrangian mesh
(theta = 0 .. pi/2 with a reflecting equator by default).  Asymmetry seeds:

  --illum 2:0.02          laser intensity  I(theta) = 1 + 0.02 P2(cos theta)
  --pert 2:2:0.5          interface 2 (capsule surface) displaced by 0.5 um * P_l, l = 2
                          (interfaces: 0 gas/ice, 1 ice/ablator, 2 outer surface)

    python examples/krf_target_2d.py --out output/krf2d --illum 2:0.02
    python -m sife.animate output/krf2d
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import numpy as np
from sife import Layer, Laser, tabulated_pulse, DT_gas, DT_ice, CH_foam_DT
from sife.simulation2d import Simulation2D, Options2D
from sife.qvalue import q_values, format_q
from sife.fusion import sigv_DT
from krf_target import DEFAULT


def build(p, n_theta=16, half=True, illum=None, pert=None, zone_scale=1.0, verbose=1, core=-1):
    gas, ice = DT_gas(p.get("rho_gas", 3.0e-4)), DT_ice(0.25)
    foam = CH_foam_DT(p["rho_foam"], 0.25)
    nz = [max(4, int(round(n * zone_scale))) for n in p["zones"][:3]]
    layers = [
        Layer(gas, p["R_gas"], nz[0], 1.0),
        Layer(ice, p["d_ice"], nz[1], fine_in=p.get("fine_ice_in", 1.06) ** (1 / zone_scale),
              fine_out=p.get("fine_ice_out", 1.03) ** (1 / zone_scale)),
        Layer(foam, p["d_foam"], nz[2], fine_in=1.03 ** (1 / zone_scale),
              fine_out=p.get("fine_foam_out", 1.04) ** (1 / zone_scale)),
    ]
    if core < 0:      # default: the whole gas region is the quasi-1D (isobaric) core
        core = nz[0]
    pulse = tabulated_pulse(p["t_ns"], p["P_TW"])
    R0 = p["R_gas"] + p["d_ice"] + p["d_foam"]
    laser = Laser(wavelength_um=0.248, power=pulse, n_rays=p["n_rays"], profile_order=p["beam_order"])
    snap = lambda t: 0.1e-9 if t < 9.3e-9 else 0.01e-9
    o = Options2D(verbose=verbose, dt_init=1e-13, dt_max=5e-12, snap_dt=snap, history_every=5,
                  alpha_n_mu=8, core_rings=core)
    sim = Simulation2D(layers, n_theta=n_theta, half=half, opts=o, laser=laser,
                       illumination=illum, perturbations=pert)
    if p["zoom"]:
        def Rb(t, sim=sim):
            rcrit = sim.critical_radius(laser.nc) * p.get("zoom_factor", 1.0)
            return min(R0, max(rcrit, 0.25 * R0))
        laser.beam_radius = Rb
    else:
        laser.beam_radius = lambda t: R0
    return sim


def diagnostics(sim):
    h = sim.history()
    d = {}
    d["E_laser_MJ"] = sim.E_laser / 1e13
    d["E_abs_MJ"] = sim.E_abs / 1e13
    d["absorption"] = sim.E_abs / max(sim.E_laser, 1e-30)
    d["yield_MJ"] = sim.E_fus / 1e13
    d["neutrons"] = sim.N_reac
    i = int(np.argmax(h["v_imp"]))
    d["v_imp_max_cm_s"] = float(h["v_imp"][i]); d["t_vmax_ns"] = float(h["t"][i] * 1e9)
    j = int(np.argmax(h["rhoR_fuel"]))
    d["rhoR_fuel_max"] = float(h["rhoR_fuel"][j])
    d["rhoR_fuel_min_over_angle_at_max"] = float(h["rhoR_fuel_min"][j])
    d["rhoR_fuel_max_over_angle_at_max"] = float(h["rhoR_fuel_max"][j])
    k = int(np.argmax(h["reac_rate"]))
    d["bang_time_ns"] = float(h["t"][k] * 1e9)
    d["Ti_hs_bang_keV"] = float(h["Ti_hs"][k] / 1e3)
    d["Ti_max_keV"] = float(h["Ti_max"].max() / 1e3)
    d["burnup"] = float(1 - np.sum(sim.m * sim.yDT) / np.sum(sim.m * sim.yDT0))
    d["energy_error"] = float(sim.energy_error())
    d["steps"] = sim.nstep; d["wall_s"] = sim.wall
    if "ice_in_P2" in h:
        d["ice_inner_P2_at_bang"] = float(h["ice_in_P2"][k]); d["ice_inner_P4_at_bang"] = float(h["ice_in_P4"][k])
        d["R_ice_inner_bang_um"] = float(h["R_ice_in"][k] * 1e4)
    fuel_nodes = np.zeros(sim.z.shape, bool)
    d["Q"] = q_values(sim.E_fus, sim.E_laser, sim.E_abs, float(h["KE_in"].max()))
    return d


def parse_modes(items, n):
    out = []
    for s in items or []:
        v = s.split(":")
        out.append(tuple(int(x) if i < n - 1 else float(x) for i, x in enumerate(v)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "krf2d"))
    ap.add_argument("--params", default=None)
    ap.add_argument("--nth", type=int, default=16, help="angular zones")
    ap.add_argument("--full", action="store_true", help="simulate 0..pi (no equatorial symmetry)")
    ap.add_argument("--zone-scale", type=float, default=1.0, help="radial resolution factor vs 1D design")
    ap.add_argument("--illum", nargs="*", help="l:a  laser intensity Legendre modes")
    ap.add_argument("--pert", nargs="*", help="interface:l:amplitude_um  surface perturbations")
    ap.add_argument("--core", type=int, default=-1,
                    help="innermost rings treated as a quasi-1D isobaric core (default -1: all gas rings)")
    ap.add_argument("--hg", type=float, default=None, help="anti-hourglass stiffness (default Options2D.hourglass)")
    ap.add_argument("--no-burn", action="store_true")
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--wall", type=float, default=None, help="wall-clock limit [s]")
    a = ap.parse_args()
    p = dict(DEFAULT)
    if a.params:
        p.update(json.load(open(a.params)))
    illum = {l: amp for l, amp in parse_modes(a.illum, 2)}
    pert = [(k, l, amp * 1e-4) for k, l, amp in parse_modes(a.pert, 3)]
    os.makedirs(a.out, exist_ok=True)
    sim = build(p, a.nth, not a.full, illum, pert, a.zone_scale, core=a.core)
    print(f"2D mesh {sim.Ni} x {sim.Nj} zones, illumination {illum}, perturbations {pert}", flush=True)
    if a.no_burn:
        sim.o.burn = False
    if a.hg is not None:
        sim.o.hourglass = a.hg
    peak = {"r": 0.0}

    def stop(s):
        if s.reac_rate > peak["r"]:
            peak["r"] = s.reac_rate
        return peak["r"] > 0 and s.reac_rate < 1e-3 * peak["r"] and s.E_fus > 1e9 and \
            s.hist.get("Ti_max") and s.Ti.max() < 0.3 * max(s.hist["Ti_max"])
    sim.run((a.t_end or p["t_end_ns"]) * 1e-9, stop=stop, wall_limit=a.wall)
    d = diagnostics(sim)
    d["mesh"] = [sim.Ni, sim.Nj]; d["core_rings"] = sim.o.core_rings; d["half"] = bool(sim.half); d["illumination"] = illum
    d["perturbations_cm"] = pert
    print(json.dumps(d, indent=1))
    print(format_q(d["Q"]))
    json.dump(dict(params=p, diagnostics=d), open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "history.npz"), **sim.history())
    sim.save(os.path.join(a.out, "snapshots2d.npz"))
    return sim, d


if __name__ == "__main__":
    main()
