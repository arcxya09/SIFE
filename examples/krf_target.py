"""KrF (248 nm) direct-drive hot-spot ignition target: design, run and diagnostics.

Target (inside out): DT vapour | cryogenic DT ice | DT-wetted CH foam ablator | thin CH skin.
Laser: shaped KrF pulse (foot -> ramp -> main drive), focal spot matched to the target
with optional zooming (beam radius follows the critical radius).
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
from sife import (Simulation, Options, Layer, Laser, tabulated_pulse, DT_gas, DT_ice,
                  CH, CH_foam_DT, constants as C)
from sife.eos import fermi_pressure_DT

# Baseline design (found with examples/scan.py): NRL-scale KrF target (outer radius
# 854 um, DT-wetted 100 mg/cc CH foam ablator, DT ice), conventional hot-spot ignition
# with an exponentially rising pulse (foot 0.8 TW -> 320 TW), 0.53 MJ.
_t_ramp = np.linspace(5.5, 9.0, 11)
_P_ramp = 0.8 * (320.0 / 0.8) ** np.linspace(0.0, 1.0, 11)
DEFAULT = dict(
    R_gas=0.0609,      # cm  inner radius of the DT ice (DT vapour inside)
    d_ice=0.0140,      # cm  cryogenic DT ice
    d_foam=0.0105,     # cm  DT-wetted CH foam (NRL-style KrF ablator)
    d_skin=0.0,        # cm  optional CH overcoat
    rho_foam=0.10,     # g/cc CH foam density (wetted with DT)
    rho_gas=2.0e-4,    # g/cc DT vapour
    zones=(40, 160, 100, 0),
    # pulse table: time [ns], power [TW]
    t_ns=[0.0, 0.2] + list(_t_ramp) + [10.0, 10.2],
    P_TW=[0.0, 0.8] + list(_P_ramp) + [320.0, 0.0],
    zoom=True, zoom_factor=1.0, beam_order=4.0, n_rays=64,
    t_end_ns=13.0,
)


def build(p, verbose=1):
    gas, ice, skin = DT_gas(p.get("rho_gas", 3.0e-4)), DT_ice(0.25), CH(1.05)
    foam = CH_foam_DT(p["rho_foam"], 0.25)
    nz = p["zones"]
    layers = [
        Layer(gas, p["R_gas"], nz[0], 1.0),
        Layer(ice, p["d_ice"], nz[1], fine_in=p.get("fine_ice_in", 1.06), fine_out=p.get("fine_ice_out", 1.03)),
        Layer(foam, p["d_foam"], nz[2], fine_in=1.03, fine_out=p.get("fine_foam_out", 1.04)),
        Layer(skin, p["d_skin"], nz[3], 1.0),
    ] if p["d_skin"] > 0 else [
        Layer(gas, p["R_gas"], nz[0], 1.0),
        Layer(ice, p["d_ice"], nz[1], fine_in=p.get("fine_ice_in", 1.06), fine_out=p.get("fine_ice_out", 1.03)),
        Layer(foam, p["d_foam"], nz[2], fine_in=1.03, fine_out=p.get("fine_foam_out", 1.04)),
    ]
    pulse = tabulated_pulse(p["t_ns"], p["P_TW"])
    R0 = p["R_gas"] + p["d_ice"] + p["d_foam"] + p["d_skin"]
    laser = Laser(wavelength_um=0.248, power=pulse, n_rays=p["n_rays"], profile_order=p["beam_order"])
    o = Options(verbose=verbose, dt_init=1e-13, dt_max=5e-12, snap_dt=lambda t: 0.05e-9 if t < 9.3e-9 else 0.005e-9, history_every=5,
                alpha_n_mu=12)
    sim = Simulation(layers, opts=o, laser=laser)
    if p["zoom"]:
        # zooming: focal spot radius follows the critical-surface radius (min of initial radius)
        def Rb(t, sim=sim):
            rcrit = sim.critical_radius(laser.nc) * p.get("zoom_factor", 1.0)
            return min(R0, max(rcrit, 0.25 * R0))
        laser.beam_radius = Rb
    else:
        laser.beam_radius = lambda t: R0
    return sim


def diagnostics(sim):
    h = sim.history()
    fuel = sim.fDT > 0.99
    d = {}
    d["E_laser_MJ"] = sim.E_laser / 1e13
    d["E_abs_MJ"] = sim.E_abs / 1e13
    d["absorption"] = sim.E_abs / max(sim.E_laser, 1e-30)
    d["yield_MJ"] = sim.E_fus / 1e13
    d["gain"] = sim.E_fus / max(sim.E_laser, 1e-30)
    d["neutrons"] = sim.N_reac
    i = int(np.argmax(h["v_imp"]))
    d["v_imp_max_cm_s"] = float(h["v_imp"][i]); d["t_vmax_ns"] = float(h["t"][i] * 1e9)
    d["KE_shell_max_kJ"] = float(h["KE_in"].max() / 1e10)
    d["hydro_eff"] = float(h["KE_in"].max() / max(sim.E_laser, 1e-30))
    j = int(np.argmax(h["rhoR_fuel"]))
    d["rhoR_fuel_max"] = float(h["rhoR_fuel"][j]); d["t_rhoR_max_ns"] = float(h["t"][j] * 1e9)
    d["rho_max"] = float(h["rho_max"].max())
    k = int(np.argmax(h["reac_rate"]))
    d["bang_time_ns"] = float(h["t"][k] * 1e9)
    d["Ti_hs_bang_keV"] = float(h["Ti_hs"][k] / 1e3)
    d["Ti_max_keV"] = float(h["Ti_max"].max() / 1e3)
    d["burnup"] = float(1 - np.sum(sim.m * sim.yDT) / np.sum(sim.m * sim.yDT0))
    d["energy_error"] = float(sim.energy_error())
    d["steps"] = sim.nstep; d["wall_s"] = getattr(sim, "wall", 0.0)
    # in-flight adiabat of the dense fuel at peak velocity (mass-averaged P/P_deg)
    s = min(sim.snaps, key=lambda s: abs(s["t"] - h["t"][i]))
    ice = (sim.mat == 1)
    ice = ice & (s["rho"] > 0.2 * s["rho"][ice].max())   # dense (unablated) fuel only
    a = s["P"][ice] / fermi_pressure_DT(s["rho"][ice])
    m = sim.m[ice]
    d["adiabat_inflight"] = float(np.sum(a * m) / np.sum(m))
    d["adiabat_inner"] = float(np.median(a[: len(a) // 3]))
    d["remaining_mass_frac"] = float(np.sum(sim.m[(s["rho"] > 0.5 * s["rho"].max()) | ice]) / np.sum(sim.m))
    # stagnation hot spot (at bang): contiguous central region with Ti > 0.3 Ti(centre)
    from sife.fusion import sigv_DT
    sb = min(sim.snaps, key=lambda s: abs(s["t"] - h["t"][k]))
    Ti = sb["Ti"]
    j = 0
    while j < len(Ti) - 1 and Ti[j + 1] > 0.3 * Ti[0]:
        j += 1
    hs = np.arange(j + 1)
    dr = np.diff(sb["r"])
    d["R_hs_bang_um"] = float(sb["r"][j + 1] * 1e4)
    d["rhoR_hs_bang"] = float(np.sum(sb["rho"][hs] * dr[hs]))
    d["rhoR_fuel_bang"] = float(np.sum((sb["rho"] * dr)[fuel]))
    d["P_hs_bang_Gbar"] = float(np.sum(sb["P"][hs] * sim.m[hs] / sb["rho"][hs]) / np.sum(sim.m[hs] / sb["rho"][hs]) / 1e15)
    wgt = np.array([sigv_DT(t / 1e3) for t in Ti]) * sb["rho"] * sim.m * (sim.fDT > 0.99)
    d["Ti_burnavg_bang_keV"] = float(np.sum(wgt * Ti) / max(np.sum(wgt), 1e-300) / 1e3)
    # gain (Q) figures
    from sife.qvalue import q_values
    fuel_nodes = np.zeros(len(sim.m) + 1, bool)
    fuel_nodes[1:][fuel] = True
    mn = np.zeros(len(sim.m) + 1); mn[:-1] += 0.5 * sim.m; mn[1:] += 0.5 * sim.m
    t_bang = h["t"][int(np.argmax(h["reac_rate"]))]
    KE_fuel = max(0.5 * np.sum((mn * np.minimum(s_["u"], 0.0) ** 2)[fuel_nodes])
                  for s_ in sim.snaps if s_["t"] <= t_bang)
    d["KE_fuel_max_kJ"] = float(KE_fuel / 1e10)
    d["Q"] = q_values(sim.E_fus, sim.E_laser, sim.E_abs, h["KE_in"].max(), KE_fuel)
    d["CR"] = float((sim.snaps[0]["r"][np.where(sim.mat == 1)[0][0]]) / max(d["R_hs_bang_um"] * 1e-4, 1e-9))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "krf"))
    ap.add_argument("--params", default=None, help="json overriding the defaults")
    ap.add_argument("--no-burn", action="store_true")
    a = ap.parse_args()
    p = dict(DEFAULT)
    if a.params:
        p.update(json.load(open(a.params)))
    os.makedirs(a.out, exist_ok=True)
    sim = build(p)
    if a.no_burn:
        sim.o.burn = False
    peak = {"r": 0.0}

    def stop(s):
        if s.reac_rate > peak["r"]:
            peak["r"] = s.reac_rate
        return peak["r"] > 0 and s.reac_rate < 1e-3 * peak["r"] and s.t > 1e-9 and s.E_fus > 1e9 and \
            s.hist["Ti_max"] and s.Ti.max() < 0.3 * max(s.hist["Ti_max"])
    sim.run(p["t_end_ns"] * 1e-9, stop=stop)
    d = diagnostics(sim)
    print(json.dumps(d, indent=1))
    from sife.qvalue import format_q
    print(format_q(d["Q"]))
    json.dump(dict(params=p, diagnostics=d), open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "history.npz"), **sim.history())
    keys = ["t", "r", "u", "rho", "Te", "Ti", "Tr", "P", "Pdep", "burn"]
    np.savez_compressed(os.path.join(a.out, "snapshots.npz"), mat=sim.mat, m=sim.m,
                        **{k: np.array([s[k] for s in sim.snaps]) for k in keys})
    return sim, d


if __name__ == "__main__":
    main()
