"""Parallel design scan: python scan.py scan.json  (list of parameter overrides)."""
import json, os, sys, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from krf_target import build, diagnostics, DEFAULT

def pulse(Pf, ts, tr, Pp, te, fmid=0.15):
    return dict(t_ns=[0.0, 0.2, ts, ts + 0.6 * tr, ts + tr, te, te + 0.2],
                P_TW=[0.0, Pf, Pf, fmid * Pp, Pp, Pp, 0.0])

def pulse_si(Pf, t1, Pm, t2, Ps, ts, ws, tr=0.1):
    """foot Pf until t1, ramp to compression power Pm at t2, held until the
    shock-ignition spike (rise tr to Ps at ts, flat ws), then off."""
    if Ps <= 0:
        return dict(t_ns=[0.0, 0.2, t1, t2, ts, ts + 0.1], P_TW=[0.0, Pf, Pf, Pm, Pm, 0.0])
    return dict(t_ns=[0.0, 0.2, t1, t2, ts - tr, ts, ts + ws, ts + ws + tr],
                P_TW=[0.0, Pf, Pf, Pm, Pm, Ps, Ps, 0.0])


def pulse_geo(Pf, t1, Pm, t2, te, n=10):
    """foot Pf until t1, geometric (exponential) ramp to Pm at t2, flat to te."""
    tr = list(np.linspace(t1, t2, n + 1))
    Pr = list(Pf * (Pm / Pf) ** np.linspace(0, 1, n + 1))
    return dict(t_ns=[0.0, 0.2] + tr[:] + [te, te + 0.2], P_TW=[0.0, Pf] + Pr + [Pm, 0.0])


def one(args):
    name, over = args
    p = dict(DEFAULT); p.update(over)
    if "pulse" in p:
        p.update(pulse(**p.pop("pulse")))
    if "pulse_geo" in p:
        p.update(pulse_geo(**p.pop("pulse_geo")))
    if "pulse_si" in p:
        p.update(pulse_si(**p.pop("pulse_si")))
    if p.get("no_burn"):
        pass
    sim = build(p, verbose=0)
    if p.get("no_burn"):
        sim.o.burn = False
    for k, v in p.get("opts", {}).items():
        setattr(sim.o, k, v)
    peak = {"r": 0.0}
    def stop(s):
        peak["r"] = max(peak["r"], s.reac_rate)
        return peak["r"] > 0 and s.reac_rate < 1e-3 * peak["r"] and s.E_fus > 1e9 and s.Ti.max() < 0.3 * max(s.hist["Ti_max"])
    try:
        sim.run(p["t_end_ns"] * 1e-9, stop=stop)
        d = diagnostics(sim)
    except Exception as e:
        d = {"error": repr(e)}
    d["name"] = name
    return name, p, d

if __name__ == "__main__":
    cases = json.load(open(sys.argv[1]))
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(HERE), "output", "scan.jsonl")
    with mp.Pool(min(4, len(cases))) as pool:
        for name, p, d in pool.imap_unordered(one, list(cases.items())):
            with open(out, "a") as f:
                f.write(json.dumps(dict(name=name, params=p, diag=d)) + "\n")
            keys = ["E_laser_MJ", "absorption", "v_imp_max_cm_s", "adiabat_inflight", "rhoR_fuel_max", "rho_max",
                    "Ti_burnavg_bang_keV", "rhoR_hs_bang", "R_hs_bang_um", "yield_MJ", "gain", "bang_time_ns", "hydro_eff", "wall_s"]
            print(name, {k: (round(d[k], 3) if isinstance(d.get(k), float) else d.get(k)) for k in keys}, d.get("error", ""), flush=True)
