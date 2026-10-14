"""Run all SIFE verification benchmarks and write output/benchmarks/summary.json."""
from common import *
import json, time
import noh, sedov, conduction, laser_absorption, alpha_stopping, burn_efficiency, hotspot_ignition
import sedov2d, noh2d, acoustic2d, conduction2d, rt2d
from sife.fusion import sigv_DT

def bosch_hale():
    # Table VIII of Bosch & Hale (1992), DT <sigma v> in cm^3/s
    ref = {1: 6.857e-21, 2: 2.977e-19, 5: 1.366e-17, 10: 1.136e-16, 20: 4.330e-16, 50: 8.649e-16, 100: 8.445e-16}
    return [dict(T_keV=T, sife=sigv_DT(T), ref=v, rel_err=abs(sigv_DT(T) - v) / v) for T, v in ref.items()]

def default(o):
    try:
        return float(o)
    except Exception:
        return str(o)

res = {}
for name, f in [("bosch_hale", bosch_hale), ("noh", noh.run), ("sedov", sedov.run), ("conduction", conduction.run),
                ("laser_absorption", laser_absorption.run), ("alpha_stopping", alpha_stopping.run),
                ("burn_efficiency", burn_efficiency.run), ("hotspot_ignition", hotspot_ignition.run),
                ("sedov2d", sedov2d.run), ("noh2d", noh2d.run), ("acoustic2d", acoustic2d.run),
                ("conduction2d", conduction2d.run), ("rt2d", rt2d.run)]:
    t0 = time.time()
    res[name] = f()
    print(f"{name:18s} done in {time.time()-t0:6.1f} s", flush=True)
json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1, default=default)
print(json.dumps(res, indent=1, default=default))
